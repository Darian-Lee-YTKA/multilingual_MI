import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch.nn.functional as F
import matplotlib
import matplotlib.pyplot as plt

# Device and model setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_name = "bigscience/bloom-3b"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
    device_map="auto",
    output_hidden_states=True
).to(device)
model.eval()

# Data
print("Tokenizer examples:")
print(tokenizer.tokenize("ihm"))  # dative masculine
print(tokenizer.tokenize("ihn"))  # accusative masculine
print(tokenizer.tokenize("ihr"))  # dative feminine
print(tokenizer.tokenize("seiner"))  # genitive masculine

# Example sentences and pronouns
true = [
    "Ich sehe",  # I see him in the park (accusative)
    "Ich antworte"]

true_case = ["ihn", "ihr"]  # Correct case forms
corrupted_acc = ["er", "sie"]  # Nominative forms (incorrect)
corrupted = [
    "Ich gebe",  # Incorrect: using nominative instead of accusative
    "Ich frage"]

# Global variables
clean_attn_output = []
corrupt_attn_output = []
attn_weights_storage = []

def get_attention_weights(prompt, storage):
    """Get attention weights and hidden states for a prompt"""
    storage.clear()
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True)
    
    # Get last hidden states
    last_hidden = outputs.hidden_states[-1].detach().cpu()
    storage.append(last_hidden)
    
    # Get attention weights from last layer
    attentions = outputs.attentions
    last_layer_attn = attentions[-1].detach().cpu()
    storage.append(last_layer_attn)
    
    return inputs

def print_topk_logits(hidden, k=5):
    """Print top k predictions from hidden states"""
    logits = model.lm_head(hidden[0, -1])
    topk = torch.topk(logits, k)
    tokens = tokenizer.batch_decode(topk.indices)
    print("🔝 Top-5 predictions:")
    for token, score in zip(tokens, topk.values):
        print(f"   {token.strip():>10} ({score.item():.2f})")

def get_attention_output(prompt, storage):
    """Get attention output for a prompt"""
    storage.clear()
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    def hook(module, input, output):
        storage.append(output[0].detach())

    # BLOOM uses self_attention instead of attn
    handle = model.transformer.h[-1].self_attention.register_forward_hook(hook)
    with torch.no_grad():
        _ = model(**inputs)
    handle.remove()
    return inputs

def get_logits_diff_by_category(hidden, positive_words, negative_words, top_k=1000):
    """Calculate logit difference between positive and negative words"""
    logits = model.lm_head(hidden[0, -1])
    topk = torch.topk(logits, top_k)
    tokens = tokenizer.batch_decode(topk.indices)

    top_pos = float('-inf')
    top_neg = float('-inf')
    best_pos = None
    best_neg = None

    # Print tokenization of target words for debugging
    print("\n🔍 Target word tokenization:")
    for word in positive_words + negative_words:
        tokens = tokenizer.tokenize(word)
        print(f"'{word}' -> {tokens}")

    # Create sets of tokenized forms for each word
    positive_tokenized = {word: set(tokenizer.tokenize(word)) for word in positive_words}
    negative_tokenized = {word: set(tokenizer.tokenize(word)) for word in negative_words}

    for logit, token in zip(topk.values, tokens):
        stripped_token = token.strip()
        token_tokens = set(tokenizer.tokenize(stripped_token))
        
        # Check if token matches any of the target words' tokenized forms
        for word, word_tokens in positive_tokenized.items():
            if token_tokens == word_tokens:
                if logit > top_pos:
                    top_pos = logit.item()
                    best_pos = stripped_token
                    break
        
        for word, word_tokens in negative_tokenized.items():
            if token_tokens == word_tokens:
                if logit > top_neg:
                    top_neg = logit.item()
                    best_neg = stripped_token
                    break

    if best_pos is None:
        print(f"⚠️ None of {positive_words} found in top-{top_k}")
        # Use a small negative value instead of -inf
        top_pos = -10.0
    if best_neg is None:
        print(f"⚠️ None of {negative_words} found in top-{top_k}")
        # Use a small negative value instead of -inf
        top_neg = -10.0

    print(f"✅ Top positive: {best_pos} ({top_pos:.2f})")
    print(f"❌ Top negative: {best_neg} ({top_neg:.2f})")

    return top_pos - top_neg

def get_logits_from_hidden(hidden, target_words, top_k=1000):
    """Get logits for target words from hidden states"""
    logits = model.lm_head(hidden[0, -1])
    topk = torch.topk(logits, top_k)
    top_probs = torch.softmax(topk.values, dim=-1)
    top_tokens = tokenizer.batch_decode(topk.indices)

    result = []
    for word in target_words:
        found = False
        # Get the tokenized form of the target word
        target_tokens = tokenizer.tokenize(word)
        target_token_str = "".join(target_tokens)
        
        for prob, token in zip(top_probs, top_tokens):
            token = token.strip()
            # Check if the token matches the joined target tokens
            if token == target_token_str:
                result.append(prob.item())
                found = True
                break
        if not found:
            print(f"⚠️ Word '{word}' (tokenized as '{target_token_str}') not found in top-{top_k} tokens")
            result.append(0.0)
    return result

def get_last_hidden(prompt):
    """Get last hidden state for a prompt"""
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    return outputs.hidden_states[-1]

def patch_single_head(orig, corrupt, head_idx, head_size):
    """Patch a single attention head"""
    patched = orig.clone()
    start = head_idx * head_size
    end = (head_idx + 1) * head_size
    patched[0, -1, start:end] = corrupt[0, -1, start:end]
    return patched

def run_with_patched_attn(patched_attn, prompt, target_words):
    """Run model with patched attention"""
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    def hook(module, input, output):
        return (patched_attn, ) + output[1:]

    # BLOOM uses self_attention instead of attn
    handle = model.transformer.h[-1].self_attention.register_forward_hook(hook)
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
        hidden = outputs.hidden_states[-1]
        logits = get_logits_from_hidden(hidden, target_words)
        print_topk_logits(hidden)
    handle.remove()
    
    return logits

def plot_attention(attn_weights, head_idx, tokens, title=None):
    """Plot attention weights for a specific head"""
    if isinstance(attn_weights, tuple):
        attn_weights = attn_weights[0]
    
    print(attn_weights.shape) 
    print(attn_weights[head_idx, :].shape)
    # Get the attention weights for the specific head
    attn = attn_weights[head_idx, :].cpu().numpy()

    plt.figure(figsize=(8, 6))
    plt.imshow(attn, cmap="viridis")
    plt.colorbar()
    plt.xticks(ticks=range(len(tokens)), labels=tokens, rotation=90)
    plt.yticks(ticks=range(len(tokens)), labels=tokens)
    if title:
        plt.title(title)
    plt.tight_layout()
    plt.show()

def analyze_case_specific_attention(attn_weights, tokens, case_type):
    """Analyze attention patterns specific to grammatical cases"""
    # Print shape for debugging
    print(f"Attention weights shape: {attn_weights.shape}")
    
    # Get attention weights for the last token (pronoun position)
    # BLOOM's attention shape is [num_heads, seq_len, seq_len]
    last_token_attn = attn_weights[:, -1, :].mean(dim=0)  # Average across heads
    
    # Find the verb position (usually second token in German)
    verb_pos = 1
    
    # Calculate attention to verb
    verb_attn = last_token_attn[verb_pos]
    
    print(f"\n🔍 Case Analysis for {case_type}:")
    print(f"Attention to verb: {verb_attn:.4f}")
    
    # Find attention to other case-marking elements
    for i, token in enumerate(tokens):
        if token in ["an", "mit", "für", "bei"]:  # Prepositions that require specific cases
            prep_attn = last_token_attn[i]
            print(f"Attention to preposition '{token}': {prep_attn:.4f}")

def compare_attention_patterns(clean_attn, corrupt_attn, tokens):
    """Compare attention patterns between correct and incorrect cases"""
    # Get attention weights for the last token
    # Average across heads for both clean and corrupt
    clean_last = clean_attn[:, -1, :].mean(dim=0)  # [seq_len]
    corrupt_last = corrupt_attn[:, -1, :].mean(dim=0)  # [seq_len]
    
    # Calculate difference
    attn_diff = clean_last - corrupt_last
    
    print("\n📊 Attention Pattern Comparison:")
    for i, token in enumerate(tokens):
        if attn_diff[i] > 0.1:  # Significant positive difference
            print(f"More attention to '{token}' in correct case: {attn_diff[i]:.4f}")
        elif attn_diff[i] < -0.1:  # Significant negative difference
            print(f"Less attention to '{token}' in correct case: {attn_diff[i]:.4f}")

# Main analysis loop
n_heads = model.config.n_head
head_size = model.config.hidden_size // n_heads

for i, (clean_prompt, corrupt_prompt, correct, wrong) in enumerate(zip(true, corrupted, true_case, corrupted_acc)):
    print(f"\n🧪 Example {i+1}: {clean_prompt}")

    # Get attention outputs
    clean_inputs = get_attention_output(clean_prompt, clean_attn_output)
    corrupt_inputs = get_attention_output(corrupt_prompt, corrupt_attn_output)
    clean_attn = clean_attn_output[0]
    corrupt_attn = corrupt_attn_output[0]

    # Analyze case-specific attention
    tokens = clean_prompt.split()
    analyze_case_specific_attention(clean_attn, tokens, "Correct case")
    analyze_case_specific_attention(corrupt_attn, tokens, "Incorrect case")
    
    # Compare attention patterns
    compare_attention_patterns(clean_attn, corrupt_attn, tokens)

    # Logits without patching
    with torch.no_grad():
        clean_hidden = get_last_hidden(clean_prompt)
        clean_logits = get_logits_from_hidden(clean_hidden, [correct, wrong])
        clean_diff = get_logits_diff_by_category(clean_hidden, true_case, corrupted_acc)
        print(f"✅ Without patch: {correct} = {clean_logits[0]:.4f}, {wrong} = {clean_logits[1]:.4f} → diff = {clean_diff:.4f}")
        print_topk_logits(clean_hidden)

    # Iterate through heads
    head_deltas = []

    for head in range(n_heads):
        patched = patch_single_head(clean_attn, corrupt_attn, head, head_size)
        patched_logits = run_with_patched_attn(patched, clean_prompt, [correct, wrong])
        patched_diff = get_logits_diff_by_category(patched, true_case, corrupted_acc)
        delta = patched_diff - clean_diff

        head_deltas.append((head, delta))
        print(f"🔁 Head {head:2d}: Patched diff = {patched_diff:.4f}, Δ = {delta:+.4f}")
        print("difference from clean diff: ", patched_diff - clean_diff)

    # Sort heads by impact
    head_deltas.sort(key=lambda x: abs(x[1]), reverse=True)
    print("\n📊 🧠 printing for prompt: ", clean_prompt, corrupt_prompt)
    for rank, (head, delta) in enumerate(head_deltas, 1):
        print(f"   {rank}. Head {head:2d}: Δ = {delta:+.4f}")
    
    get_attention_weights(clean_prompt, attn_weights_storage)
    clean_attn_weights = attn_weights_storage[0]
    tokens = clean_prompt.split(" ")
    for i in range(4):
        head, delta = head_deltas[i]
        print(f"Visualizing attention for head {head} with Δ={delta:.4f}")
        #plot_attention(clean_attn_weights, head, tokens, title=f"Clean prompt attention") 
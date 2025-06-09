import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch.nn.functional as F
import matplotlib
import matplotlib.pyplot as plt
import os
# Устройство и модель
device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
model_name = "sberbank-ai/rugpt3large_based_on_gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, output_hidden_states=True).to(device)
model.eval()

# Данные
print("tokeizer")
print(tokenizer.tokenize("своей"))
print(tokenizer.tokenize("тебя"))
print(tokenizer.tokenize("его"))
print(tokenizer.tokenize("своим"))

true = ["она дорожит", "я встретилась с"]
true_case = ["парнем", "своей", "своим", "ним", "тем", "мужчиной", "девушкой", "человеком", "тобой", "им", "этим", "вами"]
corrupted_acc = ["тебя", "его", "того", "вопроса", "меня", "малого", "себя"]
corrupted = ["она обожает", "я начала с"]

# Глобальные переменные
clean_attn_output = []
corrupt_attn_output = []

attn_weights_storage = []

def get_attention_weights(prompt, storage):
    storage.clear()
    start_token_id = tokenizer.bos_token_id
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    if start_token_id is not None:
        start_token_tensor = torch.tensor([[start_token_id]], device=device)
        input_ids = torch.cat([start_token_tensor, inputs["input_ids"]], dim=1)

        if "attention_mask" in inputs:
            start_mask = torch.ones((inputs["attention_mask"].shape[0], 1), device=device, dtype=inputs["attention_mask"].dtype)
            attention_mask = torch.cat([start_mask, inputs["attention_mask"]], dim=1)
        else:
            attention_mask = torch.ones_like(input_ids)
    else:
        input_ids = inputs["input_ids"]
        attention_mask = inputs.get("attention_mask", torch.ones_like(input_ids))
    print("LEN INPUTS:", len(inputs))
    print("MI_INPUTS:,", inputs)
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, output_attentions=True)

    attentions = outputs.attentions  
    last_layer_attn = attentions[-1].detach().cpu()  
    storage.append(last_layer_attn)



def print_topk_logits(hidden, k=5):
    logits = model.lm_head(hidden[0, -1])
    topk = torch.topk(logits, k)
    tokens = tokenizer.batch_decode(topk.indices)
    print("🔝 Top-5 predictions:")
    for token, score in zip(tokens, topk.values):
        print(f"   {token.strip():>10} ({score.item():.2f})")


def get_attention_output(prompt, storage):
    storage.clear()
    start_token_id = tokenizer.bos_token_id
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    if start_token_id is not None:
        start_token_tensor = torch.tensor([[start_token_id]], device=device)
        input_ids = torch.cat([start_token_tensor, inputs["input_ids"]], dim=1)

        if "attention_mask" in inputs:
            start_mask = torch.ones((inputs["attention_mask"].shape[0], 1), device=device, dtype=inputs["attention_mask"].dtype)
            attention_mask = torch.cat([start_mask, inputs["attention_mask"]], dim=1)
        else:
            attention_mask = torch.ones_like(input_ids)
    else:
        input_ids = inputs["input_ids"]
        attention_mask = inputs.get("attention_mask", torch.ones_like(input_ids))

    def hook(module, input, output):
        # grabs attention output and stores it in storage 
        
        atten =  output[0].detach()
        print("shape of attention output!: ", atten.shape)
        storage.append(atten)

    # attachs hook to the last layer of attention (identified as important through logit lens)
    handle = model.transformer.h[-1].attn.register_forward_hook(hook)
    with torch.no_grad():
        # run foward pass. hook grabs the attention
        _ = model(input_ids=input_ids, attention_mask=attention_mask)
    # remove hook
    handle.remove()
    return inputs




def get_logits_diff_by_category(hidden, positive_words, negative_words, top_k=10000):
    logits = model.lm_head(hidden[0, -1])
    topk = logits.topk(k=top_k, dim=-1) 
    tokens = tokenizer.batch_decode(topk.indices)

    top_pos = float('-inf')
    top_neg = float('-inf')
    best_pos = None
    best_neg = None

    print(positive_words)
    print(negative_words)
    for logit, token in zip(topk.values, tokens):
        stripped_token = token.strip()

        
        if stripped_token in positive_words and logit > top_pos:

            top_pos = logit.item()
            best_pos = stripped_token
        elif stripped_token in negative_words and logit > top_neg:
            top_neg = logit.item()
            best_neg = stripped_token

    if best_pos is None:
        print(f"⚠️ Ни одно слово из {positive_words} не найдено в топ-{top_k}")
    if best_neg is None:
        print(f"⚠️ Ни одно слово из {negative_words} не найдено в топ-{top_k}")

    print(f"✅ Top positive: {best_pos} ({top_pos:.2f})")
    print(f"❌ Top negative: {best_neg} ({top_neg:.2f})")

    return top_pos - top_neg



# Получение логитов по последнему скрытому состоянию
def get_logits_from_hidden(hidden, target_words, top_k=10000):
    logits = model.lm_head(hidden[0, -1])  # logits для последнего токена
    topk = torch.topk(logits, top_k)  # Получаем топ top_k логитов и индексов
    top_probs = torch.softmax(topk.values, dim=-1)  # Преобразуем логиты в вероятности
    top_tokens = tokenizer.batch_decode(topk.indices)  # Декодируем индексы в токены

    result = []
    top_pos = -999999
    best_pos = None 
    for logit, token in zip(topk.values, top_tokens):
        stripped_token = token.strip()

        
        if stripped_token in target_words and logit > top_pos:

            top_pos = logit.item()
            best_pos = stripped_token

    result = best_pos
    return result

# Получение последнего скрытого состояния
def get_last_hidden(prompt):
    start_token_id = tokenizer.bos_token_id
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    if start_token_id is not None:
        start_token_tensor = torch.tensor([[start_token_id]], device=device)
        input_ids = torch.cat([start_token_tensor, inputs["input_ids"]], dim=1)

        if "attention_mask" in inputs:
            start_mask = torch.ones((inputs["attention_mask"].shape[0], 1), device=device, dtype=inputs["attention_mask"].dtype)
            attention_mask = torch.cat([start_mask, inputs["attention_mask"]], dim=1)
        else:
            attention_mask = torch.ones_like(input_ids)
    else:
        input_ids = inputs["input_ids"]
        attention_mask = inputs.get("attention_mask", torch.ones_like(input_ids))
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
    return outputs.hidden_states[-1]

# Подмена одной attention головы
def patch_single_head(orig, corrupt, head_idx, head_size):
    patched = orig.clone()
    start = head_idx * head_size
    end = (head_idx + 1) * head_size
    patched[0, -1, start:end] = corrupt[0, -1, start:end]
    return patched

# Запуск модели с подменённым attention выходом
def run_with_patched_attn(patched_attn, prompt, target_words):
    start_token_id = tokenizer.bos_token_id
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    if start_token_id is not None:
        start_token_tensor = torch.tensor([[start_token_id]], device=device)
        input_ids = torch.cat([start_token_tensor, inputs["input_ids"]], dim=1)

        if "attention_mask" in inputs:
            start_mask = torch.ones((inputs["attention_mask"].shape[0], 1), device=device, dtype=inputs["attention_mask"].dtype)
            attention_mask = torch.cat([start_mask, inputs["attention_mask"]], dim=1)
        else:
            attention_mask = torch.ones_like(input_ids)
    else:
        input_ids = inputs["input_ids"]
        attention_mask = inputs.get("attention_mask", torch.ones_like(input_ids))

    def hook(module, input, output):
        print("printing patched attention!")
        print(patched_attn.shape)
        return (patched_attn, ) + output[1:]


    handle = model.transformer.h[-1].attn.register_forward_hook(hook)
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        hidden = outputs.hidden_states[-1]
        logits = get_logits_from_hidden(hidden, target_words)
        print_topk_logits(hidden)  # ← здесь
    handle.remove()
    
    return logits


def plot_attention(attn_weights, head_idx, tokens, title=None, save_dir="attn_images"):
    # Если attn_weights — кортеж, берем первый элемент
    if isinstance(attn_weights, tuple):
        attn_weights = attn_weights[0]

    print(tokens)
    
    if tokens == ["она", "дорожит"]:
        tokens = ["она", "доро", "жит"]
    tokens = ["<BOS>"] + tokens
    
    print(attn_weights.shape)  # [batch_size, num_heads, seq_len, seq_len]
    attn = attn_weights[0, head_idx].cpu().numpy()  # shape: [seq_len, seq_len]
    print(attn.shape)

    plt.figure(figsize=(8, 6))
    plt.imshow(attn, cmap="viridis")
    plt.colorbar()
    plt.xticks(ticks=range(len(tokens)), labels=tokens, rotation=90)
    plt.yticks(ticks=range(len(tokens)), labels=tokens)
    if title:
        plt.title(title)
    plt.tight_layout()

    # Создаем директорию, если её нет
    os.makedirs(save_dir, exist_ok=True)

    # Формируем имя файла
    token_str = "_".join(tokens).replace("/", "_").replace(" ", "_")[:100]
    filename = f"head{head_idx}_{token_str}.png"
    filepath = os.path.join(save_dir, filename)
    
    plt.savefig(filepath, dpi=300)
    print(f"Saved to {filepath}")
    plt.show()





# Основной цикл
n_heads = model.config.n_head
head_size = model.config.n_embd // n_heads




for i, (clean_prompt, corrupt_prompt, correct, wrong) in enumerate(zip(true, corrupted, true_case, corrupted_acc)):
    print(f"\n🧪 Пример {i+1}: {clean_prompt}")

    print("/n/n CLEAN PROMPT: ", clean_prompt)
    print("CORRUPT PROMPT: ", corrupt_prompt)


    # Получаем attention-выходы
    clean_inputs = get_attention_output(clean_prompt, clean_attn_output) #puts the attention in storage
    print("clean_inputs: ", clean_inputs)
    corrupt_inputs = get_attention_output(corrupt_prompt, corrupt_attn_output)
    clean_attn = clean_attn_output[0]  # [1, seq_len, emb_dim]
    corrupt_attn = corrupt_attn_output[0]

    # Логиты без патчинга
    with torch.no_grad():
        clean_hidden = get_last_hidden(clean_prompt)
        clean_diff = get_logits_diff_by_category(clean_hidden, true_case, corrupted_acc)
        print(f"✅ Без патча: {correct} diff = {clean_diff:.4f}")
        print_topk_logits(clean_hidden)  # 🔍 вот тут


    # Перебор голов
    head_deltas = []


    for head in range(n_heads):
        print("clean_atten shape!, ", clean_attn.shape)
        patched = patch_single_head(clean_attn, corrupt_attn, head, head_size)
        patched_diff = get_logits_diff_by_category(patched, true_case, corrupted_acc)
        
        delta = patched_diff - clean_diff

        head_deltas.append((head, delta))
        print(f"🔁 Head {head:2d}: Patched diff = {patched_diff:.4f}, Δ = {delta:+.4f}")
        print("difference from clean diff: ", patched_diff - clean_diff)

    # Сортировка голов по влиянию
    
    head_deltas.sort(key=lambda x: abs(x[1]), reverse=True)
    print("\nprinting for prompt: ", clean_prompt, corrupt_prompt)
    for rank, (head, delta) in enumerate(head_deltas, 1):
        print(f"   {rank}. Head {head:2d}: Δ = {delta:+.4f}")
    
    get_attention_weights(clean_prompt, attn_weights_storage)
    clean_attn_weights = attn_weights_storage[0]
    print("clean attn_weights")
    print(clean_attn_weights.shape)

    tokens = clean_prompt.split(" ")
    for i in range(3):
        if i == 0: 
            t = "best"
        if i == 1:
            t = "second best"
        if i == 2:
            t = "third best"
        head, delta = head_deltas[i]
        print(f"Визуализация внимания для головы {head} с Δ={delta:.4f}")
        title = "Clean prompt attention for " + t + "head: " + str(head)
        plot_attention(clean_attn_weights, head, tokens, title=title)

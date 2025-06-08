import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch.nn.functional as F
import matplotlib
import matplotlib.pyplot as plt
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
true_case = ["своей", "своим", "ним", "тем"]
corrupted_acc = ["тебя", "его", "того", "вопроса"]
corrupted = ["она любит", "я начала с"]

# Глобальные переменные
clean_attn_output = []
corrupt_attn_output = []

attn_weights_storage = []

def get_attention_weights(prompt, storage):
    storage.clear()
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True)
    
    # Получаем последние скрытые состояния
    last_hidden = outputs.hidden_states[-1].detach().cpu()  # [batch, seq_len, embed_dim]
    storage.append(last_hidden)
    
    # Получаем веса внимания последнего слоя
    attentions = outputs.attentions  # список тензоров по слоям
    last_layer_attn = attentions[-1].detach().cpu()  # [batch, heads, seq_len, seq_len]
    storage.append(last_layer_attn)
    
    return inputs


def print_topk_logits(hidden, k=5):
    logits = model.lm_head(hidden[0, -1])
    topk = torch.topk(logits, k)
    tokens = tokenizer.batch_decode(topk.indices)
    print("🔝 Top-5 predictions:")
    for token, score in zip(tokens, topk.values):
        print(f"   {token.strip():>10} ({score.item():.2f})")

# Получение выходов attention слоя
def get_attention_output(prompt, storage):
    storage.clear()
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    def hook(module, input, output):
        storage.append(output[0].detach())

    handle = model.transformer.h[-1].attn.register_forward_hook(hook)
    with torch.no_grad():
        _ = model(**inputs)
    handle.remove()
    return inputs




def get_logits_diff_by_category(hidden, positive_words, negative_words, top_k=1000):
    logits = model.lm_head(hidden[0, -1])
    topk = torch.topk(logits, top_k)
    tokens = tokenizer.batch_decode(topk.indices)

    top_pos = float('-inf')
    top_neg = float('-inf')
    best_pos = None
    best_neg = None

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
def get_logits_from_hidden(hidden, target_words, top_k=1000):
    logits = model.lm_head(hidden[0, -1])  # logits для последнего токена
    topk = torch.topk(logits, top_k)  # Получаем топ top_k логитов и индексов
    top_probs = torch.softmax(topk.values, dim=-1)  # Преобразуем логиты в вероятности
    top_tokens = tokenizer.batch_decode(topk.indices)  # Декодируем индексы в токены

    result = []
    for word in target_words:
        found = False
        word = " " + word
        for prob, token in zip(top_probs, top_tokens):
            if token == word or token == word[1:]:
                result.append(prob.item())
                found = True
                break
            if word == " тебя":
                if token == "его" or token == " его": # его might be more likely incorrect than тебя
                    result.append(prob.item())
                    found = True
                    break
        if not found:
            print(f"⚠️ Слово '{word}' и слово '{word[1:]}' НЕ найдено среди топ-{top_k} токенов")
            result.append(0.0)
    return result

# Получение последнего скрытого состояния
def get_last_hidden(prompt):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
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
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    def hook(module, input, output):
        return (patched_attn, ) + output[1:]


    handle = model.transformer.h[-1].attn.register_forward_hook(hook)
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
        hidden = outputs.hidden_states[-1]
        logits = get_logits_from_hidden(hidden, target_words)
        print_topk_logits(hidden)  # ← здесь
    handle.remove()
    
    return logits


def plot_attention(attn_weights, head_idx, tokens, title=None):
    # Если attn_weights — кортеж, берем первый элемент
    if isinstance(attn_weights, tuple):
        attn_weights = attn_weights[0]
    
    # attn_weights shape: [batch_size, num_heads, seq_len, seq_len]
    print(attn_weights.shape)
    attn = attn_weights[0, head_idx].cpu().numpy()  # shape: [seq_len, seq_len]

    plt.figure(figsize=(8, 6))
    plt.imshow(attn, cmap="viridis")
    plt.colorbar()
    plt.xticks(ticks=range(len(tokens)), labels=tokens, rotation=90)
    plt.yticks(ticks=range(len(tokens)), labels=tokens)
    if title:
        plt.title(title)
    plt.tight_layout()
    plt.show()





# Основной цикл
n_heads = model.config.n_head
head_size = model.config.n_embd // n_heads




for i, (clean_prompt, corrupt_prompt, correct, wrong) in enumerate(zip(true, corrupted, true_case, corrupted_acc)):
    print(f"\n🧪 Пример {i+1}: {clean_prompt}")

    # Получаем attention-выходы
    clean_inputs = get_attention_output(clean_prompt, clean_attn_output)
    corrupt_inputs = get_attention_output(corrupt_prompt, corrupt_attn_output)
    clean_attn = clean_attn_output[0]  # [1, seq_len, emb_dim]
    corrupt_attn = corrupt_attn_output[0]

    # Логиты без патчинга
    with torch.no_grad():
        clean_hidden = get_last_hidden(clean_prompt)
        clean_logits = get_logits_from_hidden(clean_hidden, [correct, wrong])
        clean_diff = get_logits_diff_by_category(clean_hidden, true_case, corrupted_acc)
        print(f"✅ Без патча: {correct} = {clean_logits[0]:.4f}, {wrong} = {clean_logits[1]:.4f} → diff = {clean_diff:.4f}")
        print_topk_logits(clean_hidden)  # 🔍 вот тут


    # Перебор голов
    head_deltas = []


    for head in range(n_heads):
        patched = patch_single_head(clean_attn, corrupt_attn, head, head_size)
        patched_logits = run_with_patched_attn(patched, clean_prompt, [correct, wrong])
        patched_diff = get_logits_diff_by_category(patched, true_case, corrupted_acc)
        delta = patched_diff - clean_diff

        head_deltas.append((head, delta))
        print(f"🔁 Head {head:2d}: Patched diff = {patched_diff:.4f}, Δ = {delta:+.4f}")
        print("difference from clean diff: ", patched_diff - clean_diff)

    # Сортировка голов по влиянию
    
    head_deltas.sort(key=lambda x: abs(x[1]), reverse=True)
    print("\n📊 🧠 printing for prompt: ", clean_prompt, corrupt_prompt)
    for rank, (head, delta) in enumerate(head_deltas, 1):
        print(f"   {rank}. Head {head:2d}: Δ = {delta:+.4f}")
    
    get_attention_weights(clean_prompt, attn_weights_storage)
    clean_attn_weights = attn_weights_storage[0]
    tokens = clean_prompt.split(" ")
    for i in range(4):
        head, delta = head_deltas[i]
        print(f"Визуализация внимания для головы {head} с Δ={delta:.4f}")

        plot_attention(clean_attn_weights, head, tokens, title=f"Clean prompt attention")

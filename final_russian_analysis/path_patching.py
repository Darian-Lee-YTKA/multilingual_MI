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


# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# this should be a new file 

# probing classifier for first token 
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split
from tqdm import tqdm

class InstrumentalClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim=128):
        super().__init__()
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.dropout = nn.Dropout(0.1)
        self.linear2 = nn.Linear(hidden_dim, 1) 
     
    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = self.dropout(x)
        x = self.linear2(x)
        return torch.sigmoid(x)  

# added more instrumental stems 
true = [
    "я горжусь", "она дорожит", "мы управляем", "ты восхищаешься", "они владеют",
    "я наслаждаюсь", "он увлечён", "мы интересуемся", "ты занимаешься", "она довольна",
    "он пользуется", "они шутят над", "ты смеёшься над", "мы издеваемся над", "она ухаживает за",
    "я наблюдаю за", "он следит за", "ты заботишься о", "мы воюем с", "они сотрудничают с",
    "я борюсь с", "он делится с", "она спорит с", "мы сражаемся с", "ты дерёшься с",
    "они знакомятся с", "я работаю над", "ты занимаешься с", "он спорит с", "она контактирует с",
    "мы взаимодействуем с", "я ругаюсь с", "ты миришься с", "они переписываются с", "мы советуемся с",
    "он консультируется с", "она играет с", "я соревнуюсь с", "ты разговариваешь с", "мы обсуждаем с",
    "они общаются с", "он делится впечатлениями с", "я встречаюсь с", "она посоветовалась с",
    "мы сотрудничаем с", "ты переписываешься с", "они взаимодействуют с", "он экспериментирует с",
    "я сопереживаю с", "она смеётся с", "мы боремся вместе с", "ты сражаешься рядом с",
    "они обмениваются с", "я делюсь знаниями с", "он разговаривает с", "она шутит над",
    "мы насмехаемся над", "ты заботишься о", "они ухаживают за", "я воюю с", "он консультируется у",
    "она пользуется", "мы управляем", "ты гордишься", "они занимаются", "я доволен",
    "он интересуется", "она восхищается", "мы владеем", "ты наслаждаешься", "они дорожат",
    "я увлекаюсь", "он увлекается", "она взаимодействует с", "мы контактируем с", "ты работаешь над",
    "они строят с", "я экспериментирую с", "он сотрудничают с", "она делится опытом с",
    "мы взаимодействуем", "ты обсуждаешь с", "они делятся мнением с", "я поддерживаю контакт с",
    "он шутит с", "она советуется с", "мы смеёмся над", "ты рассказываешь о", "они делятся мыслями с",
    "я взаимодействую с", "он взаимодействует с", "она контактирует с", "мы сражаемся рядом с",
    "ты работаешь вместе с", "они борются с", "я ругаюсь на", "он занимается спортом с",
    "она обсуждает с", "мы обсуждаем вместе с", "ты делишься с", "они переписываются с"
]

false = [
    "я люблю", "мы обсуждаем", "она видит", "они спрашивают",
    "я хочу", "ты слушаешь", "он изучает", "мы осуждаем", "они терпят",
    "она принимает", "ты читаешь", "я знаю", "мы помним", "он забывает",
    "ты ждёшь", "она начинает", "они бросают", "мы получаем", "он замечает",
    "я выбираю", "ты пробуешь", "она строит", "мы создаём", "они покупают",
    "ты проверяешь", "она пишет", "я нахожу", "он ловит", "мы выносим",
    "они читают", "она слушает", "ты видишь", "я слышу", "он хочет",
]

true = [tokenizer.bos_token + " " + s for s in true]
false = [tokenizer.bos_token + " " + s for s in false]

model.eval()

@torch.no_grad()
def get_first_token_embedding(sentences):
    embeddings = []
    for sent in tqdm(sentences):
        inputs = tokenizer(sent, return_tensors="pt").to(device)
        # Добавляем BOS токен в начало input_ids
        bos_id = tokenizer.bos_token_id
        if bos_id is None:
            raise ValueError("Tokenizer has no bos_token_id")
        input_ids = inputs.input_ids
        # Создаём новый tensor с BOS в начале
        input_ids_with_bos = torch.cat([
            torch.tensor([[bos_id]], device=device),
            input_ids
        ], dim=1)
        inputs['input_ids'] = input_ids_with_bos

        # Также если есть attention_mask, добавим туда 1 в начале
        if 'attention_mask' in inputs:
            attention_mask = inputs['attention_mask']
            attention_mask_with_bos = torch.cat([
                torch.tensor([[1]], device=device),
                attention_mask
            ], dim=1)
            inputs['attention_mask'] = attention_mask_with_bos

        outputs = model(**inputs, output_hidden_states=True)
        hidden_states = outputs.hidden_states[-1]  
        first_token_embed = hidden_states[:, 0, :]  
        embeddings.append(first_token_embed.squeeze(0).cpu())
    return torch.stack(embeddings)


X_true = get_first_token_embedding(true)
X_false = get_first_token_embedding(false)

X = torch.cat([X_true, X_false], dim=0)
y = torch.cat([
    torch.ones(len(X_true)),   
    torch.zeros(len(X_false)) 
], dim=0)

dataset = TensorDataset(X, y)
n_total = len(dataset)
n_test = int(n_total * 0.15)
n_train = n_total - n_test
train_dataset, test_dataset = random_split(dataset, [n_train, n_test], generator=torch.Generator().manual_seed(42))

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=8)

clf = InstrumentalClassifier(input_dim=X.shape[1]).to(device)
loss_fn = nn.BCELoss()
optimizer = torch.optim.Adam(clf.parameters(), lr=1e-3)

for epoch in range(3):
    total_loss = 0
    clf.train()
    for batch_x, batch_y in train_loader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device).unsqueeze(1)
        optimizer.zero_grad()
        preds = clf(batch_x)
        loss = loss_fn(preds, batch_y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}: loss = {total_loss:.4f}")

clf.eval()
correct = 0
total = 0
with torch.no_grad():
    for batch_x, batch_y in test_loader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        preds = clf(batch_x).squeeze()
        predicted = (preds > 0.5).long()
        correct += (predicted == batch_y.long()).sum().item()
        total += batch_y.size(0)

print(f"Test accuracy: {correct / total:.2%}")
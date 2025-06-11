import torch
import pandas as pd
from transformers import AutoTokenizer, GPT2LMHeadModel

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")


model_name = "bigscience/bloom-3b"
model = GPT2LMHeadModel.from_pretrained(model_name, output_hidden_states=True).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_name)

def logit_lens(prompt, token_position=-1, top_k=5):
    print("\n")
    print("===== prompt!: ", prompt)
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)

    hidden_states = outputs.hidden_states  # Список из [embeddings, layer1, ..., final]
    

    for i, layer_hidden in enumerate(hidden_states[1:], 1):  # Пропускаем embeddings
        token_vector = layer_hidden[0, token_position]  # (hidden_dim,)
        logits = model.lm_head(token_vector)  # (vocab_size,)
        top = torch.topk(logits, top_k)
        predicted_tokens = tokenizer.batch_decode(top.indices)
        print(f"Layer {i:2d}: ", end="")
        print(", ".join(f"{tok.strip()} ({logit.item():.2f})" for tok, logit in zip(predicted_tokens, top.values)))

rumi_data = pd.read_csv('de_mi_dataset.csv', encoding='utf-8')
print(rumi_data.columns)
relevant_sentences = rumi_data[rumi_data['correct_case '] == "ins"].iloc[:, 0].tolist()
print(relevant_sentences)

for sentence in relevant_sentences:
    logit_lens(prompt=sentence)
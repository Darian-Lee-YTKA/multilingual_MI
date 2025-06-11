import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM

device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")


model_name = "bigscience/bloom-3b"
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,  # Use half precision for memory efficiency
    low_cpu_mem_usage=True,
    device_map="auto"  # Added for better memory management with BLOOM
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

def logit_lens(prompt, token_position=-1, top_k=5):
    print("\n")
    print("===== prompt!: ", prompt)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)

    hidden_states = outputs.hidden_states  # Список из [embeddings, layer1, ..., final]
    

    for i, layer_hidden in enumerate(hidden_states[1:], 1):  # Пропускаем embeddings
        token_vector = layer_hidden[0, token_position]  # (hidden_dim,)
        logits = model.lm_head(token_vector)  # (vocab_size,)
        top = torch.topk(logits, top_k)
        predicted_tokens = tokenizer.batch_decode(top.indices)
        print(f"Layer {i:2d}: ", end="")
        print(", ".join(f"{tok.strip()} ({logit.item():.2f})" for tok, logit in zip(predicted_tokens, top.values)))

# rumi_data = pd.read_csv('en_mi_dataset.csv', encoding='utf-8')
# print(rumi_data.columns)
# relevant_sentences = rumi_data[rumi_data['correct_case '] == "ins"].iloc[:, 0].tolist()
# print(relevant_sentences)

relevant_sentences = ["Jack and Jill went up the hill. He got a gift for ","Jack and Jill went up the hill. She got a gift for ",] 
# de german examples
relevant_sentences = ["Jack und Jill gingen den Hügel hinauf. Er bekam ein Geschenk für ", "Jack und Jill gingen den Hügel hinauf. Sie bekam ein Geschenk für"]
correct_case = ["sie", "ihn"]
for sentence in relevant_sentences:
    logit_lens(prompt=sentence)
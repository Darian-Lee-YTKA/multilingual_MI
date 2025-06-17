import torch
from transformers import AutoTokenizer, GPT2LMHeadModel

device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")

# Загружаем ruGPT
model_name = "sberbank-ai/rugpt3large_based_on_gpt2"
model = GPT2LMHeadModel.from_pretrained(model_name, output_hidden_states=True).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_name)

def logit_lens(prompt, token_position=0, top_k=5):
    print("Seed: ", prompt)
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)

    hidden_states = outputs.hidden_states  # Список из [embeddings, layer1, ..., final]
    print(f"Token: {tokenizer.decode(inputs.input_ids[0, token_position])} (index {token_position})")

    for i, layer_hidden in enumerate(hidden_states[1:], 1):  # Пропускаем embeddings
        token_vector = layer_hidden[0, token_position]  # (hidden_dim,)
        logits = model.lm_head(token_vector)  # (vocab_size,)
        top = torch.topk(logits, top_k)
        predicted_tokens = tokenizer.batch_decode(top.indices)
        print(f"Layer {i:2d}: ", end="")
        print(", ".join(f"{tok.strip()} ({logit.item():.2f})" for tok, logit in zip(predicted_tokens, top.values)))


logit_lens("я играю с ", token_position=2)

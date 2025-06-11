import torch
from transformers import BloomForCausalLM, BloomTokenizerFast
import pandas as pd
from transformers import AutoTokenizer, GPT2LMHeadModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = "cuda:2"
model = BloomForCausalLM.from_pretrained("bigscience/bloom-3b")
tokenizer = BloomTokenizerFast.from_pretrained("bigscience/bloom-3b")
# use rugpt2
# model_name = "sberbank-ai/rugpt3large_based_on_gpt2"
# model = GPT2LMHeadModel.from_pretrained(model_name, output_hidden_states=True).to(device)
# tokenizer = AutoTokenizer.from_pretrained(model_name)


import sys
sys.stdout = open('logit_lens_bloom_en.txt', 'w', encoding='utf-8')

def logit_lens(prompt, token_position=-1, top_k=5, model=model, tokenizer=tokenizer, device=device, match_tokens=None):
    print("\n")
    print("===== prompt!: ", prompt)
    model = model.to(device)
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    hidden_states = outputs.hidden_states  # [embeddings, layer1, ..., final]

    for i, layer_hidden in enumerate(hidden_states[1:], 1):  # Skip embeddings
        token_vector = layer_hidden[0, token_position]  # (hidden_dim,)
        logits = model.lm_head(token_vector)  # (vocab_size,)
        top = torch.topk(logits, top_k)
        predicted_tokens = tokenizer.batch_decode(top.indices)

        if match_tokens:
            # Check for any matches in the top predictions
            matches = [tok for tok in predicted_tokens if tok.strip() in match_tokens]
            if matches:
                print(f"Layer {i:2d}: ", end="")
                print(", ".join(f"{tok.strip()} ({logit.item():.2f})" 
                                for tok, logit in zip(predicted_tokens, top.values)))
        else:
            print(f"Layer {i:2d}: ", end="")
            print(", ".join(f"{tok.strip()} ({logit.item():.2f})" 
                            for tok, logit in zip(predicted_tokens, top.values)))

df = pd.read_csv('russian_mi_dataset.csv', encoding='utf-8')
df.columns = df.columns.str.strip()
relevant_sentences = df[df['correct_case'] == "nom"]

# with tqdm
# from tqdm import tqdm
# for index, row in tqdm(relevant_sentences.iterrows()):
#     row_str = row['sentence']
#     row_val = row[row['correct_case']]
#     match_tokens = [row_val," "+row_val, row_val+" "]
#     match_tokens = None
#     logit_lens(row_str, token_position=-1, top_k=5, model=model, tokenizer=tokenizer, device=device, match_tokens=match_tokens)

test_strs = [
    "Sarah borrowed John’s car all day, and when she returned it, she handed the keys to",
    "The principal invited all the teachers to the auditorium, and at the end of the assembly, he thanked",
    "I saw Mark walking alone in the park, and he seemed to be in a hurry. So, I followed",
    "After finishing the beta test, the developers sent their bug report to",
    "Sarah and Jack went to the market together, but she came home without",
    "I introduced Jack to Sarah, but it seemed like he already knew",
]

# test_strs = [
#     "Sarah lieh Johns Auto den ganzen Tag, und als sie es zurückgab, übergab sie die Schlüssel ihm",
#     "Der Schulleiter lud alle Lehrkräfte in die Aula ein, und am Ende der Versammlung dankte er ihnen",
#     "Ich sah Mark allein im Park spazieren, und er schien es eilig zu haben. Also folgte ich ihm",
#     "Nachdem sie den Beta-Test abgeschlossen hatten, schickten die Entwickler ihren Fehlerbericht ihm",
#     "Sarah und Jack gingen zusammen zum Markt, aber sie kam ohne ihn",
#     "Ich stellte Jack Sarah vor, aber es schien, als wüsste er von ihr",
# ]

for test_str in test_strs:
    logit_lens(test_str, token_position=-1, top_k=5, model=model, tokenizer=tokenizer, device=device, match_tokens=None)

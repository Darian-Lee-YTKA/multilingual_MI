import torch
from transformers import BloomForCausalLM, BloomTokenizerFast
import pandas as pd
from transformers import AutoTokenizer, GPT2LMHeadModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = "cuda:2"
# model = BloomForCausalLM.from_pretrained("bigscience/bloom-3b")
# tokenizer = BloomTokenizerFast.from_pretrained("bigscience/bloom-3b")
# use rugpt2
model_name = "sberbank-ai/rugpt3large_based_on_gpt2"
model = GPT2LMHeadModel.from_pretrained(model_name, output_hidden_states=True).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_name)


import sys
sys.stdout = open('logit_lens_RUGPT_en.txt', 'w', encoding='utf-8')

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
# "Sarah lieh sich den ganzen Tag Johns Auto aus, und als sie es zurückgab, gab sie die Schlüssel zu",
# "Der Schulleiter lud alle Lehrer zu dem Treffen ein und dankte",
# "Ich sah Mark allein im Park spazieren gehen, und er schien es eilig zu haben. Also folgte ich",
# "Nach Abschluss des Betatests schickten die Entwickler ihren Fehlerbericht an",
# "Sarah und Jack gingen zusammen einkaufen, aber sie kam ohne zurück",
# "Ich stellte Jack Sarah vor, aber es schien, als wüsste er es schon",
# ]

# test_strs = [
# "Сара весь день пользовалась машиной Джона, а когда вернула её, она передала ключи к",
# "Директор пригласил всех учителей в актовый зал и в конце встречи поблагод",
# "Я увидел Марка, идущего в одиночестве по парку, и он, казалось, торопился. Поэтому я последова",
# "После завершения бета-тестирования разработчики отправили свой отчет об ошибке к",
# "Сара и Джек вместе пошли на рынок, но она вернулась домой без",
# "Я познакомил Джека с Сарой, но, похоже, он уже все знал о",
# ]
for test_str in test_strs:
    logit_lens(test_str, token_position=-1, top_k=5, model=model, tokenizer=tokenizer, device=device, match_tokens=None)

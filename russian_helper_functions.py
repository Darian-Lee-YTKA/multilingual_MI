
import pandas as pd
import sys
from pathlib import Path
import torch as t
from torch import Tensor
import numpy as np
import einops
from tqdm.notebook import tqdm
import plotly.express as px
import re
import itertools
from jaxtyping import Float, Int, Bool
from typing import Literal, Callable
from functools import partial
from IPython.display import display, HTML
from rich.table import Table, Column
from rich import print as rprint
import circuitsvis as cv
from transformer_lens.hook_points import HookPoint
from transformer_lens import utils, HookedTransformer, ActivationCache
from transformer_lens.components import Embed, Unembed, LayerNorm, MLP
import pandas as pd
import os


import torch
import torch.nn.functional as F
from transformer_lens import HookedTransformer
from transformers import AutoTokenizer
import utils  # твой модуль

import torch
import torch.nn.functional as F
from transformer_lens import HookedTransformer
from transformers import AutoTokenizer

import torch
import torch.nn.functional as F
from transformer_lens import HookedTransformer
from transformers import AutoTokenizer

# 💻 Устройство — принудительно cuda:3, если доступно
device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")

print(f"Используем устройство: {device}")

# 🧠 Загружаем модель Mistral на device
model = HookedTransformer.from_pretrained(
    "mistralai/Mistral-7B-v0.1",
    center_unembed=True,
    center_writing_weights=True,
    fold_ln=True,
    refactor_factored_attn_matrices=False,
    device=device,  # важно указать при загрузке
)
model.to(device)  # на всякий случай дополнительно

# Токенизатор (у токенизатора device нет)
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")

# 🔍 Функция логит-линзы с явным переносом всех тензоров на device
def run_logit_lens(text: str):
    tokens = model.to_tokens(text, prepend_bos=False).to(device)
    token_strs = model.to_str_tokens(tokens)

    logits, cache = model.run_with_cache(tokens)

    print(f"\n🔍 Logit lens для строки: {text!r}")
    print(f"📎 Токены: {token_strs}\n")

    for layer in range(model.cfg.n_layers):
        resid = cache["resid_post", layer][0, -1]
        if resid.device != device:
            resid = resid.to(device)

        logits_lens = model.unembed(resid)
        if logits_lens.device != device:
            logits_lens = logits_lens.to(device)

        probs = F.softmax(logits_lens, dim=-1)
        topk = torch.topk(probs, k=5)

        print(f"--- Слой {layer} ---")
        for i in range(5):
            token = model.to_string(topk.indices[i])
            prob = topk.values[i].item()
            print(f"{i+1}. {token!r:<10} → {prob:.4f}")

# 🎯 Проверка (без utils), также все тензоры на device
def test_prompt(prompt: str, target: str):
    print(f"\n🎯 Проверка: prompt = {prompt!r}, target = {target!r}")
    tokens = model.to_tokens(prompt + target, prepend_bos=True).to(device)

    logits = model(tokens)
    target_token = tokens[0, -1]

    last_logits = logits[0, -2]
    if last_logits.device != device:
        last_logits = last_logits.to(device)

    probs = F.softmax(last_logits, dim=-1)
    target_prob = probs[target_token].item()

    print(f"Вероятность токена {target!r}: {target_prob:.4f}")

# Запускаем логит-линзу
run_logit_lens("Ich liebe")
run_logit_lens("Ich gab")

# Запускаем проверки с двумя вариантами ответов
for prompt in ["Ich liebe ", "Ich gab "]:
    for answer in ["den", "dem"]:
        test_prompt(prompt, answer)

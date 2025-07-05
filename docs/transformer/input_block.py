from dataclasses import dataclass
import math
import time
from typing import List, Optional, Tuple

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F

device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
with open('tiny_shakespeare.txt', 'r') as f:
    data = f.read()
vocab = sorted(list(set(data)))
vocab.extend(['<|begin_of_text|>', '<|end_of_text|>', '<|pad_id|>'])
vocab_size = len(vocab)
itos = {i: ch for i, ch in enumerate(vocab)}
stoi = {ch: i for i, ch in enumerate(vocab)}


def encode(s):
    return [stoi[ch] for ch in s]


def decode(l):
    return ''.join(itos[i] for i in l)


token_bos = torch.tensor([stoi['<|begin_of_text|>']], dtype=torch.int, device=device)
token_eos = torch.tensor([stoi['<|end_of_text|>']], dtype=torch.int, device=device)
token_pad = torch.tensor([stoi['<|pad_id|>']], dtype=torch.int, device=device)
prompts = "Hello World"
encoded_tokens = encode(prompts)
decoded_text = decode(encoded_tokens)

print(f"Length of shakespeare in character: {len(data)}")
print(f"The vocabulary looks like this: {''.join(vocab)}\n")
print(f"Vocab size: {vocab_size}")
print(f"encoded_tokens: {encoded_tokens}")
print(f"decoded_text: {decoded_text}")
"""
Length of shakespeare in character: 1115394
The vocabulary looks like this:
 !$&',-.3:;?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz<|begin_of_text|><|end_of_text|><|pad_id|>

Vocab size: 68
encoded_tokens: [20, 43, 50, 50, 53, 1, 35, 53, 56, 50, 42]
decoded_text: Hello World
"""

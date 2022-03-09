import os, sys, json
from tqdm import tqdm
from collections import defaultdict

import numpy as np
import torch
import torchtext
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchtext.data import get_tokenizer
from gensim.models import Word2Vec

from utils import parse_lines
from dataset import NGramDataset
from NeuralLM import NeuralLM
from LstmLM import LstmLM

EMBEDDING_DIM = 300
CONTEXT_SIZE = 5

list_list_words= parse_lines('data/shakespeare.txt')

model = Word2Vec(list_list_words, vector_size=EMBEDDING_DIM, window=CONTEXT_SIZE, min_count=2, workers=16)

words = model.wv.index_to_key
weights = torch.FloatTensor(model.wv.vectors)
embedding = nn.Embedding.from_pretrained(weights)

filtered_data = [word for line in list_list_words for word in line if word in words]

shakespeare_data = filtered_data
vocab = words
dataset = NGramDataset(shakespeare_data, vocab, CONTEXT_SIZE)
dataloader = DataLoader(dataset, batch_size=4096,
            shuffle=True, num_workers=8)
losses = []
loss_function = nn.NLLLoss()

# %%
epochs = 20
model = LstmLM(len(vocab), EMBEDDING_DIM, CONTEXT_SIZE, hidden_dim=512, 
        embedding_weights=weights).cuda()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=300, gamma=0.1)

# %%
model.cuda()
model.zero_grad()
for epoch in range(epochs):
    losses = []
    for idx, batch in enumerate(dataloader):
        context_idx = torch.stack(batch[0], axis=1).cuda()
        target_idx = batch[1].cuda()
        log_probs = model(context_idx)
        loss = loss_function(log_probs, target_idx)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    print(f"epoch {epoch} loss : {np.mean(losses)}")
    scheduler.step()


# %%

model.cpu()

def generate(prompt = None, length=10):
    if prompt is None:
        prompt = input(f"Input the Prompt of size {CONTEXT_SIZE} : ")
    prompt = prompt.split()

    while len(prompt) < CONTEXT_SIZE:
        prompt = ['.'] + prompt

    for word in prompt:
        if word not in dataset.word_to_idx:
            print(f"Error: word {word} not in the vocabulary.\n")
            return 

    for _ in range(length):
        context = prompt[-CONTEXT_SIZE:]
        context_idx = torch.tensor([dataset.word_to_idx[w] for w in context], dtype=torch.long)
        log_probs = model(context_idx)
        probs = F.softmax(log_probs, dim=1).detach().numpy().reshape(-1,)
        idx = np.random.choice(len(vocab), p=probs)
        prompt.append(dataset.idx_to_word[idx])
        
    return ' '.join(prompt)

def interactive_generate(length=10, n=10):
    for _ in range(n):
        print(generate(length=length))

generate(length=20, prompt = None)
# %%

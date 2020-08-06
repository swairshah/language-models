import os, sys, json, glob
from io import open
import unicodedata
from collections import defaultdict

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchtext
from torchtext.data import get_tokenizer

def parse_lines(fname):
    tokenizer = get_tokenizer("basic_english")
    vocab = defaultdict(int)
    data = []
    with open(fname) as f:
        for line in f.readlines():
            line = tokenizer(line.strip())
            if len(line) <= 1:
                continue
            data.append(line)
            for word in line:
                vocab[word] += 1
    return data

def get_ngrams(word_list):
    unigrams = list(set(word_list))
    bigrams = defaultdict(int)
    trigrams = defaultdict(int)

    for idx in range(len(word_list)-1):
        bigrams[(word_list[idx], word_list[idx+1])] += 1

    for idx in range(len(word_list) - 2):
        trigrams[(word_list[idx], word_list[idx + 1], word_list[idx + 2])] += 1

    return unigrams, bigrams, trigrams

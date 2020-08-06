import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchtext
from torchtext.data import get_tokenizer
from collections import defaultdict

class UnigramSampler:
    def __init__(self, freq_dict):
        self.freq_dict = freq_dict
        self.tokens = list(freq_dict.keys())
        self.n = len(self.tokens)
        self.weights = list(freq_dict.values())
        self.weights = np.array(self.weights)
        self.weights = self.weights/self.weights.sum()
        
    def sample(self):
        return np.random.choice(self.tokens, p=self.weights)

    def generate(self, length=10, context=""):
        sent = context.split()
        for _ in range(length):
            sent.append(self.sample())
        return ' '.join(sent)
    
class BigramSampler:
    def __init__(self, freq_dict):
        self.freq_dict = freq_dict
    
    def sample(self, first):
        freq_dict = {i:j for (i,j) in self.freq_dict.items() if i[0] == first}
        tokens = [i[1] for i in freq_dict.keys()]
        n = len(tokens)
        weights = list(freq_dict.values())
        weights = np.array(weights)
        weights = weights/weights.sum()

        return np.random.choice(tokens, p=weights)     

    def generate(self, length=10, context="."):
        sent = context.split()
        for _ in range(length):
            sent.append(self.sample(sent[-1]))
        return ' '.join(sent)
 

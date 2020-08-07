import numpy as np
import torch
import torchtext
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

class NeuralLM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, context_size, hidden_dim=256, 
                 embedding_weights=None):
        super(NeuralLM, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.context_size = context_size
        if embedding_weights is not None:
            self.embeddings = nn.Embedding.from_pretrained(embedding_weights)
        else:
            self.embeddings = nn.Embedding(vocab_size, embedding_dim)

        self.linear1 = nn.Linear(context_size * embedding_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, vocab_size)

    def forward(self, inputs):
        embeds = self.embeddings(inputs).view((-1, self.context_size*self.embedding_dim))
        out = F.relu(self.linear1(embeds))
        out = self.linear2(out)
        log_probs = F.log_softmax(out, dim=1)
        return log_probs

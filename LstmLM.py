import torch
import torchtext
import torch.nn as nn
import torch.nn.functional as F

class LstmLM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, context_size, hidden_dim=256, 
                 embedding_weights=None):
        super(LstmLM, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.context_size = context_size
        self.hidden_dim=hidden_dim
        if embedding_weights is not None:
            self.embeddings = nn.Embedding.from_pretrained(embedding_weights)
        else:
            self.embeddings = nn.Embedding(vocab_size, embedding_dim)

        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.linear = nn.Linear(context_size*hidden_dim, vocab_size)

    def forward(self, inputs):
        embeds = self.embeddings(inputs)
        embeds = self.embeddings(inputs).view((-1, self.context_size, self.embedding_dim))
        out, _ = self.lstm(embeds)
        out = self.linear(out.reshape(-1, self.context_size*self.hidden_dim))
        log_probs = F.log_softmax(out, dim=1)
        return log_probs

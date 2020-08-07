import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
from torchtext.data import get_tokenizer
from utils import parse_lines

class NGramDataset(Dataset):
    def __init__(self, data, vocab, context_size):
        self.data = data
        self.length = len(data)
        self.vocab = vocab
        self.context_size = context_size

        self.word_to_idx = {}
        self.idx_to_word = {}
        for idx,word in enumerate(vocab):
            self.word_to_idx[word] = idx
            self.idx_to_word[idx] = word

    def __len__(self):
        return self.length-self.context_size

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        context_idx = [self.word_to_idx[self.data[i+idx]] for i in range(self.context_size)]
        label_idx = self.word_to_idx[self.data[idx+self.context_size]]
        return context_idx, label_idx


if __name__ == "__main__":
    context_size = 5

    lines = parse_lines('data/shakespeare.txt')
    word_data = [word for line in lines for word in line]
    vocab = list(set(word_data))
    
    dataset = NGramDataset(word_data, vocab, context_size)
    dataloader = DataLoader(dataset, batch_size=1)
    for idx, batch in enumerate(dataloader):
        context = batch[0]
        label = batch[1]
        print(f"context : {context} \nlabel : {label}")
        break

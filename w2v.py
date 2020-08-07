import torch
import torch.nn as nn
from gensim.models import Word2Vec
from torchtext.data import get_tokenizer
from collections import defaultdict
from utils import *

list_list_words= parse_lines('data/shakespeare.txt')

model = Word2Vec(list_list_words, size=300, window=3, min_count=2, workers=16)
model.wv.save_word2vec_format("shakespeare_w2v.txt", binary=False)

import IPython
IPython.embed()

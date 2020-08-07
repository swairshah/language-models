from utils import *
from ProbabilisticLM import UnigramSampler, BigramSampler

lines = parse_lines("data/sherlock.txt")
words = [word for line in lines for word in line] 

# %%

def get_ngrams(word_list):
    unigrams = defaultdict(int)
    bigrams = defaultdict(int)
    trigrams = defaultdict(int)

    length = len(word_list)
    for idx in range(length - 2):
        unigrams[word_list[idx]] += 1
        bigrams[(word_list[idx], word_list[idx+1])] += 1
        trigrams[(word_list[idx], word_list[idx + 1], word_list[idx + 2])] += 1

    unigrams[word_list[length-1]] += 1
    unigrams[word_list[length-2]] += 1
    bigrams[(word_list[length-1], word_list[length-2])] += 1

    return unigrams, bigrams, trigrams


unigrams, bigrams, trigrams = get_ngrams(words)

# %%

filtered = {i:j for i,j in trigrams.items() if j >= 3}

# %%

unigramLM = UnigramSampler(unigrams)

# %%



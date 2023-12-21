from gensim.models.keyedvectors import KeyedVectors

import torch as th
from torch.utils.data import Dataset
import pickle
import torch.nn.functional as F
import numpy as np
import re
from collections import defaultdict
from torch.utils.data.dataloader import default_collate

from stop_words import ENGLISH_STOP_WORDS

we_dim = 300
max_words = 50

we = KeyedVectors.load_word2vec_format('/ssd_scratch/cvit/darshan/data/GoogleNews-vectors-negative300.bin', binary=True)


def _zero_pad_tensor(tensor, size):
    if len(tensor) >= size:
        return tensor[:size]
    else:
        zero = np.zeros((size - len(tensor), we_dim), dtype=np.float32)
        return np.concatenate((tensor, zero), axis=0)


def _tokenize_text(sentence):
    w = re.findall(r"[\w']+", str(sentence))
    return w

def _words_to_we(words):
    # words = [word for word in words if word in self.we.vocab]
    words = list(map(lambda word: word.lower(), words))
    words = [word for word in words if (word in we.vocab) and (word not in ENGLISH_STOP_WORDS)]
    if words:
        we_t = _zero_pad_tensor(we[words], max_words)
        return th.from_numpy(we_t)
    else:
        return th.zeros(max_words, we_dim)


cap = "Time is suspect Train arrives at 7 o'clock Two simultaneous events Do two different observer agree? NO! Relativity of simultaneity"

caption = _words_to_we(_tokenize_text(cap))

print(cap, caption.shape)
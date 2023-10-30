import numpy as np
import torch

def renormalizer(data, min_val, max_val):
    data = data * max_val
    data = data + min_val
    return data

def ones_target(size, min, max):
    return torch.FloatTensor(size, 1).uniform_(min, max)

def zeros_target(size, min, max):
    return torch.FloatTensor(size, 1).uniform_(min, max)

def target_flip(size, min, max, ratio):
    ori = torch.FloatTensor(size, 1).uniform_(min, max)
    index = np.random.choice(size, int(size * ratio), replace=False)
    ori[index] = 1 - ori[index]
    return ori

def np_sigmoid(x):
    z = np.exp(-x)
    sig = 1 / (1 + z)
    return sig

def np_rounding(prob):
    y = np.round(prob)
    return y
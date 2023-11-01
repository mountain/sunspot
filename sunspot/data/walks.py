import numpy as np
import torch as th
from torch.utils.data import Dataset


def load_walks():
    with open("data/walks.txt") as f:
        lines = f.readlines()
    return [np.array([int(i) for i in line.split(',')], dtype=np.int32) for line in lines]


class Walks(Dataset):
    def __init__(self, transform=None):
        self.walks = load_walks()
        self.transform = transform

    def __len__(self):
        return len(self.walks)

    def __getitem__(self, ix):
        walk = self.walks[ix]
        if self.transform:
            walk = self.transform(walk)
        return th.tensor(walk, dtype=th.long)

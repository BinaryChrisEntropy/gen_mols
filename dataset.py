import torch
from torch.utils.data import Dataset
import numpy as np


class ChemDataset(Dataset):
    def __init__(self, text, seq_len):
        self.text = text
        self.seq_len = seq_len
        self.alphabet = tuple(sorted(set(text)))
        self.int2char = dict(enumerate(self.alphabet))
        self.int2char[39] = "X"  # start token
        self.int2char[40] = "Y"  # pad token
        self.char2int = {c: i for i, c in self.int2char.items()}

        # prepend and append start and end token to each sequence
        # start of seq. = X, end of seq. = \n
        self.samples = text.split()
        self.samples = ["X" + s + "\n" for s in self.samples]
        self.one_hot = torch.nn.functional.one_hot(torch.arange(0, 41)).float()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        # append padding tokens Y
        x = self.samples[index][:-1] + "Y" * (self.seq_len - len(self.samples[index]) + 1)
        y = self.samples[index][1:] + "Y" * (self.seq_len - len(self.samples[index]) + 1)

        # encode from char to integer
        x = np.array([self.char2int[c] for c in x])
        y = np.array([self.char2int[c] for c in y])

        # one hot encoding for input
        x = torch.tensor(x, dtype=torch.long)
        x = [self.one_hot[i] for i in x]
        x = torch.stack(x)
        y = torch.tensor(y, dtype=torch.long)

        return x, y

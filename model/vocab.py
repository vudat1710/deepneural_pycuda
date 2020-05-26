import numpy as np
import pandas as pd

PAD, UNK = '<PAD>', '<UNK>'


def load_vocab_file(fn):
    with open(fn, mode='r', encoding='utf8') as f:
        ids = f.read().strip().split()
        f.close()
    return ids


def load_glove_emb(fn):
    df = pd.read_csv(fn, sep=' ', header=None)
    words = df.iloc[:, 0].values
    vectors = df.iloc[:, 1:].values
    return words, vectors


class Vocab:
    def __init__(
            self,
            words=None,
            vocab_file=None,
            additional_terms=True,
    ):
        assert words is not None or vocab_file is not None
        if words is None:
            words = load_vocab_file(vocab_file)
        if additional_terms:
            self.i2w = [*words, UNK, PAD]
        else:
            self.i2w = words
        self.w2i = {word: i for i, word in enumerate(self.i2w)}
        self.unk_index, self.padding_index = len(words), len(words) + 1

    def word2index(self, w):
        return self.w2i.get(w, self.unk_index)

    def words2indices(self, ws: list):
        return [self.word2index(w) for w in ws]

    def __len__(self):
        return len(self.i2w)

    def __getitem__(self, i):
        return self.i2w[i]

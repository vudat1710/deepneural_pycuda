from model.tensor_new import Tensor
from model.vocab import Vocab
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import settings


class RedditDataset(Dataset):
    def __init__(
            self,
            df: pd.DataFrame,
            source_sub_col='source_sub',
            target_sub_col='target_sub',
            user_col='user',
            content_col='title',
            label_col='label',
            user_vocab: Vocab = None,
            sub_vocab: Vocab = None,
            word_vocab: Vocab = None,
            label_vocab: Vocab = None,
    ):
        super(RedditDataset, self).__init__()
        self.df = df[[source_sub_col, target_sub_col, user_col, content_col, label_col]]
        self.source_sub_col = source_sub_col
        self.target_sub_col = target_sub_col
        self.user_col = user_col
        self.content_col = content_col
        self.user_vocab = user_vocab
        self.sub_vocab = sub_vocab
        self.word_vocab = word_vocab
        self.label_vocab = label_vocab

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        source_sub, target_sub, user, content, label = self.df.iloc[i]
        return (
            (self.sub_vocab.word2index(source_sub),
             self.sub_vocab.word2index(target_sub),
             self.user_vocab.word2index(user),
             self.word_vocab.words2indices(content)),
            self.label_vocab.word2index(label)
        )


class CollateFn:
    def __init__(
            self,
            # user_padding_index,
            # sub_padding_index,
            word_padding_index,
    ):
        # self.user_padding_index = user_padding_index
        # self.sub_padding_index = sub_padding_index
        self.word_padding_index = word_padding_index

    @staticmethod
    def padding_collate(samples, padding_index):
        max_length = max([len(sample) for sample in samples])
        x = np.zeros((max_length, len(samples)), dtype=np.int) + padding_index
        for i, sample in enumerate(samples):
            x[:len(sample), i] = sample
        return x

    def __call__(self, data):
        xs, ys = zip(*data)
        source_subs, target_subs, users, contents = zip(*xs)
        return (
            (Tensor(users, device=settings.DEVICE, d_type=np.int32),
             Tensor(source_subs, device=settings.DEVICE, d_type=np.int32),
             Tensor(target_subs, device=settings.DEVICE, d_type=np.int32),
             Tensor(self.padding_collate(contents, padding_index=self.word_padding_index), device=settings.DEVICE,
                    d_type=np.int32),
             ),
            Tensor(ys, device='cpu', d_type=np.int32, autograd=True)
        )





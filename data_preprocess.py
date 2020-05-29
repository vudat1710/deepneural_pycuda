import numpy as np
import pandas as pd
from ast import literal_eval
import settings
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

from model.data import RedditDataset, CollateFn, DataLoader
from model.vocab import Vocab, load_glove_emb


def load_npy_file(fn):
    return np.load(fn)


def load_post_cross_links(fn):
    df = pd.read_csv(fn, sep='\t', header=None)
    df.columns = ['source_sub', 'target_sub', 'time1', 'user', 'time2']
    df['source_post'] = df['time1'].apply(lambda x: x.split('T', maxsplit=1)[0])
    df['target_post'] = df['time2'].apply(lambda x: x.split('T', maxsplit=1)[0])
    return df[['source_sub', 'target_sub', 'source_post', 'target_post']]


def load_label_info(fn):
    df = pd.read_csv(fn, sep='\t', header=None)
    df.columns = ['posts', 'label']
    df[['source_post', 'target_post']] = df[['posts']].apply(lambda x: literal_eval(x[0]), axis=1, result_type='expand')
    return df[['source_post', 'target_post', 'label']]


def load_posts(fn):
    df = pd.read_csv(fn, sep='\t', header=None)
    df.columns = ['source_sub', 'source_post', 'user', 'time', 'title', 'body']
    df['title'] = df['title'].apply(lambda x: [int(i) for i in x.split(':')[-1].strip().split(',') if i != ''])
    df['body'] = df['body'].apply(lambda x: [int(i) for i in x.split(':')[-1].strip().split(',') if i != ''])
    # df['body'] = df['body'].apply(lambda x: list(map(int, x.split(':')[-1].strip().split(','))))
    return df[['source_sub', 'source_post', 'user', 'title', 'body']]


if __name__ == '__main__':
    if not os.path.exists(settings.POST_CROSS_LINKS_DUMP_FN):
        post_cross_links = load_post_cross_links(settings.POST_CROSS_LINKS_FN)
        with open(settings.POST_CROSS_LINKS_DUMP_FN, mode='wb') as f:
            pickle.dump(post_cross_links, f)
            f.close()
    else:
        with open(settings.POST_CROSS_LINKS_DUMP_FN, mode='rb') as f:
            post_cross_links = pickle.load(f)
            f.close()
    print(post_cross_links.head())
    print(f'post_cross_links length: {len(post_cross_links)}')
    print('-' * 30)

    if not os.path.exists(settings.LABEL_INFO_DUMP_FN):
        label_info = load_label_info(settings.LABEL_INFO_FN)
        with open(settings.LABEL_INFO_DUMP_FN, mode='wb') as f:
            pickle.dump(label_info, f)
            f.close()
    else:
        with open(settings.LABEL_INFO_DUMP_FN, mode='rb') as f:
            label_info = pickle.load(f)
            f.close()
    print(label_info.head())
    print(f'label_info length: {len(label_info)}')
    print('-' * 30)

    if not os.path.exists(settings.POST_INFO_DUMP_FN):
        post_info = load_posts(settings.POST_INFO_FN)
        with open(settings.POST_INFO_DUMP_FN, mode='wb') as f:
            pickle.dump(post_info, f)
            f.close()
    else:
        with open(settings.POST_INFO_DUMP_FN, mode='rb') as f:
            post_info = pickle.load(f)
            f.close()
    print(post_info.head())
    print(f'post_info length: {len(post_info)}')
    print('-' * 30)

    cross_label = post_cross_links.merge(label_info, how='inner', on=['source_post', 'target_post'])
    print(cross_label.head())
    print(f'cross_label length: {len(cross_label)}')
    print('-' * 30)

    if not os.path.exists(settings.PROCESSED_DATA):
        cross_label_tokenized = cross_label.merge(post_info, how='inner', on=['source_post', 'source_sub'])
        with open(settings.PROCESSED_DATA, mode='wb') as f:
            pickle.dump(cross_label_tokenized, f)
            f.close()
    else:
        with open(settings.PROCESSED_DATA, mode='rb') as f:
            cross_label_tokenized = pickle.load(f)
            f.close()
    print(cross_label_tokenized.head())
    print(f'cross_label_tokenized length: {len(cross_label_tokenized)}')
    print('-' * 30)

    cross_label_tokenized['content'] = cross_label_tokenized.apply(lambda x: [*x[-2], *x[-1]][:settings.MAX_LENGTH],
                                                                   axis=1)
    print(cross_label_tokenized.head())
    print('-' * 30)

    print('splitting train, dev, test')
    # burst_cross_label = cross_label_tokenized[cross_label_tokenized['label'] == 'burst']
    # non_burst_cross_label = cross_label_tokenized[cross_label_tokenized['label'] == 'non-burst']
    # train_burst, test_burst = train_test_split(burst_cross_label, test_size=0.2)
    # train_burst, dev_burst = train_test_split(train_burst, test_size=0.3)
    # train_non_burst, test_non_burst = train_test_split(non_burst_cross_label, test_size=0.2)
    # train_non_burst, dev_non_burst = train_test_split(train_non_burst, test_size=0.3)
    # 
    # train = shuffle(pd.concat((train_burst, train_non_burst), ignore_index=True))
    # test = shuffle(pd.concat((test_burst, test_non_burst), ignore_index=True))
    # dev = shuffle(pd.concat((dev_burst, dev_non_burst), ignore_index=True))
    
    train_post_ids = pd.read_csv('data/dump/train_post_ids.csv', sep='\t')
    val_post_ids = pd.read_csv('data/dump/val_post_ids.csv', sep='\t')
    test_post_ids = pd.read_csv('data/dump/test_post_ids.csv', sep='\t')
    train = cross_label_tokenized[cross_label_tokenized['source_post'].isin(train_post_ids['source_post'])]
    test = cross_label_tokenized[cross_label_tokenized['source_post'].isin(test_post_ids['source_post'])]
    dev = cross_label_tokenized[cross_label_tokenized['source_post'].isin(val_post_ids['source_post'])]
    print(f'train_length = {len(train)}, test_length = {len(test)}, dev_length = {len(dev)}')

    if not os.path.exists(settings.TRAIN_DF_DUMP):
        with open(settings.TRAIN_DF_DUMP, mode='wb') as f:
            pickle.dump(train, f)

    if not os.path.exists(settings.TEST_DF_DUMP):
        with open(settings.TEST_DF_DUMP, mode='wb') as f:
            pickle.dump(test, f)

    if not os.path.exists(settings.DEV_DF_DUMP):
        with open(settings.DEV_DF_DUMP, mode='wb') as f:
            pickle.dump(dev, f)

    print('Loading vocab...')
    user_vocab = Vocab(vocab_file=settings.USER_VOCAB_FN)
    sub_vocab = Vocab(vocab_file=settings.SUB_VOCAB_FN)
    words, word_vecs = load_glove_emb(fn=settings.GLOVE_EMBEDDING_FN)
    word_vocab = Vocab(words=list(range(len(words))))
    label_vocab = Vocab(words=['non-burst', 'burst'])

    ds = RedditDataset(
        df=cross_label_tokenized,
        user_vocab=user_vocab,
        sub_vocab=sub_vocab,
        word_vocab=word_vocab,
        label_vocab=label_vocab,
        content_col='content',
    )

    # ds_it = iter(ds)
    # print(next(ds_it))

    collate_fn = CollateFn(word_padding_index=word_vocab.padding_index)
    dl = DataLoader(dataset=ds, batch_size=512, shuffle=True, collate_fn=collate_fn, drop_last=True)
    for i, (x, y) in enumerate(dl):
        if i == 10: break
        users, source_subs, target_subs, contents = x
        print(users.shape, source_subs.shape, target_subs.shape, contents.shape, y.shape)
        print('-' * 30 + 'user' + '-' * 30)
        print(users)
        print('-' * 30 + 'source subs' + '-' * 30)
        print(source_subs)
        print('-' * 30 + 'target subs' + '-' * 30)
        print(target_subs)
        print('-' * 30 + 'y' + '-' * 30)
        print(y)



import numpy as np
import pandas as pd
import pickle
from tqdm import tqdm, trange
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.utils import shuffle
import warnings

from model.tensor_new import *
from model.data import RedditDataset, DataLoader, CollateFn
from model.reddit_model import *
import settings
from model.vocab import Vocab, load_glove_emb

warnings.filterwarnings('always')
warnings.filterwarnings('ignore')


def load_df(fn):
    with open(fn, mode='rb') as f:
        df = pickle.load(f)
        f.close()

    return df


# load dump df
train_df: pd.DataFrame = load_df(settings.TRAIN_DF_DUMP)
test_df: pd.DataFrame = load_df(settings.TEST_DF_DUMP)
dev_df: pd.DataFrame = load_df(settings.DEV_DF_DUMP)

# sampling
if settings.DOWN_SAMPLING:
    train_df_burst = train_df[train_df['label'] == 'burst']
    train_df_non_burst = train_df[train_df['label'] == 'non-burst']
    train_df = shuffle(pd.concat((train_df_non_burst.sample(n=int(len(train_df_burst) * settings.LABEL_RATIO)), train_df_burst),
                                 ignore_index=True))

# print(len(train_df[train_df['label'] == 'burst']))
# print(len(train_df[train_df['label'] == 'non-burst']))

# load vocab
user_vocab = Vocab(vocab_file=settings.USER_VOCAB_FN)
sub_vocab = Vocab(vocab_file=settings.SUB_VOCAB_FN)
words, word_vectors = load_glove_emb(fn=settings.GLOVE_EMBEDDING_FN)
word_vocab = Vocab(words=list(range(len(words))), additional_terms=False)
label_vocab = Vocab(words=['non-burst', 'burst'], additional_terms=False)

# make dataset
train_ds = RedditDataset(
    df=train_df,
    user_vocab=user_vocab,
    sub_vocab=sub_vocab,
    word_vocab=word_vocab,
    label_vocab=label_vocab,
    content_col='content',
)

dev_ds = RedditDataset(
    df=dev_df,
    user_vocab=user_vocab,
    sub_vocab=sub_vocab,
    word_vocab=word_vocab,
    label_vocab=label_vocab,
    content_col='content',
)

test_ds = RedditDataset(
    df=test_df,
    user_vocab=user_vocab,
    sub_vocab=sub_vocab,
    word_vocab=word_vocab,
    label_vocab=label_vocab,
    content_col='content',
)

# make dataloader
collate_fn = CollateFn(word_padding_index=word_vocab.padding_index)

train_dl = DataLoader(
    dataset=train_ds,
    batch_size=settings.TRAIN_BS,
    shuffle=True,
    drop_last=True,
    collate_fn=collate_fn,
)

dev_dl = DataLoader(
    dataset=dev_ds,
    batch_size=settings.DEV_BS,
    shuffle=True,
    drop_last=False,
    collate_fn=collate_fn,
)

test_dl = DataLoader(
    dataset=test_ds,
    batch_size=settings.TEST_BS,
    shuffle=True,
    drop_last=False,
    collate_fn=collate_fn,
)

# test dl
# train_dl_it = iter(train_dl)
# test_dl_it = iter(test_dl)
# dev_dl_it = iter(dev_dl)

# for it in [train_dl_it, test_dl_it, dev_dl_it]:
#     (users, source_subs, target_subs, contents), labels = next(it)
#     # print(users, source_subs, target_subs, contents, labels)
#     print(users.shape, source_subs.shape, target_subs.shape, contents.shape, labels.shape)

# get embedding
user_vectors = np.load(settings.USER_VECS_FN)
user_vectors = np.concatenate((
    user_vectors,
    np.random.normal(loc=0, scale=1 / user_vectors.shape[1] ** 0.5, size=(1, user_vectors.shape[1])),
    np.zeros((1, user_vectors.shape[1]), dtype=np.float),
), axis=0)
user_embedding = Embedding.from_pretrained(
    vectors=np.ascontiguousarray(user_vectors, dtype=np.float32),
    device=settings.DEVICE,
    autograd=True,
    padding_index=user_vocab.padding_index,
)

sub_vectors = np.load(settings.SUB_VECS_FN)
sub_vectors = np.concatenate((
    sub_vectors,
    np.random.normal(loc=0, scale=1 / sub_vectors.shape[1] ** 0.5, size=(1, sub_vectors.shape[1])),
    np.zeros((1, sub_vectors.shape[1]), dtype=np.float),
), axis=0)
sub_embedding = Embedding.from_pretrained(
    vectors=np.ascontiguousarray(sub_vectors, dtype=np.float32),
    device=settings.DEVICE,
    autograd=True,
    padding_index=sub_vocab.padding_index,
)

word_vectors = np.concatenate((
    word_vectors,
    np.random.normal(loc=0, scale=1 / word_vectors.shape[1] ** 0.5, size=(1, word_vectors.shape[1])),
    np.zeros((1, word_vectors.shape[1]), dtype=np.float32),
), axis=0)
word_embedding = Embedding.from_pretrained(
    vectors=np.ascontiguousarray(word_vectors, dtype=np.float32),
    device=settings.DEVICE,
    autograd=True,
    padding_index=word_vocab.padding_index,
)

# build model
model = RedditModel(
    user_embedding=user_embedding,
    sub_embedding=sub_embedding,
    word_embedding=word_embedding,
    hidden_dim=settings.HIDDEN_DIM,
    output_dim=len(label_vocab),
    device=settings.DEVICE,
    p_dropout=settings.P_DROPOUT,
)

criterion = CrossEntropyLoss()
# criterion = BCEWithLogitsLoss()
# optimizer = SGD(parameters=model.get_parameters(), lr=settings.LR, beta=0.9)
optimizer = Adam(
    parameters=model.get_parameters(),
    lr=settings.LR,
    lr_scheduler=None,
)

# train
epoch_bar = tqdm(range(settings.NUM_EPOCHS), position=0)

for epoch in epoch_bar:
    epoch_bar.set_description(f'Epoch {epoch}')
    # epoch_losses = []
    total_train_loss = 0
    model.train()
    train_batch_bar = tqdm(train_dl, position=1)
    optimizer.lr_update()
    for i, (x, y) in enumerate(train_batch_bar):
        output = model(x)
        loss = criterion(output, target=y)
        loss.backward()
        optimizer.step(zero=True)
        total_train_loss += loss.data
        train_batch_bar.set_description(f'Train batch {i}: loss = {total_train_loss / (i + 1)}')
    train_avg_loss = total_train_loss / len(train_dl)
    train_batch_bar.close()

    # eval dev
    eval_preds = []
    eval_trues = []
    total_eval_loss = 0
    model.eval()
    eval_batch_bar = tqdm(dev_dl, position=1)
    eval_outputs = []
    for i, (x, y) in enumerate(eval_batch_bar):
        output = model(x)
        eval_outputs.append(output.cpu().data)
        loss = criterion(output, target=y)
        preds = output.argmax(dim=1).cpu().data.astype(np.int32)
        eval_preds.extend(preds)
        eval_trues.extend(y.cpu().data)
        total_eval_loss += loss.data
        eval_batch_bar.set_description(f'Eval batch {i}: loss = {total_eval_loss / (i + 1)}')

    eval_avg_loss = total_eval_loss / len(dev_dl)

    #acc = accuracy_score(y_true=eval_trues, y_pred=eval_preds)
    #f1 = f1_score(y_true=eval_trues, y_pred=eval_preds, labels=[0, 1])
    auc_score = roc_auc_score(y_true=eval_trues, y_score=np.concatenate(eval_outputs, axis=0)[:, 1])
    eval_batch_bar.close()

    # eval test
    test_preds = []
    test_trues = []
    total_test_loss = 0
    model.eval()
    test_batch_bar = tqdm(test_dl, position=1)
    test_outputs = []
    for i, (x, y) in enumerate(test_batch_bar):
        output = model(x)
        test_outputs.append(output.cpu().data)
        loss = criterion(output, target=y)
        preds = output.argmax(dim=1).cpu().data.astype(np.int32)
        test_preds.extend(preds)
        test_trues.extend(y.cpu().data)
        total_test_loss += loss.data
        test_batch_bar.set_description(f'Test batch {i}: loss = {total_test_loss / (i + 1)}')

    test_avg_loss = total_test_loss / len(test_dl)

    #test_acc = accuracy_score(y_true=test_trues, y_pred=test_preds)
    #test_f1 = f1_score(y_true=test_trues, y_pred=test_preds, labels=[0, 1])
    test_auc_score = roc_auc_score(y_true=test_trues, y_score=np.concatenate(test_outputs, axis = 0)[:, 1])
    test_batch_bar.close()

    epoch_bar.write(f'Epoch {epoch}: \ntrain_avg_loss = {train_avg_loss}\r')
    epoch_bar.write(f'Eval: avg_loss = {eval_avg_loss}, auc_score = {auc_score}\r')
    epoch_bar.write( f'Test: avg_loss = {test_avg_loss}, auc_score = {test_auc_score}\r')
    epoch_bar.write('-' * 30)

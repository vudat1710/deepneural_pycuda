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
train_df_burst = train_df[train_df['label'] == 'burst']
train_df_non_burst = train_df[train_df['label'] == 'non-burst']

# expected_len = len(train_df_non_burst) * 3 // 7
# train_df_burst = pd.concat([train_df_burst] * (expected_len // len(train_df_burst)), ignore_index=True)
#
train_df = shuffle(pd.concat((train_df_non_burst.sample(n=int(len(train_df_burst) * 1.3)), train_df_burst), ignore_index=True))

print(len(train_df[train_df['label'] == 'burst']))
print(len(train_df[train_df['label'] == 'non-burst']))

# load vocab
user_vocab = Vocab(vocab_file=settings.USER_VOCAB_FN)
sub_vocab = Vocab(vocab_file=settings.SUB_VOCAB_FN)
words, word_vectors = load_glove_emb(fn=settings.GLOVE_EMBEDDING_FN)
word_vocab = Vocab(words=words)
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
    drop_last=True,
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
user_vectors = np.concatenate((np.zeros((2, user_vectors.shape[1]), dtype=np.float), user_vectors), axis=0)
user_embedding = Embedding.from_pretrained(
    vectors=user_vectors,
    device=settings.DEVICE,
    autograd=False,
)

sub_vectors = np.load(settings.SUB_VECS_FN)
sub_vectors = np.concatenate((np.zeros((2, sub_vectors.shape[1]), dtype=np.float), sub_vectors), axis=0)
sub_embedding = Embedding.from_pretrained(
    vectors=sub_vectors,
    device=settings.DEVICE,
    autograd=False,
)

word_vectors = np.concatenate((np.zeros((2, word_vectors.shape[1]), dtype=np.float), word_vectors), axis=0)
word_embedding = Embedding.from_pretrained(
    vectors=word_vectors,
    device=settings.DEVICE,
    autograd=False,
)

# build model
model = RedditModel(
    user_embedding=user_embedding,
    sub_embedding=sub_embedding,
    word_embedding=word_embedding,
    hidden_dim=300,
    output_dim=len(label_vocab),
    device=settings.DEVICE,
    p_dropout=0.5,
)

criterion = CrossEntropyLoss()
optimizer = SGD(parameters=model.get_parameters(), lr=settings.LR)

# train
epoch_bar = tqdm(range(settings.NUM_EPOCHS), position=0)

for epoch in epoch_bar:
    epoch_bar.set_description(f'Epoch {epoch}')
    # epoch_losses = []
    total_train_loss = 0
    model.train()
    train_batch_bar = tqdm(train_dl, position=1)
    for i, (x, y) in enumerate(train_batch_bar):
        output = model(x)
        loss = criterion(output, target=y)
        loss.backward()
        optimizer.step(zero=True)
        # epoch_losses.append(loss.data)
        total_train_loss += loss.data
        train_batch_bar.set_description(f'Train batch {i}: loss = {total_train_loss / (i + 1)}')
    train_avg_loss = total_train_loss / (len(train_dl) + 1)
    train_batch_bar.close()

    eval_preds = []
    eval_trues = []
    # eval_losses = []
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
        # eval_losses.append(loss.data)
        total_eval_loss += loss.data
        eval_batch_bar.set_description(f'Eval batch {i}: loss = {total_eval_loss / (i + 1)}')

    eval_avg_loss = total_eval_loss / (len(dev_dl) + 1)

    acc = accuracy_score(y_true=eval_trues, y_pred=eval_preds)
    f1 = f1_score(y_true=eval_trues, y_pred=eval_preds, labels=[0, 1])
    auc_score = roc_auc_score(y_true=eval_trues, y_score=np.concatenate(eval_outputs, axis=0)[:, 1])
    eval_batch_bar.close()

    # train_batch_bar.reset()
    # eval_batch_bar.reset()

    # epoch_bar.write(f'Epoch {epoch}: \ntrain_avg_loss = {np.average(epoch_losses)}\r')
    # epoch_bar.write(f'Eval: avg_loss = {np.average(eval_losses)}, accuracy = {acc}, f1_score = {f1}\r')
    epoch_bar.write(f'Epoch {epoch}: \ntrain_avg_loss = {train_avg_loss}\r')
    epoch_bar.write(f'Eval: avg_loss = {eval_avg_loss}, accuracy = {acc}, f1_score = {f1}, auc_score = {auc_score}\r')
    epoch_bar.write('-' * 30)

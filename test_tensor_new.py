from model.tensor_new import Tensor, LSTMCell, Embedding, CrossEntropyLoss, SGD, LSTMLayer, Linear, Layer, Sequential
from model.vocab import Vocab, load_glove_emb
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as f
from tqdm import tqdm
import time


def test_basic_ops(x: Tensor, y: Tensor):
    print('x = ', x)
    print('y = ', y)
    print('x + y = ', x + y)
    print('x.sigmoid = ', x.sigmoid())
    print('x.tanh = ', x.tanh())
    # print('x.softmax = ', x.softmax())
    print('2 * x = ', Tensor(2) * x)
    print('x * y = ', x * y)
    print('x - y = ', x - y)
    print('x - 2 = ', x - Tensor(2, device='cuda'))
    print('x + 2 = ', x + Tensor(2, device='cuda'))
    print('-x = ', -x)
    print('x.sum = ', x.sum(0))
    print('x.expand(0, 3) = ', x.expand(0, 3))
    print('x.expand(1, 3) = ', x.expand(1, 3))


def test_mm():
    x1 = x.expand(0, 3)
    x2 = x.expand(1, 3)
    print(x1.mm(x2))
    x = Tensor(np.random.rand(1, 3).astype(np.float32), device='cuda')
    y = Tensor(np.random.rand(3, 1).astype(np.float32), device='cuda')
    print(x, y)
    x_t = x.transpose()
    y_t = y.transpose()
    print(x_t.mm(y_t))
    x = Tensor(np.random.rand(6), device='cuda')
    y = Tensor(np.random.rand(6), device='cuda')
    x.reshape((2, 3)).mm(y.reshape((3, 2)))


def test_backward():
    x_np = np.random.rand(2, 3)
    w_np = np.random.rand(3, 5)
    print(x_np)
    print(w_np)
    x = Tensor(x_np, device='cuda', autograd=True)
    w = Tensor(w_np, device='cuda', autograd=True)
    res = (x - Tensor(1, d_type=np.float32, device='cuda', autograd=True)).mm(w)
    print('res_cuda = ', res)
    print('res_softmax = ', res.softmax())
    loss = res.cross_entropy(Tensor([1, 2], device='cpu', d_type=np.int32))
    print('loss_cuda = ', loss)
    loss.backward()
    print('x.grad = ', x.grad)
    print('w.grad = ', w.grad)

    # torch
    print('-------------------------')
    x_torch = torch.Tensor(x_np)
    w_torch = torch.Tensor(w_np)
    y_torch = torch.Tensor([1, 2])
    res_torch = (x_torch - 1).mm(w_torch)
    print('res_torch = ', res_torch)
    print('res_torch_softmax = ', f.softmax(res_torch, dim=1))
    loss_func = nn.NLLLoss()
    loss_torch = loss_func(f.softmax(res_torch, dim=1), y_torch.long())
    print('loss_torch', loss_torch)


def test_index_select():
    x_cpu = np.random.rand(10, 5)
    x = Tensor(x_cpu, device='cuda', autograd=True)
    indices = Tensor([1, 2, 3], device='cpu', d_type=np.int32)
    embs = x.index_select(indices)
    print(x)
    print(embs)
    # print(x_cpu[[1, 2, 3]])


def test_get_item():
    x_cpu = np.random.rand(10, 5, 5)
    x = Tensor(x_cpu, device='cuda')
    print(x[:, :, 1])


def test_lstm_cell():
    embeddings = Embedding.init(
        vocab_size=10,
        embedding_dim=5,
        device='cuda',
        autograd=True,
    )
    lstm_cell = LSTMCell(
        input_dim=5,
        hidden_dim=100,
        device='cuda',
    )
    print('weight before backward')
    print(embeddings.weight)

    x = embeddings(Tensor([[1, 2, 3],
                           [2, 3, 4]], device='cpu'))
    print('x')
    print(x)
    hidden = None
    for time_step in x:
        _, hidden = lstm_cell(time_step, hidden=hidden)
    target = Tensor([3, 5, 2], device='cpu', d_type=np.int32)
    criterion = CrossEntropyLoss()
    optimizer = SGD(parameters=[
        *embeddings.get_parameters(),
        *lstm_cell.get_parameters(),
    ],
        lr=0.01,
    )
    loss = criterion(hidden[0], target)
    print('loss = ', loss)
    loss.backward()
    optimizer.step(zero=True)
    print('weight after backward')
    print(embeddings.weight)


def test_mm_graph():
    x = Tensor(np.random.rand(10, 4).astype(np.float32), device='cuda', autograd=True)
    y = Tensor(np.random.rand(4, 5).astype(np.float32), device='cuda', autograd=True)
    res = x.mm(y)
    print(f'x: {x.children}')
    print(f'y: {y.children}')
    print(f'res: {res.children}')


def test_lstm_layer():
    embeddings = Embedding.init(
        vocab_size=10,
        embedding_dim=5,
        device='cuda',
        autograd=True,
    )
    lstm = LSTMLayer(
        input_dim=5,
        hidden_dim=100,
        device='cuda',
    )
    h2o = Linear(
        n_inputs=100,
        n_outputs=10,
        bias=True,
        device='cuda',
    )
    criterion = CrossEntropyLoss()
    optimizer = SGD(parameters=[
        *embeddings.get_parameters(),
        *lstm.get_parameters(),
        *h2o.get_parameters(),
    ])
    print(len(optimizer.parameters))
    x = embeddings(Tensor([[1, 2, 3],
                           [2, 3, 4]], device='cpu'))
    target = Tensor([3, 5, 2], device='cpu', d_type=np.int32)
    output = h2o(lstm(x)[0][-1])
    loss = criterion(input=output, target=target)
    loss.backward()
    print('embedding before backward')
    print(embeddings.weight)
    optimizer.step()
    print('--------------')
    print('embedding after backward')
    print(embeddings.weight)


class Model(Layer):
    def __init__(
            self,
            embedding,
            hidden_dim,
            output_dim,
            device='cuda',
            **kwargs,
    ):
        super(Model, self).__init__()
        self.device = device
        if embedding is not None:
            self.embedding = embedding
            self.embedding_dim = embedding.embedding_dim
        else:
            vocab_size = kwargs.get('vocab_size')
            embedding_dim = kwargs.get('embedding_dim')
            assert vocab_size is not None, 'vocab_size is required'
            assert embedding_dim is not None, 'embedding_dim is required'
            self.embedding_dim = embedding_dim
            self.embedding = Embedding.init(
                vocab_size=vocab_size,
                embedding_dim=embedding_dim,
                device=device,
                autograd=False,
            )

        self.lstm = LSTMLayer(
            input_dim=self.embedding_dim,
            hidden_dim=hidden_dim,
            device=device,
        )

        self.h2o = Linear(
            n_inputs=hidden_dim,
            n_outputs=output_dim,
            bias=True,
            device=device,
        )

    def forward(self, input):
        hs, _ = self.lstm(self.embedding(input=input))
        output = self.h2o(hs[-1])
        return output

    def __call__(self, input):
        return self.forward(input)


def test_build_model():
    model = Model(
        embedding=None,
        hidden_dim=300,
        output_dim=10,
        device='cuda',
        embedding_dim=300,
        vocab_size=10000,
    )

    x = Tensor([[*range(i, i + 20)] for i in range(20)], device='cpu')
    target = Tensor(np.random.randint(0, 10, 20), device='cpu', d_type=np.int32, autograd=True)

    criterion = CrossEntropyLoss()
    optimizer = SGD(parameters=model.get_parameters())
    for _ in tqdm(range(0, 10)):
        output = model(x)
        loss = criterion(output, target)
        t1 = time.time()
        loss.backward()
        t2 = time.time()
        print(f'time to backward loss: {t2 - t1}')

        t1 = time.time()
        optimizer.step()
        t2 = time.time()
        print(f'time to step: {t2 - t1}')
    # print('embedding before backward')
    # print(model.embedding.weight)
    # print('-------------------')
    # print('embedding after backward')
    # print(model.embedding.weight)


def test_vocab():
    user_vocab = Vocab(vocab_file='data/prediction/embeddings/user_vecs.vocab')
    print(f'user vocab length: {len(user_vocab)}')
    print([user_vocab[i] for i in range(5)])
    user_vectors = np.load('data/prediction/embeddings/user_vecs.npy')
    user_vectors = np.concatenate((np.zeros((2, user_vectors.shape[1]), dtype=np.float), user_vectors), axis=0)
    print(f'user vectors shape: {user_vectors.shape}')
    print('-' * 30)

    sub_vocab = Vocab(vocab_file='data/prediction/embeddings/sub_vecs.vocab')
    print(f'sub vocab length: {len(sub_vocab)}')
    print([sub_vocab[i] for i in range(5)])
    sub_vectors = np.load('data/prediction/embeddings/sub_vecs.npy')
    sub_vectors = np.concatenate((np.zeros((2, sub_vectors.shape[1]), dtype=np.float), sub_vectors), axis=0)
    print(f'sub vectors shape: {sub_vectors.shape}')
    print('-' * 30)

    words, word_vectors = load_glove_emb('data/prediction/embeddings/glove_word_embeds.txt')
    word_vectors = np.concatenate((np.zeros((2, word_vectors.shape[1]), dtype=np.float), word_vectors), axis=0)
    word_vocab = Vocab(words=words)
    print(f'word vocab length: {len(word_vocab)}')
    print([word_vocab[i] for i in range(5)])
    print(f'word vectors shape: {word_vectors.shape}')


# test softmax
def test_softmax():
    data = np.random.rand(5, 2)
    x_gpu = Tensor(data=data, device='cuda')
    x_cpu = Tensor(data=data, device='cpu')

    print(x_gpu.softmax())
    print(x_cpu.softmax())


# test dropout
def test_dropout():
    data = np.random.rand(10, 10)
    x_gpu = Tensor(data=data, device='cuda')
    x_dropout = x_gpu.dropout(0.1).cpu()
    print(x_gpu)
    print(x_dropout)
    print(x_dropout.shape)
    print(np.sum(x_dropout.data == 0))


def test_relu():
    data = np.random.rand(10, 10) - 0.5
    x_gpu = Tensor(data, device='cuda')
    x_cpu = Tensor(data, device='cpu')
    x_relu = x_gpu.relu()
    print(x_relu)
    print(x_relu.relu_grad())
    print(np.sum(np.sum(x_relu.cpu().data == 0)))

    # print(x_gpu.tanh())
    # print((np.exp(data) - np.exp(-data))/(np.exp(data) + np.exp(-data)))

    # print(x_gpu.sigmoid())
    # print(1/(1 + np.exp(-data)))

    # print(x_gpu.sigmoid_grad())
    # print(data * (1 - data))


def test_expand():
    data = np.array([1, 2, 3])
    x_gpu = Tensor(data, device='cuda')
    x_cpu = Tensor(data, device='cpu')
    print(x_gpu.expand(dim=0, copies=5))
    print(x_cpu.expand(dim=0, copies=5))
    print('norm', x_gpu.norm())


def test_gpu_vs_cpu():
    vectors = np.random.normal(0, 1, size=(20, 8)) * (2/28) ** 0.5
    vectors[0, :] = 0

    embedding_gpu = Embedding.from_pretrained(
        vectors=vectors,
        padding_index=0,
        device='cuda',
        autograd=True,
    )

    embedding_cpu = Embedding.from_pretrained(
        vectors=vectors,
        padding_index=0,
        device='cpu',
        autograd=True,
    )

    indices = Tensor([[0, 1, 0, 4, 5],
                      [1, 4, 0, 1, 2]], device='cpu', d_type=np.int32)
    embeds_gpu = embedding_gpu(indices)
    embeds_cpu = embedding_cpu(indices)

    linear_cpu = Linear(8, 2, device='cpu', bias=True)
    linear_gpu = Linear(8, 2, device='cuda', bias=True)
    linear_gpu.weight = linear_cpu.weight.to('cuda')
    linear_gpu.bias = linear_cpu.bias.to('cuda')
    # print(embeds_gpu[0].shape)

    out_cpu = linear_cpu(embeds_cpu[0])
    out_gpu = linear_gpu(embeds_gpu[0])
    # print(out_cpu)
    # print(out_gpu)

    target = Tensor([1, 0, 1, 0, 0], device='cpu', d_type=np.int32)
    loss_gpu = out_gpu.cross_entropy(target)
    loss_cpu = out_cpu.cross_entropy(target)
    print(loss_cpu, loss_gpu)
    loss_gpu.backward()
    loss_cpu.backward()

    print(linear_gpu.weight.grad)
    print(linear_cpu.weight.grad)

    print(embedding_gpu.weight.grad)
    print(embedding_cpu.weight.grad)


def test_index_select_v2():
    x_cpu = np.random.rand(8, 5)
    x = Tensor(x_cpu, device='cuda', autograd=True)
    indices = Tensor([[1, 2, 3], [0, 0, 0]], device='cuda', d_type=np.int32)
    embs = x.index_selectv2(indices)
    print(x)
    print(embs)
    # print(x_cpu[[1, 2, 3]])


if __name__ == '__main__':
    # test basic op
    # x = Tensor([1, 2, 3], device='cuda', autograd=True)
    # y = Tensor([4, 5, 6], device='cuda', autograd=True)
    # test_basic_ops(x, y)

    # test backward
    # test_backward()

    # test index select
    # test_index_select()

    # test get item
    # test_get_item()

    # test mm graph
    # test_mm_graph()

    # test lstm cell
    # test_lstm_cell()

    # test lstm layer
    # test_lstm_layer()

    # test build model
    # test_build_model()

    # test vocab
    # test_vocab()

    # test softmax
    # test_softmax()

    # test dropout
    test_dropout()

    # test relu
    # test_relu()

    # test expand
    # test_expand()

    # test_gpu_vs_cpu()

    # test_index_select_v2()

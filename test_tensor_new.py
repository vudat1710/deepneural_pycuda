from model.tensor_new import Tensor, LSTMCell, Embedding, CrossEntroyLoss, SGD
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as f


def test_basic_ops(x: Tensor, y: Tensor):
    print('x = ', x)
    print('y = ', y)
    print('x + y = ', x + y)
    print('x.sigmoid = ', x.sigmoid())
    print('x.tanh = ', x.tanh())
    print('x.softmax = ', x.softmax())
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
    embeddings = Embedding(
        vocab_size=10,
        embedding_dim=5,
        device='cuda',
        autograd=True,
    )
    lstm_cell = LSTMCell(
        input_dim=5,
        hidden_dim=100,
        output_dim=10,
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
        hidden = lstm_cell(time_step, hidden=hidden)
    target = Tensor([3, 5, 2], device='cpu', d_type=np.int32)
    criterion = CrossEntroyLoss()
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

    # test lstm cell
    test_lstm_cell()

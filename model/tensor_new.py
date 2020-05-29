import numpy as np
import pycuda.autoinit
from pycuda import gpuarray
# from utils.utils import *
# from utils.activation import *
from model.operations import *
from skcuda import misc
from skcuda import linalg
import time

__all__ = [
    'Tensor',
    'Linear',
    'Embedding',
    'LSTMLayer',
    'LSTMCell',
    'SGD',
    'Sequential',
    'Sigmoid',
    'Tanh',
    'ReLu',
    'CrossEntropyLoss',
    'BCEWithLogitsLoss',
    'Layer',
    'Dropout',
    'LSTMDropout',
    'Adam',
]
linalg.init()

DEVICE = 'cuda'
dtype = np.float32
EPSILON = 1e-12
A_MIN = EPSILON
A_MAX = 1 - EPSILON


def xavier_init(n_input, n_output):
    return np.random.normal(loc=0, scale=1, size=(n_input, n_output)) * np.sqrt(2 / (n_input + n_output))


def truncated_normal(mean, std, out_shape):
    samples = np.random.normal(loc=mean, scale=std, size=out_shape)
    reject = np.logical_or(samples >= mean + 2 * std, samples <= mean - 2 * std)
    while any(reject.flatten()):
        resamples = np.random.normal(loc=mean, scale=std, size=reject.sum())
        samples[reject] = resamples
        reject = np.logical_or(samples >= mean + 2 * std, samples <= mean - 2 * std)
    return samples


class Tensor:
    def __init__(
            self,
            data,
            autograd=False,
            creators=None,
            creation_op=None,
            tensor_id=None,
            device='cuda',
            d_type=None,
    ):
        assert device in [None, 'cuda', 'cpu'], 'wrong device'
        assert d_type in [None, np.int, np.int32, np.float32, np.float64]
        self.device = device if device is not None else 'cpu'
        if type(data) is gpuarray.GPUArray:
            self.data = data if device == 'cuda' else data.get()
        elif type(data) in [list, np.ndarray, set, tuple]:
            data = np.array(data, dtype=dtype if d_type is None else d_type)
            self.data = gpuarray.to_gpu(data) if self.device == 'cuda' else data
        elif type(data) in [float, np.float32, np.float64]:
            data = dtype(data) if d_type is None else d_type(data)
            self.data = data
        elif type(data) == int:
            data = np.int32(data) if d_type is None else d_type(data)
            self.data = data
        else:
            raise Exception("wrong data format")

        self.autograd = autograd
        self.grad = None

        self.tensor_id = tensor_id if tensor_id is not None else np.random.randint(0, 1e9)

        self.creators = creators
        self.creator_op = creation_op
        self.children = {}

        if creators is not None:
            for c in creators:
                if self.tensor_id not in c.children:
                    c.children[self.tensor_id] = 1
                else:
                    c.children[self.tensor_id] += 1

    def all_children_grads_accounted_for(self):
        for tensor_id, cnt in self.children.items():
            if cnt != 0:
                return False
        return True

    def backward(self, grad=None, grad_origin=None):
        if self.autograd:
            if grad is None:
                grad = self.ones_like(self)

            if grad_origin is not None:
                if self.children[grad_origin.tensor_id] == 0:
                    # return
                    # print(f'{self.children} {grad_origin.tensor_id}')
                    raise Exception("cannot backprop anymore")
                else:
                    self.children[grad_origin.tensor_id] -= 1

            if self.grad is None:
                self.grad = grad
            else:
                self.grad.add_(other=grad)

            # grads must not have grads of their own
            assert grad.autograd is False
            assert self.grad.autograd is False

            if (self.creators is not None and
                    (self.all_children_grads_accounted_for() or grad_origin is None)):
                if self.creator_op == "add":
                    self.creators[0].backward(grad=self.grad, grad_origin=self)
                    self.creators[1].backward(grad=self.grad, grad_origin=self)
                elif self.creator_op == "sub":
                    self.creators[0].backward(grad=self.grad, grad_origin=self)
                    self.creators[1].backward(grad=-self.grad, grad_origin=self)
                elif self.creator_op == "mul":
                    new = self.grad * self.creators[1]
                    self.creators[0].backward(grad=new, grad_origin=self)
                    new = self.grad * self.creators[0]
                    self.creators[1].backward(grad=new, grad_origin=self)
                elif self.creator_op == "mm":
                    c0, c1 = self.creators
                    new = self.grad.mm(c1.transpose())
                    c0.backward(grad=new, grad_origin=self)
                    new = self.grad.transpose().mm(c0).transpose()
                    c1.backward(grad=new, grad_origin=self)
                elif self.creator_op == "transpose":
                    self.creators[0].backward(self.grad.transpose(), grad_origin=self)
                elif "sum" in self.creator_op:
                    dim = int(self.creator_op.split("_")[1])
                    grad_expand = self.grad.expand(dim, self.creators[0].data.shape[dim])
                    self.creators[0].backward(
                        grad_expand,
                        grad_origin=self,
                    )
                elif "expand" in self.creator_op:
                    dim = int(self.creator_op.split("_")[1])
                    grad_sum = self.grad.sum(dim)
                    self.creators[0].backward(grad_sum, grad_origin=self)
                elif self.creator_op == 'neg':
                    grad_neg = - self.grad
                    self.creators[0].backward(grad_neg, grad_origin=self)
                elif self.creator_op == "sigmoid":
                    grad_sigmoid = self.grad * self.sigmoid_grad()
                    self.creators[0].backward(grad_sigmoid, grad_origin=self)
                elif self.creator_op == "tanh":
                    grad_tanh = self.grad * self.tanh_grad()
                    self.creators[0].backward(grad_tanh, grad_origin=self)
                elif self.creator_op == "relu":
                    grad_relu = self.grad * self.relu_grad()
                    self.creators[0].backward(grad_relu, grad_origin=self)
                elif self.creator_op == "index_select":
                    new_grad = self.zeros_like(self.creators[0])
                    if self.device == 'cuda':
                        indices_ = self.__getattribute__("index_select_indices").data.get().flatten().tolist()
                    else:
                        indices_ = self.__getattribute__("index_select_indices").data.flatten().tolist()
                    padding_index = self.__getattribute__('padding_index')
                    grad_ = grad.reshape((len(indices_), -1))
                    for i in range(len(indices_)):
                        # new_grad[indices_[i]] += grad_[i].to(self.creators[0].device)
                        if indices_[i] != padding_index:
                            # new_grad.data[indices_[i]] += grad_.data[i]
                            add_(new_grad.data[indices_[i]], grad_.data[i])
                    self.creators[0].backward(new_grad, grad_origin=self)
                elif self.creator_op == "cross_entropy":
                    dx = self.__getattribute__("softmax_output") - self.__getattribute__("target_dist")
                    self.creators[0].backward(Tensor(data=dx, device=self.creators[0].device), grad_origin=self)
                elif self.creator_op == "bce":
                    dx = self.__getattribute__("dx")
                    self.creators[0].backward(Tensor(data=dx, device=self.creators[0].device), grad_origin=self)

    def __add__(self, other):
        assert self.device == other.device, f"expect {self.device} assert trigger but {other.device}"
        assert self.data.shape == other.data.shape or self.shape == () or other.shape == (), f"{self.data.shape} not match {other.data.shape}"
        if self.autograd and other.autograd:
            return Tensor(
                data=self.data + other.data,
                autograd=True,
                creators=[self, other],
                creation_op='add',
                device=self.device,
            )
        return Tensor(
            data=self.data + other.data,
            device=self.device,
        )

    def add_(self, other):
        assert self.device == other.device, f"expect {self.device} assert trigger but {other.device}"
        assert self.data.shape == other.data.shape or self.shape == () or other.shape == (), f"{self.data.shape} not match {other.data.shape}"
        if self.device == 'cuda':
            data = add_(self.data, other.data)
        else:
            data = self.data + other.data
        return Tensor(
            data=data,
            device=self.device,
            autograd=self.autograd
        )

    def __neg__(self):
        if self.autograd:
            return Tensor(
                data=self.data * -1,
                autograd=True,
                creators=[self],
                creation_op='neg',
                device=self.device,
            )
        return Tensor(data=self.data * -1, device=self.device)

    def __sub__(self, other):
        assert self.device == other.device, f"expect {self.device} assert trigger but {other.device}"
        assert self.data.shape == other.data.shape or self.shape == () or other.shape == (), f"{self.data.shape} not match {other.data.shape}"
        if self.autograd and other.autograd:
            return Tensor(
                data=self.data - other.data,
                autograd=True,
                creators=[self, other],
                creation_op="sub",
                device=self.device,
            )
        return Tensor(
            data=self.data - other.data,
            device=self.device,
        )

    def sub_(self, other):
        assert self.device == other.device, f"expect {self.device} assert trigger but {other.device}"
        assert self.data.shape == other.data.shape or self.shape == () or other.shape == (), f"{self.data.shape} not match {other.data.shape}"
        if self.device == 'cuda':
            data = sub_(self.data, other.data, 1)
        else:
            data = self.data - other.data
        return Tensor(
            data=data,
            autograd=self.autograd,
            device=self.device,
        )

    def __mul__(self, other):
        assert self.device == other.device, f"expect {self.device} assert trigger but {other.device}"
        assert self.data.shape == other.data.shape or self.shape == () or other.shape == (), f"{self.data.shape} not match {other.data.shape}"
        # assert self.data.shape == other.data.shape or type(self.data) in [np.float32, np.float64] or type(
        #     other.data) in [np.float32, np.float64], f"{self.data.shape} not match {other.data.shape}"
        if self.autograd and other.autograd:
            return Tensor(
                data=self.data * other.data,
                autograd=True,
                creators=[self, other],
                creation_op="mul",
                device=self.device,
            )
        return Tensor(
            data=self.data * other.data,
            device=self.device,
        )

    def sum(self, dim):
        if self.device == 'cuda':
            data = sum_gpu(
                x_gpu=self.data,
                axis=dim,
            )
        else:
            data = self.data.sum(dim)
        if self.autograd:
            return Tensor(
                data=data,
                autograd=True,
                creators=[self],
                creation_op="sum_" + str(dim),
                device=self.device,
            )
        return Tensor(
            data=data,
            device=self.device,
        )

    def expand(self, dim, copies):
        if self.device == "cuda":
            data = expand_gpu(x=self.data, dim=dim, copies=copies)
        else:
            trans_dims = list(range(0, len(self.data.shape)))
            trans_dims.insert(dim, len(self.data.shape))
            data = (self.data.repeat(copies)
                    .reshape(list(self.data.shape) + [copies]).transpose(trans_dims))

        if self.autograd:
            return Tensor(
                data=data,
                autograd=True,
                creators=[self],
                creation_op="expand_" + str(dim),
                device=self.device,
            )
        return Tensor(
            data=data,
            device=self.device,
        )

    def transpose(self):
        if self.device == 'cuda':
            data = linalg.transpose(self.data)
        else:
            data = self.data.transpose()
        if self.autograd:
            return Tensor(
                data=data,
                autograd=True,
                creators=[self],
                creation_op="transpose",
                device=self.device,
            )
        return Tensor(
            data=data,
            device=self.device,
        )

    def mm(self, x):
        assert self.device == x.device, f"expect {self.device} assert trigger but {x.device}"
        assert self.data.shape[1] == x.data.shape[0]
        if self.device == 'cuda':
            data = linalg.dot(
                x_gpu=self.data,
                y_gpu=x.data,
                transa='n',
                transb='n',
            )
        else:
            data = self.data.dot(x.data)
        if self.autograd:
            return Tensor(
                data=data,
                autograd=True,
                creators=[self, x],
                creation_op="mm",
                device=self.device,
            )
        return Tensor(
            data=data,
            device=self.device,
        )

    def sigmoid(self):
        if self.device == 'cuda':
            data = sigmoid_gpu(x=self.data)
        else:
            data = sigmoid(x=self.data)
        if self.autograd:
            return Tensor(
                data=data,
                autograd=True,
                creators=[self],
                creation_op="sigmoid",
                device=self.device,
            )
        return Tensor(
            data=data,
            device=self.device,
        )

    def sigmoid_grad(self):
        if self.device == 'cuda':
            data = sigmoid_grad_gpu(x=self.data)
        else:
            data = self.data * (1 - self.data)
        return Tensor(
            data=data,
            autograd=False,
            device=self.device,
        )

    def tanh(self):
        if self.device == 'cuda':
            data = tanh_gpu(x=self.data)
        else:
            data = tanh(x=self.data)
        if self.autograd:
            return Tensor(
                data=data,
                autograd=True,
                creators=[self],
                creation_op="tanh",
                device=self.device,
            )
        return Tensor(
            data=data,
            device=self.device,
        )

    def tanh_grad(self):
        if self.device == 'cuda':
            data = tanh_grad_gpu(x=self.data)
        else:
            data = 1 - self.data ** 2
        return Tensor(
            data=data,
            autograd=False,
            device=self.device,
        )

    def relu(self):
        if self.device == 'cuda':
            data = relu_gpu(self.data)
        else:
            data = relu(self.data)
        if self.autograd:
            return Tensor(
                data=data,
                autograd=True,
                creators=[self],
                creation_op="relu",
                device=self.device,
            )
        return Tensor(
            data=data,
            device=self.device,
        )

    def relu_grad(self):
        if self.device == 'cuda':
            data = relu_grad_gpu(self.data)
        else:
            data = relu_grad(self.data)
        return Tensor(
            data=data,
            autograd=False,
            device=self.device,
        )

    def dropout(self, p=0.):
        if p == 0.:
            return self
        dropout_func = dropout_mask_gpu if self.device == 'cuda' else dropout_mask
        mask = Tensor(
            data=dropout_func(self.data, p=p),
            autograd=True,
            device=self.device,
        )
        return self * mask

    def dropout_lstm(self, p=0.):
        if p == 0.:
            return self
        dropout_func = dropout_mask_lstm_gpu if self.device == 'cuda' else dropout_mask_lstm
        mask = Tensor(
            data=dropout_func(self.data, p=p),
            autograd=True,
            device=self.device,
        )
        return self * mask

    def index_select(self, indices, device=None):
        if indices.device == 'cuda':
            indices = indices.cpu().astype(np.int)
        else:
            indices = indices.astype(np.int32)
        if self.device == 'cuda':
            # data = gpuarray.to_gpu(self.data[indices.data])
            data = self._index_select_cuda(indices.data.tolist())
        else:
            data = self.data[indices.data]
        if self.autograd:
            new = Tensor(
                # data=self.data[indices.data],
                data=data,
                autograd=True,
                creators=[self],
                creation_op="index_select",
                device=self.device if device is None else device,
            )
            new.index_select_indices = indices
            return new
        return Tensor(
            data=data,
            autograd=True,
            device=self.device if device is None else device,
        )

    def _index_select_cuda(self, indices):
        data = gpuarray.empty(shape=(len(indices), *self.shape[1:]), dtype=self.dtype)
        for i, index in enumerate(indices):
            gpuarray._memcpy_discontig(dst=data[i], src=self.data[index])
        return data

    def index_selectv2(self, indices, device=None):
        assert self.device == indices.device
        if self.device == 'cuda':
            data = indices_select(self.data, indices.data)
        else:
            data = self.data[indices.data]

        if self.autograd:
            new = Tensor(
                data=data,
                autograd=True,
                creators=[self],
                creation_op="index_select",
                device=self.device if device is None else device,
            )
            new.index_select_indices = indices
            return new
        return Tensor(
            data=data,
            autograd=True,
            device=self.device if device is None else device,
        )

    def softmax(self):
        if self.device == 'cuda':
            # data = softmax_gpu(self.data)
            data = softmax_gpu2d(self.data, dim=1)
        else:
            data = softmax(self.data)
        if self.autograd:
            return Tensor(
                data=data,
                autograd=True,
                creators=[self],
                device=self.device,
            )
        return Tensor(data=data, device=self.device, )

    def cross_entropy(self, target_indices):
        if self.device == 'cuda':
            softmax_output = softmax_gpu2d(self.data, dim=1).get()
        else:
            softmax_output = softmax(self.data)

        t = target_indices.data.flatten()
        p = np.clip(softmax_output.reshape(len(t), -1), a_min=A_MIN, a_max=A_MAX)
        target_dist = np.eye(p.shape[1])[t]
        loss = -(np.log(p + EPSILON) * target_dist).sum(1).mean()

        if self.autograd:
            loss_tensor = Tensor(
                data=loss,
                autograd=True,
                creators=[self],
                creation_op="cross_entropy",
                device=self.device,
            )
            loss_tensor.softmax_output = softmax_output
            loss_tensor.target_dist = target_dist
            return loss_tensor
        return Tensor(data=loss, device=self.device, )

    def bce_loss_with_logits(self, target):
        assert self.data.shape[1] == 2, 'only available for array have (n, 2) dims'
        if self.device == 'cuda':
            # data = self.data.get()
            data = softmax_gpu2d(self.data, dim=1).get()
        else:
            # data = self.data
            data = softmax(self.data)
        t = target.data.flatten()
        target_dist = np.eye(data.shape[1])[t]
        dx = bce_with_logits(data, target_dist)

        if self.autograd:
            loss_tensor = Tensor(
                data=dx.mean(),
                autograd=True,
                creators=[self],
                creation_op="bce",
                device=self.device,
            )
            loss_tensor.dx = dx
            return loss_tensor
        return Tensor(data=dx.mean(), device=self.device, )

    def argmax(self, dim):
        if self.device == 'cuda':
            arg_max = misc.argmax(self.data, axis=dim, keepdims=False)
        else:
            arg_max = np.argmax(self.data, axis=dim)
        return Tensor(
            data=arg_max,
            device=self.device,
            d_type=np.int32,
        )

    def norm(self):
        if self.device == 'cuda':
            return norm_gpu(self.data)
        else:
            return np.linalg.norm(self.data)

    @property
    def shape(self):
        return self.data.shape

    @property
    def dtype(self):
        return self.data.dtype

    def reshape(self, shape):
        self.data = self.data.reshape(shape)
        return self

    def __getitem__(self, i):
        item = Tensor(
            data=self.data[i],
            autograd=self.autograd,
            device=self.device,
            creators=self.creators,
            creation_op=self.creator_op,
        )
        if hasattr(self, 'padding_index') and hasattr(self, 'index_select_indices'):
            item.padding_index = self.padding_index
            item.index_select_indices = self.index_select_indices[i]

        return item

    def __setitem__(self, index, value):
        assert 0 <= index < self.data.shape[0], 'out of index'
        assert tuple(self.shape[1:]) == value.shape
        assert self.device == value.device
        self.data[index] = value.data

    def cpu(self):
        if self.device == 'cuda':
            data = Tensor(
                data=self.data.get(),
                autograd=self.autograd,
                device='cpu',
            )
            data.grad = self.grad.cpu() if self.grad is not None else None
            return data
        else:
            return self

    def cuda(self):
        if self.device == 'cpu':
            data = Tensor(
                data=gpuarray.to_gpu(self.data),
                autograd=self.autograd,
                device='cuda',
            )
            data.grad = self.grad.cuda() if self.grad is not None else None
            return data
        else:
            return self

    def to(self, device):
        assert device in ['cuda', 'cpu']
        if device == 'cuda':
            return self.cuda()
        else:
            return self.cpu()

    def astype(self, d_type):
        self.data = self.data.astype(dtype=d_type)
        return self

    def __repr__(self):
        return str(self.data.__repr__())

    def __str__(self):
        return str(self.data.__str__())

    @staticmethod
    def ones_like(tensor):
        if tensor.device == 'cuda':
            return Tensor(
                data=(gpuarray.ones_like(tensor.data, dtype=dtype)
                      if tensor.shape != () else np.ones_like(tensor.data)),
                device=tensor.device,
            )
        else:
            return Tensor(
                data=np.ones_like(tensor.data, dtype=dtype),
                device=tensor.device,
            )

    @staticmethod
    def zeros_like(tensor, autograd=False):
        if tensor.device == "cuda":
            return Tensor(
                data=gpuarray.zeros_like(tensor.data, dtype=dtype),
                device=tensor.device,
                autograd=autograd,
            )
        else:
            return Tensor(
                data=np.zeros_like(tensor.data, dtype=dtype),
                device=tensor.device,
                autograd=autograd,
            )

    @staticmethod
    def zeros(shape, device=None, d_type=None):
        if device == "cuda":
            return Tensor(
                data=gpuarray.zeros(shape=shape, dtype=d_type if d_type is not None else dtype),
                device=device,
            )
        else:
            return Tensor(
                data=np.zeros(shape=shape, dtype=d_type if d_type is not None else dtype),
                device=device,
            )


class Layer:
    _training = True

    def get_parameters(self):
        params = []
        for key, value in self.__dict__.items():
            if type(value) is Tensor and value.autograd:
                params.append(value)
            elif issubclass(type(value), Layer):
                params.extend(value.get_parameters())
        return params

    def train(self):
        self._training = True
        for key, value in self.__dict__.items():
            if issubclass(type(value), Layer):
                value.train()

    def eval(self):
        self._training = False
        for key, value in self.__dict__.items():
            if issubclass(type(value), Layer):
                value.eval()


class SGD:
    def __init__(self, parameters: list, lr=.01, beta=0.9):
        self.parameters = parameters
        self.lr = lr
        self.beta = beta
        self.prev_grads = [p.grad.data if p.grad is not None else None for p in self.parameters]

    def zero(self):
        for p in self.parameters:
            p.grad.data *= 0

    def step(self, zero=True):
        # for p in self.parameters:
        #     p.data -= p.grad.to(p.device).data * self.lr
        #     if zero:
        #         p.grad.data *= 0

        # use moment
        prev_grads = []
        for p, prev_g in zip(self.parameters, self.prev_grads):
            if prev_g is not None:
                grad = self.beta * prev_g + (1 - self.beta) * p.grad.to(p.device).data
            else:
                grad = p.grad.to(p.device).data

            prev_grads.append(grad)
            p.data -= grad * self.lr
            if zero:
                p.grad.data *= 0
        self.prev_grads = prev_grads


class Adam:
    def __init__(
            self,
            parameters: list,
            lr=.001,
            decay1=0.9,
            decay2=0.999,
            eps=1e-7,
            clip_norm=None,
            lr_scheduler=None,
            **kwargs,
    ):
        super(Adam, self).__init__()
        self.parameters = parameters
        self.lr = lr
        self.decay1 = decay1
        self.decay2 = decay2
        self.eps = eps
        self.clip_norm = clip_norm
        self.lr_scheduler = lr_scheduler
        self.cache = {}

    def zero(self):
        for p in self.parameters:
            p.grad.data *= 0

    def step(self, zero=True):
        for p in self.parameters:
            if p.tensor_id not in self.cache:
                self.cache[p.tensor_id] = {
                    "t": 0,
                    "mean": Tensor.zeros_like(p.grad).data,
                    "var": Tensor.zeros_like(p.grad).data,
                    "grad": Tensor.zeros_like(p.grad).data,
                }

            t = np.inf if self.clip_norm is None else self.clip_norm
            grad_norm = p.grad.norm()
            if grad_norm > t:
                p.grad.data = p.grad.data * (t / grad_norm)

            t = self.cache[p.tensor_id]["t"] + 1
            var = self.cache[p.tensor_id]["var"]
            mean = self.cache[p.tensor_id]["mean"]
            grad = self.cache[p.tensor_id]["grad"]

            self.cache[p.tensor_id]["t"] = t
            if p.device == 'cuda':
                self.cache[p.tensor_id]["var"] = adam_var(var, p.grad.data, d2=self.decay2, out=var)
                self.cache[p.tensor_id]["mean"] = adam_mean(mean, p.grad.data, d1=self.decay1, out=mean)

                grad = adam_grad(
                    mean=self.cache[p.tensor_id]["mean"],
                    var=self.cache[p.tensor_id]["var"],
                    d1=self.decay1,
                    d2=self.decay2,
                    eps=self.eps,
                    t=t,
                    out=grad,
                )
                sub_(x=p.data, y=grad, c=self.lr)
                if zero:
                    zero_(p.grad.data)
            else:
                self.cache[p.tensor_id]["var"] = self.decay2 * var + (1 - self.decay2) * p.grad.data ** 2
                self.cache[p.tensor_id]["mean"] = self.decay1 * mean + (1 - self.decay1) * p.grad.data

                v_hat = self.cache[p.tensor_id]["var"] / (1 - self.decay2 ** t)
                m_hat = self.cache[p.tensor_id]["mean"] / (1 - self.decay1 ** t)
                grad = m_hat / (v_hat ** 0.5 + self.eps)

                p.data -= self.lr * grad
                if zero:
                    p.grad.data *= 0


class Linear(Layer):
    def __init__(
            self,
            n_inputs,
            n_outputs,
            bias=True,
            device='cuda',
    ):
        super(Linear, self).__init__()
        self.use_bias = bias
        self.device = device

        w = xavier_init(n_input=n_inputs, n_output=n_outputs)
        # w = truncated_normal(0., 1., (n_inputs, n_outputs))
        self.weight = Tensor(w, device=device, autograd=True)

        if self.use_bias:
            self.bias = Tensor(np.zeros(n_outputs, dtype=np.float32), device=device, autograd=True)

    def forward(self, input: Tensor):
        if self.use_bias:
            return input.mm(self.weight) + self.bias.expand(0, input.shape[0])
        return input.mm(self.weight)

    def __call__(self, input: Tensor):
        return self.forward(input)


class Sequential(Layer):
    def __init__(self, layers=None):
        super(Sequential, self).__init__()
        self.layers = layers if layers is not None else []

    def add(self, layer: Layer):
        self.layers.append(layer)

    def forward(self, input):
        for layer in self.layers:
            input = layer.forward(input)
        return input

    def __call__(self, input):
        return self.forward(input)

    def get_parameters(self):
        params = []
        for layer in self.layers:
            params.extend(layer.get_parameters())
        return params


class Embedding(Layer):
    def __init__(
            self,
            vectors: np.ndarray,
            padding_index,
            device='cuda',
            autograd=False,
    ):
        super(Embedding, self).__init__()
        assert len(vectors.shape) == 2, 'vectors must have 2 dim'
        self.vocab_size, self.embedding_dim = vectors.shape
        self.autograd = autograd
        self.device = device
        self.padding_index = padding_index

        self.weight = Tensor(
            data=vectors,
            autograd=autograd,
            device=self.device,
        )

    @classmethod
    def from_pretrained(cls, vectors: np.ndarray, padding_index, device='cuda', autograd=False):
        embedding = cls(vectors=vectors, padding_index=padding_index, device=device, autograd=autograd)
        return embedding

    @classmethod
    def init(cls, vocab_size, embedding_dim, padding_index=0, device='cuda', autograd=False):
        vectors = (np.random.rand(vocab_size, embedding_dim) - 0.5) / embedding_dim
        return cls(vectors=vectors, padding_index=padding_index, device=device, autograd=autograd)

    def forward(self, input):
        assert len(input.shape) in [1, 2]
        # if len(input.shape) == 1:
        #     data = self.weight.index_select(input, device=self.device)
        #     data.padding_index = self.padding_index
        # else:
        #     data = [
        #         self.weight.index_select(input[i], device=self.device)
        #         for i in range(input.shape[0])
        #     ]
        #     for item in data:
        #         item.padding_index = self.padding_index
        data = self.weight.index_selectv2(input)
        data.padding_index = self.padding_index
        return data

    def __call__(self, input):
        return self.forward(input)


class Tanh(Layer):
    def __init__(self):
        super(Tanh, self).__init__()

    @staticmethod
    def forward(input: Tensor):
        return input.tanh()

    def __call__(self, input: Tensor):
        return self.forward(input)


class Sigmoid(Layer):
    def __init__(self):
        super(Sigmoid, self).__init__()

    @staticmethod
    def forward(input: Tensor):
        return input.sigmoid()

    def __call__(self, input: Tensor):
        return self.forward(input)


class ReLu(Layer):
    def __init__(self):
        super(ReLu, self).__init__()

    @staticmethod
    def forward(input: Tensor):
        return input.relu()

    def __call__(self, input: Tensor):
        return self.forward(input)


class Dropout(Layer):
    def __init__(self, p=0.):
        super(Dropout, self).__init__()
        self.p = p

    def forward(self, input: Tensor):
        if not self._training or self.p == 0.:
            return input
        return input.dropout(self.p)

    def __call__(self, input: Tensor):
        return self.forward(input)


class LSTMDropout(Layer):
    def __init__(self, p=0.):
        super(LSTMDropout, self).__init__()
        self.p = p

    def forward(self, inputs: list):
        if not self._training or self.p == 0.:
            return inputs
        n_dropout = int(len(inputs) * self.p)
        x = [0] * n_dropout + [1] * (len(inputs) - n_dropout)
        np.random.shuffle(x)
        inputs = [inp if drop == 1 else Tensor.zeros_like(inp, autograd=True) for inp, drop in zip(inputs, x)]
        return inputs

    def __call__(self, inputs):
        return self.forward(inputs)


class CrossEntropyLoss:
    @staticmethod
    def forward(input: Tensor, target: Tensor):
        return input.cross_entropy(target)

    def __call__(self, input: Tensor, target: Tensor):
        return self.forward(input, target)


class BCEWithLogitsLoss:
    @staticmethod
    def forward(input: Tensor, target: Tensor):
        return input.bce_loss_with_logits(target)

    def __call__(self, input: Tensor, target: Tensor):
        return self.forward(input, target)


class LSTMCell(Layer):
    def __init__(
            self,
            input_dim,
            hidden_dim,
            device='cuda',
    ):
        super(LSTMCell, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.device = device

        self.xf = Linear(n_inputs=input_dim, n_outputs=hidden_dim, device=device)
        self.xi = Linear(n_inputs=input_dim, n_outputs=hidden_dim, device=device)
        self.xo = Linear(n_inputs=input_dim, n_outputs=hidden_dim, device=device)
        self.xc = Linear(n_inputs=input_dim, n_outputs=hidden_dim, device=device)

        self.hf = Linear(n_inputs=hidden_dim, n_outputs=hidden_dim, bias=False, device=device)
        self.hi = Linear(n_inputs=hidden_dim, n_outputs=hidden_dim, bias=False, device=device)
        self.ho = Linear(n_inputs=hidden_dim, n_outputs=hidden_dim, bias=False, device=device)
        self.hc = Linear(n_inputs=hidden_dim, n_outputs=hidden_dim, bias=False, device=device)

    def forward(self, input, hidden=None):
        if hidden is not None:
            prev_hidden, prev_cell = hidden
        else:
            batch_size = input.shape[0]
            prev_hidden, prev_cell = self.init_hidden(batch_size=batch_size)
        f = (self.xf(input) + self.hf(prev_hidden)).sigmoid()
        i = (self.xi(input) + self.hi(prev_hidden)).sigmoid()
        o = (self.xo(input) + self.ho(prev_hidden)).sigmoid()
        u = (self.xc(input) + self.hc(prev_hidden)).tanh()
        c = (f * prev_cell) + (i * u)
        h = o * c.tanh()
        return h, (h, c)

    def __call__(self, input, hidden=None):
        return self.forward(input, hidden)

    def init_hidden(self, batch_size=1):
        h_data = np.zeros((batch_size, self.hidden_dim))
        c_data = np.zeros((batch_size, self.hidden_dim))
        h_data[:, 0] = 1
        c_data[:, 0] = 1
        h = Tensor(h_data, autograd=True, device=self.device)
        c = Tensor(c_data, autograd=True, device=self.device)
        return h, c


class LSTMLayer(Layer):
    def __init__(
            self,
            input_dim,
            hidden_dim,
            device='cuda',
    ):
        super(LSTMLayer, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.device = device
        self.lstm_cell = LSTMCell(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            device=device,
        )

    def forward(self, inputs):
        hidden = None
        outputs = []
        for i in range(len(inputs)):
            output, hidden = self.lstm_cell(inputs[i], hidden)
            outputs.append(output)
        return outputs, hidden

    def __call__(self, inputs):
        return self.forward(inputs)

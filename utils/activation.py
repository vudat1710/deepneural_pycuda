import numpy as np
import pycuda.autoinit
import pycuda.cumath
import pycuda.gpuarray
from pycuda.reduction import ReductionKernel
from pycuda.elementwise import ElementwiseKernel
from deepneural_pycuda.tensor import Tensor
from numpy import ndarray

### GPU Support

def tanh_gpu(x: Tensor) -> Tensor:
    tanh_ = ElementwiseKernel(
        "double *Y, double *X",
        "Y[i] = (exp (X[i]) - exp (-X[i])) / (exp (X[i]) + exp (-X[i]))",
        "tanh_")
    y = pycuda.gpuarray.empty_like(x)
    tanh_(y, x)
    return y

def sigmoid_gpu(x: Tensor) -> Tensor:
    sigmoid = ElementwiseKernel(
        "double *Y, double *X",
        "Y[i] = 1.0 / (1.0 + exp (-X[i]) )",
        "sigmoid")
    y = pycuda.gpuarray.empty_like(x)
    sigmoid(y, x)
    return y


def softmax_gpu(x: Tensor) -> Tensor:
    exp_sum = ReductionKernel(np.float64, neutral="0.0",
            reduce_expr="a+b", map_expr="exp (x[i])",
            arguments="double *x")
    softmax = ElementwiseKernel(
        "double *Y, double *X, double s",
        "Y[i] = exp (X[i]) / s",
        "softmax")
    y = pycuda.gpuarray.empty_like(x)
    s = exp_sum(x).get()
    softmax(y, x, s)
    return y

### CPU Support

def sigmoid(X: ndarray) -> ndarray:
    func = np.vectorize(lambda x: 1 / (1 + np.exp(-x)))
    return func(X)


def tanh(X: ndarray) -> ndarray:
    func = np.vectorize(lambda x: np.tanh(x))
    return func(X)


def softmax(X: ndarray) -> ndarray:
    smax = np.empty_like(X)
    for i in range(X.shape[1]):
        exps = np.exp(X[:, i] - np.max(X[:, i]))
        smax[:,i] = exps / np.sum(exps)
    return smax
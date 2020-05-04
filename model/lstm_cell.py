from ..tensor import Tensor
from ..utils.activation import *
from ..utils.utils import *
from typing import Dict, Tuple
from pycuda import gpuarray

def lstm_cell_forward_gpu(xt: Tensor, a_prev: Tensor, c_prev: Tensor, parameters: Dict) -> Tuple[Tensor, Tensor, Tensor, Tuple]:
    # Retrieve parameters from "parameters"
    Wf = parameters["Wf"]
    bf = parameters["bf"]
    Wi = parameters["Wi"]
    bi = parameters["bi"]
    Wc = parameters["Wc"]
    bc = parameters["bc"]
    Wo = parameters["Wo"]
    bo = parameters["bo"]

    # Retrieve dimensions from shapes of xt and Wy
    n_x, m = xt.shape  # Shape of input at timestep t
    n_a, n_t = Wf.shape  # Shape of hidden state at timestep t

    # Concatenate a_prev and xt
    concat = gpuarray.empty(((n_a + n_x), m), dtype=np.float64)
    concat[: n_a, :] = a_prev
    concat[n_a:, :] = xt


    # Compute values for ft, it, cct, c_next, ot, a_next
    a1 = binaryop_matvec('+', dot(Wf, concat), bf)
    a2 = bia1 = binaryop_matvec('+', dot(Wi, concat), bi)
    a3 = bia1 = binaryop_matvec('+', dot(Wc, concat), bc)
    a4 = bia1 = binaryop_matvec('+', dot(Wo, concat), bo)

    ft = sigmoid_gpu(a1)
    it = sigmoid_gpu(a2)
    cct = tanh_gpu(a3)
    ot = sigmoid_gpu(a4)


    c_next = ft * c_prev + it * cct
    a_next = ot * tanh_gpu(c_next)

    # Compute prediction of the LSTM cell
    yt_pred = a_next

    # store values needed for backward propagation in cache
    cache = (a_next, c_next, a_prev, c_prev, ft, it, cct, ot, xt, parameters)

    return a_next, c_next, yt_pred, cache
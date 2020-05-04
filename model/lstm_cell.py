from tensor import Tensor
from utils.activation import *
from utils.utils import *
from typing import Dict, Tuple
from pycuda import gpuarray


def lstm_cell_forward(
        xt: Tensor,
        a_prev: Tensor,
        c_prev: Tensor,
        parameters: Dict
) -> Tuple[Tensor, Tensor, Tensor, Tuple]:
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


def lstm_cell_backward(da_next: Tensor, dc_next: Tensor, cache: Tuple) -> Dict[str, Tensor]:
    # Retrieve information from "cache"
    (a_next, c_next, a_prev, c_prev, ft, it, cct, ot, xt, parameters) = cache

    # Retrieve dimensions from xt's and a_next's shape
    n_x, m = xt.shape
    n_a, m = a_next.shape

    # Compute gates related derivatives
    dot = da_next * tanh_gpu(c_next) * ot * (1 - ot)
    dcct = (dc_next * it + ot * (1 - square_gpu(tanh_gpu(c_next))) * it * da_next) * (1 - square_gpu(cct))
    dit = (dc_next * cct + ot * (1 - square_gpu(tanh_gpu(c_next))) * cct * da_next) * it * (1 - it)
    dft = (dc_next * c_prev + ot * (1 - square_gpu(tanh_gpu(c_next))) * c_prev * da_next) * ft * (1 - ft)

    concat = pycuda.gpuarray.empty(((n_a + n_x), m), dtype=np.float64)
    concat[:n_a, :] = a_prev
    concat[n_a:, :] = xt

    # Compute parameters related derivatives. Use equations (11)-(14) (≈8 lines)
    dWf = dot(dft, transpose(concat))
    dWi = dot(dit, transpose(concat))
    dWc = dot(dcct, transpose(concat))
    dWo = dot(dot, transpose(concat))
    dbf = sum_gpu(dft, axis=1, keepdims=True)
    dbi = sum_gpu(dit, axis=1, keepdims=True)
    dbc = sum_gpu(dcct, axis=1, keepdims=True)
    dbo = sum_gpu(dot, axis=1, keepdims=True)

    # Compute derivatives w.r.t previous hidden state, previous memory state and input.
    da_prev = dot(transpose(parameters['Wf'][:, :n_a]), dft) + dot(transpose(parameters['Wi'][:, :n_a]), dit) + dot(
        transpose(parameters['Wc'][:, :n_a]), dcct) + dot(transpose(parameters['Wo'][:, :n_a]), dot)
    dc_prev = dc_next * ft + ot * from_one_gpu(square_gpu(tanh_gpu(c_next))) * ft * da_next
    dxt = dot(transpose(parameters['Wf'][:, n_a:]), dft) + dot(transpose(parameters['Wi'][:, n_a:]), dit) + dot(
        transpose(parameters['Wc'][:, n_a:]), dcct) + dot(transpose(parameters['Wo'][:, n_a:]), dot)

    # Save gradients in dictionary
    gradients = {"dxt": dxt, "da_prev": da_prev, "dc_prev": dc_prev, "dWf": dWf, "dbf": dbf, "dWi": dWi, "dbi": dbi,
                 "dWc": dWc, "dbc": dbc, "dWo": dWo, "dbo": dbo}

    return gradients

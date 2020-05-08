from .lstm_cell import *
from pycuda import gpuarray
from ..tensor import Tensor
from typing import Dict, Tuple
import numpy as np
import pycuda.driver as drv


def lstm_forward(x: Tensor, a0: Tensor, parameters: Dict[str, Tensor]) -> Tuple[Tensor, Tensor, Tensor, Tuple]:
    # Initialize "caches", which will track the list of all the caches
    caches = []

    # Retrieve dimensions from shapes of x and parameters['Wy'] (≈2 lines)
    n_x, m, T_x = x.shape
    n_a, n_t = parameters['Wf'].shape

    # initialize "a", "c" and "y" with zeros (≈3 lines)
    a = gpuarray.zeros((n_a, m, T_x), dtype=np.float64)
    c = gpuarray.zeros((n_a, m, T_x), dtype=np.float64)
    y = gpuarray.zeros((n_a, m, T_x), dtype=np.float64)

    # Initialize a_next and c_next (≈2 lines)
    a_next = gpuarray.to_gpu(a0)
    c_next = gpuarray.zeros((n_a, m), dtype=np.float64)

    # transfer X to gpu
    x_gpu = gpuarray.to_gpu(x)

    # loop over all time-steps
    for t in range(T_x):
        # Update next hidden state, next memory state, compute the prediction, get the cache (≈1 line)
        a_next, c_next, yt, cache = lstm_cell_forward(x_gpu[:,:,t], a_next, c_next, parameters)
        # Save the value of the new "next" hidden state in a (≈1 line)
        a[:, :, t] = a_next
        # Save the value of the prediction in y (≈1 line)
        y[:, :, t] = yt
        # Save the value of the next cell state (≈1 line)
        c[:, :, t] = c_next
        # Append the cache into caches (≈1 line)
        caches.append(cache)

    # store values needed for backward propagation in cache
    caches = (caches, x_gpu)

    return a, y, c, caches


def lstm_backward_gpu(da: Tensor, caches: Tuple) -> Dict[str, Tensor]:
    # Retrieve values from the first cache (t=1) of caches.
    (caches, x) = caches
    (a1, c1, a0, c0, f1, i1, cc1, o1, x1, parameters) = caches[0]

    # Retrieve dimensions from da's and x1's shapes (≈2 lines)
    n_a, m, T_x = da.shape
    n_x, m = x1.shape

    # initialize the gradients with the right sizes (≈12 lines)
    dx = gpuarray.zeros((n_x, m, T_x), dtype=np.float64)
    da0 = gpuarray.zeros((n_a, m), dtype=np.float64)
    da_prevt = gpuarray.zeros(da0.shape, dtype=np.float64)
    dc_prevt = gpuarray.zeros(da0.shape, dtype=np.float64)
    dWf = gpuarray.zeros((n_a, n_a + n_x), dtype=np.float64)
    dWi = gpuarray.zeros(dWf.shape, dtype=np.float64)
    dWc = gpuarray.zeros(dWf.shape, dtype=np.float64)
    dWo = gpuarray.zeros(dWf.shape, dtype=np.float64)
    dbf = gpuarray.zeros((n_a, 1), dtype=np.float64)
    dbi = gpuarray.zeros(dbf.shape, dtype=np.float64)
    dbc = gpuarray.zeros(dbf.shape, dtype=np.float64)
    dbo = gpuarray.zeros(dbf.shape, dtype=np.float64)

    gradients = {}
    # loop back over the whole sequence
    for t in reversed(range(T_x)):
        da_gpu = gpuarray.to_gpu(da[:, :, t])
        # Compute all gradients using lstm_cell_backward
        gradients = lstm_cell_backward(da_gpu, dc_prevt, caches[t])
        # Store or add the gradient to the parameters' previous step's gradient
        dx[:, :, t] = gradients["dxt"]
        dWf += gradients["dWf"]
        dWi += gradients["dWi"]
        dWc += gradients["dWc"]
        dWo += gradients["dWo"]
        dbf += gradients["dbf"]
        dbi += gradients["dbi"]
        dbc += gradients["dbc"]
        dbo += gradients["dbo"]
    # Set the first activation's gradient to the backpropagated gradient da_prev.
    da0 = gradients["da_prev"]

    # Store the gradients in a python dictionary
    gradients = {"dx": dx, "da0": da0, "dWf": dWf, "dbf": dbf, "dWi": dWi, "dbi": dbi,
                 "dWc": dWc, "dbc": dbc, "dWo": dWo, "dbo": dbo}

    return gradients


def update_weights(parameters: Dict[str, Tensor], gradients: Dict[str, Tensor], learning_rate: np.float32) -> None:
    parameters['Wf'] = parameters['Wf'] - learning_rate * gradients['dWf']
    parameters['bf'] = parameters['bf'] - learning_rate * gradients['dbf']
    parameters['Wi'] = parameters['Wi'] - learning_rate * gradients['dWi']
    parameters['bi'] = parameters['bi'] - learning_rate * gradients['dbi']
    parameters['Wc'] = parameters['Wc'] - learning_rate * gradients['dWc']
    parameters['bc'] = parameters['bc'] - learning_rate * gradients['dbc']
    parameters['Wo'] = parameters['Wo'] - learning_rate * gradients['dWo']
    parameters['bo'] = parameters['bo'] - learning_rate * gradients['dbo']


def layer_to_gpu(parameters: dict) -> Tensor:
    """
    copy cell weights to GPU
    :param parameters: dictionary of cell weights
    :return: gpu_params: parameters transferred as PyCuda GPUArray
    """
    gpu_params = {}
    for parameter in parameters.items():
        gpu_params[parameter[0]] = gpuarray.to_gpu(parameter[1])

    del parameters
    return gpu_params


def layer_to_gpu_async(parameters: dict, stream: drv.Stream) -> Tensor:
    """
    copy cell weights to GPU
    :param parameters: dictionary of cell weights
    :return: gpu_params: parameters transferred as PyCuda GPUArray
    """
    gpu_params = {}
    for parameter in parameters.items():
        gpu_params[parameter[0]] = gpuarray.to_gpu_async(parameter[1], stream=stream)

    return gpu_params


def layer_from_gpu(gpu_parameters: dict) -> np.ndarray:
    """
    copy cell weights from GPU
    :param gpu_parameters: dictionary of cell weights with weights being PyCuda GPUArrays
    :return: parameters: parameters transferred from PyCuda GPUArray to numpy arrays
    """
    parameters = {}

    for gpu_param in gpu_parameters.items():
        parameter = gpu_param[1].get()
        parameters[gpu_param[0]] = parameter

    del gpu_parameters
    return parameters


def layer_from_gpu_async(gpu_parameters: dict, stream: drv.Stream) -> np.ndarray:
    """
    copy cell weights from GPU
    :param gpu_parameters: dictionary of cell weights with weights being PyCuda GPUArrays
    :return: parameters: parameters transferred from PyCuda GPUArray to numpy arrays
    """
    parameters = {}
    for gpu_param in gpu_parameters.items():
        parameter = gpu_param[1].get_async(stream=stream)
        parameters[gpu_param[0]] = parameter

    return parameters

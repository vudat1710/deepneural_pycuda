import pycuda.autoinit
from pycuda import gpuarray
import pycuda.driver as drv
from skcuda import cublas
import numbers
import numpy as np
from pycuda.reduction import ReductionKernel
from pycuda.elementwise import ElementwiseKernel
from pycuda.compiler import SourceModule
from skcuda import misc, linalg

_global_cublas_allocator = drv.mem_alloc
_global_cublas_handle = cublas.cublasCreate()

sigmoid_float_ker = ElementwiseKernel(
    f"float *Y, float *x",
    "Y[i] = 1.0 / (1.0 + exp (-x[i]) )",
    "sigmoid_float")

sigmoid_double_ker = ElementwiseKernel(
    f"double *Y, double *x",
    "Y[i] = 1.0 / (1.0 + exp (-x[i]) )",
    "sigmoid_double")

tanh_float_ker = ElementwiseKernel(
    f"float *Y, float *x",
    "Y[i] = (exp (x[i]) - exp (-x[i])) / (exp (x[i]) + exp (-x[i]))",
    "tanh_float")

tanh_double_ker = ElementwiseKernel(
    f"double *Y, double *x",
    "Y[i] = (exp (x[i]) - exp (-x[i])) / (exp (x[i]) + exp (-x[i]))",
    "tanh_double"
)

exp_sum_float_ker = ReductionKernel(np.float32, neutral="0.0",
                                    reduce_expr="a+b", map_expr="exp (x[i])",
                                    arguments=f"float *x"
                                    )

softmax_float_ker = ElementwiseKernel(
    f"float *Y, float *x, float s",
    "Y[i] = exp (x[i]) / s",
    "softmax_float"
)

exp_sum_double_ker = ReductionKernel(np.float32, neutral="0.0",
                                     reduce_expr="a+b", map_expr="exp (x[i])",
                                     arguments=f"double *x"
                                     )

softmax_double_ker = ElementwiseKernel(
    f"double *Y, double *x, double s",
    "Y[i] = exp (x[i]) / s",
    "softmax_double"
)

exp_float_ker = ElementwiseKernel(
    arguments=f"float *inp, float *out",
    operation=f"out[i] = exp(inp[i])",
    name='exp_float_ker',
)

exp_double_ker = ElementwiseKernel(
    arguments=f"double *inp, double *out",
    operation=f"out[i] = exp(inp[i])",
    name='exp_double_ker',
)

square_float_ker = ElementwiseKernel(
    f"float *Y, float *x",
    "Y[i] = x[i] * x[i]",
    "square_float_ker")

square_double_ker = ElementwiseKernel(
    f"double *Y, double *x",
    "Y[i] = x[i] * x[i]",
    "square_double_ker")

sigmoid_grad_float_ker = ElementwiseKernel(
    arguments=f"float *inp, float *out",
    operation="out[i] = inp[i] * (1.0 - inp[i])",
    name="sigmoid_grad_float_ker",
)

sigmoid_grad_double_ker = ElementwiseKernel(
    arguments=f"double *inp, double *out",
    operation="out[i] = inp[i] * (1.0 - inp[i])",
    name="sigmoid_grad_double_ker",
)

tanh_grad_float_ker = ElementwiseKernel(
    arguments="float *inp, float *out",
    operation="out[i] = 1.0 - inp[i] * inp[i]",
    name="tanh_grad_float_ker",
)

tanh_grad_double_ker = ElementwiseKernel(
    arguments="double *inp, double *out",
    operation="out[i] = 1.0 - inp[i] * inp[i]",
    name="tanh_grad_double_ker",
)

relu_float_ker = ElementwiseKernel(
    arguments="float *inp, float *out",
    operation="out[i] = (inp[i] > 0) ? inp[i] : 0",
    name="relu_float_ker",
)

relu_double_ker = ElementwiseKernel(
    arguments="double *inp, double *out",
    operation="out[i] = (inp[i] > 0) ? inp[i] : 0",
    name="relu_double_ker",
)

relu_grad_float_ker = ElementwiseKernel(
    arguments="float *inp, float *out",
    operation="out[i] = (inp[i] > 0) ? 1 : 0",
    name="relu_grad_float_ker",
)

relu_grad_double_ker = ElementwiseKernel(
    arguments="double *inp, double *out",
    operation="out[i] = (inp[i] > 0) ? 1 : 0",
    name="relu_grad_double_ker",
)

# random kernel
random_1d_ker_template = """
#include <curand_kernel.h>
#define ULL unsigned long long
extern "C" {
    __global__ void random_1d_%(p)s_array(%(p)s *arr, float p) {
    curandState cr_state; 
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    curand_init( (ULL) clock() + (ULL) tid, (ULL) 0, (ULL) 0, &cr_state);

    float x = curand_uniform(&cr_state);  
    if (x < p) arr[tid] = 0;
    else arr[tid] = 1;
    return;
    }
}
"""

random_1d_int_func = SourceModule(
    no_extern_c=True,
    source=random_1d_ker_template % {'p': 'int'},
).get_function('random_1d_int_array')

random_1d_float_func = SourceModule(
    no_extern_c=True,
    source=random_1d_ker_template % {'p': 'float'},
).get_function('random_1d_float_array')

random_1d_double_func = SourceModule(
    no_extern_c=True,
    source=random_1d_ker_template % {'p': 'double'},
).get_function('random_1d_double_array')

# 2d

random_2d_ker_template = """
#include <curand_kernel.h>
#define ULL unsigned long long
extern "C" {
    __global__ void random_2d_%(p)s_array(%(p)s *arr, int dim, float p) {
    curandState cr_state; 
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;
    int tid = row * dim + col;
    curand_init( (ULL) clock() + (ULL) tid, (ULL) 0, (ULL) 0, &cr_state);

    float x = curand_uniform(&cr_state);  
    if (x < p) arr[tid] = 0;
    else arr[tid] = 1;
    return;
    }
}
"""

random_2d_float_func = SourceModule(
    no_extern_c=True,
    source=random_2d_ker_template % {'p': 'float'}
).get_function('random_2d_float_array')

random_2d_int_func = SourceModule(
    no_extern_c=True,
    source=random_2d_ker_template % {'p': 'int'}
).get_function('random_2d_int_array')

random_2d_double_func = SourceModule(
    no_extern_c=True,
    source=random_2d_ker_template % {'p': 'double'}
).get_function('random_2d_double_array')

random_lstm_ker_template = """
#include <curand_kernel.h>
#define ULL unsigned long long
extern "C" {
    __global__ void random_lstm_%(p)s_array(%(p)s *array, int dim, float p) {
    curandState cr_state; 
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    curand_init( (ULL) clock() + (ULL) row, (ULL) 0, (ULL) 0, &cr_state);
    float x = curand_uniform(&cr_state);  
    %(p)s v;
    if (x < p) v = 0;
    else v = 1;
    for (int i = row * dim; i < (row + 1) * dim; i++) {
        array[i] = v;
    }
    return;
    }
}
"""

random_lstm_int_func = SourceModule(
    no_extern_c=True,
    source=random_lstm_ker_template % {'p': 'int'}
).get_function('random_lstm_int_array')

random_lstm_float_func = SourceModule(
    no_extern_c=True,
    source=random_lstm_ker_template % {'p': 'float'}
).get_function('random_lstm_float_array')

random_lstm_double_func = SourceModule(
    no_extern_c=True,
    source=random_lstm_ker_template % {'p': 'double'}
).get_function('random_lstm_double_array')


def ones(shape: tuple, dtype: np.dtype, order: str = 'C', allocator=drv.mem_alloc):
    """
    Return an array of the given shape and dtype filled with ones.

    Parameters
    ----------
    shape : tuple
        Array shape.
    dtype : data-type
        Data type for the array.
    order : {'C', 'F'}, optional
        Create array using row-major or column-major format.
    allocator : callable, optional
        Returns an object that represents the memory allocated for
        the requested array.

    Returns
    -------
    out : pycuda.gpuarray.GPUArray
        Array of ones with the given shape, dtype, and order.
    """

    out = gpuarray.GPUArray(shape, dtype, allocator, order=order)
    o = np.ones((), dtype)
    out.fill(o)
    return out


def sum_gpu(
        x_gpu: gpuarray.GPUArray,
        axis: int = None,
        out: gpuarray.GPUArray = None,
        keepdims: bool = False,
        calc_mean: bool = False,
        ddof: int = 0
) -> gpuarray.GPUArray:
    """
    Compute the sum along the specified axis.

    Parameters
    ----------
    ddof
    calc_mean
    x_gpu : pycuda.gpuarray.GPUArray
        Array containing numbers whose sum is desired.
    axis : int (optional)
        Axis along which the sums are computed. The default is to
        compute the sum of the flattened array.
    out : pycuda.gpuarray.GPUArray (optional)
        Output array in which to place the result.
    keepdims : bool (optional, default False)
        If True, the axes which are reduced are left in the result as
        dimensions with size one.

    Returns
    -------
    out : pycuda.gpuarray.GPUArray
        sum of elements, or sums of elements along the desired axis.
    """
    assert isinstance(ddof, numbers.Integral)

    if axis is None or len(x_gpu.shape) <= 1:
        out_shape = (1,) * len(x_gpu.shape) if keepdims else ()
        if calc_mean is False:
            return gpuarray.sum(x_gpu).reshape(out_shape)
        else:
            return gpuarray.sum(x_gpu).reshape(out_shape) / (x_gpu.dtype.type(x_gpu.size - ddof))

    if axis < 0:
        axis += 2
    if axis > 1:
        raise ValueError('invalid axis')

    if x_gpu.flags.c_contiguous:
        n, m = x_gpu.shape[1], x_gpu.shape[0]
        lda = x_gpu.shape[1]
        trans = "n" if axis == 0 else "t"
        sum_axis, out_axis = (m, n) if axis == 0 else (n, m)
    else:
        n, m = x_gpu.shape[0], x_gpu.shape[1]
        lda = x_gpu.shape[0]
        trans = "t" if axis == 0 else "n"
        sum_axis, out_axis = (n, m) if axis == 0 else (m, n)

    if calc_mean:
        alpha = (1.0 / (sum_axis - ddof))
    else:
        alpha = 1.0
    if x_gpu.dtype == np.complex64:
        gemv = cublas.cublasCgemv
    elif x_gpu.dtype == np.float32:
        gemv = cublas.cublasSgemv
    elif x_gpu.dtype == np.complex128:
        gemv = cublas.cublasZgemv
    elif x_gpu.dtype == np.float64:
        gemv = cublas.cublasDgemv
    else:
        raise Exception('invalid dtype')

    alloc = _global_cublas_allocator
    ons = ones((sum_axis,), x_gpu.dtype, allocator=alloc)

    if keepdims:
        out_shape = (1, out_axis) if axis == 0 else (out_axis, 1)
    else:
        out_shape = (out_axis,)

    if out is None:
        out = gpuarray.empty(out_shape, x_gpu.dtype, alloc)
    else:
        assert out.dtype == x_gpu.dtype
        assert out.size >= out_axis

    gemv(_global_cublas_handle, trans, n, m,
         alpha, x_gpu.gpudata, lda,
         ons.gpudata, 1, 0.0, out.gpudata, 1)
    return out


def tanh_gpu(x: gpuarray.GPUArray) -> gpuarray.GPUArray:
    tanh = tanh_float_ker if x.dtype == np.float32 else tanh_double_ker
    y = pycuda.gpuarray.empty_like(x)
    tanh(y, x)
    return y


def tanh_grad_gpu(x: gpuarray.GPUArray) -> gpuarray.GPUArray:
    tanh_grad = tanh_grad_float_ker if x.dtype == np.float32 else tanh_grad_double_ker
    y = gpuarray.empty_like(x)
    tanh_grad(x, y)
    return y


def sigmoid_gpu(x: gpuarray.GPUArray, out: gpuarray.GPUArray = None) -> gpuarray.GPUArray:
    sigmoid = sigmoid_float_ker if x.dtype == np.float32 else sigmoid_double_ker
    if out is None:
        y = pycuda.gpuarray.empty_like(x)
    else:
        y = out
    sigmoid(y, x)
    return y


def sigmoid_grad_gpu(x: gpuarray.GPUArray, out: gpuarray.GPUArray = None) -> gpuarray.GPUArray:
    sigmoid_grad = sigmoid_grad_float_ker if x.dtype == np.float32 else sigmoid_grad_double_ker
    if out is None:
        y = gpuarray.empty_like(x)
    else:
        y = out
    sigmoid_grad(x, y)
    return y


def relu_gpu(x: gpuarray.GPUArray):
    relu = relu_float_ker if x.dtype == np.float32 else relu_double_ker
    y = gpuarray.empty_like(x)
    relu(x, y)
    return y


def relu_grad_gpu(x: gpuarray.GPUArray):
    relu_grad = relu_grad_float_ker if x.dtype == np.float32 else relu_grad_double_ker
    y = gpuarray.empty_like(x)
    relu_grad(x, y)
    return y


def softmax_gpu(x: gpuarray.GPUArray, out: gpuarray.GPUArray = None) -> gpuarray.GPUArray:
    if x.dtype == np.float32:
        exp_sum = exp_sum_float_ker
        softmax = softmax_float_ker
    else:
        exp_sum = exp_sum_double_ker
        softmax = softmax_double_ker
    if y is None:
        y = pycuda.gpuarray.empty_like(x)
    else:
        y = out
    s = exp_sum(x).get()
    softmax(y, x, s)
    return y


def expand(x: gpuarray.GPUArray, dim: int, copies):
    trans_dims = list(range(0, len(x.shape)))
    trans_dims.insert(dim, len(x.shape))
    order = 'F' if dim == 0 else 'C'
    data = (linalg.dot(x.reshape(-1, 1), misc.ones((1, copies), dtype=x.dtype))
            .reshape((*x.shape, copies), order=order)).transpose(trans_dims)
    return data


def softmax_gpu2d(x: gpuarray.GPUArray, dim):
    assert len(x.shape) == 2, 'expected 2-dimension array'
    assert 0 <= dim <= 1, "expected 0 <= dim <=1"
    exp_ker = exp_float_ker if x.dtype == np.float32 else exp_double_ker
    x_exp = gpuarray.empty_like(x)
    exp_ker(x, x_exp)
    x_exp_sum = misc.sum(x_gpu=x_exp, axis=dim)
    x_exp = misc.div_matvec(x_gpu=x_exp, a_gpu=x_exp_sum, axis=1 - dim)
    return x_exp


def square_gpu(x: gpuarray.GPUArray) -> gpuarray.GPUArray:
    square = square_float_ker if x.dtype == np.float32 else square_double_ker
    y = pycuda.gpuarray.empty_like(x)
    square(y, x)
    return y


random_ker = {
    1: {
        np.int32: random_1d_int_func,
        np.float32: random_1d_float_func,
        np.float64: random_1d_double_func,
    },
    2: {
        np.int32: random_2d_int_func,
        np.float32: random_2d_float_func,
        np.float64: random_2d_double_func,
    }
}


def dropout_mask_gpu(x: gpuarray.GPUArray, p=0.):
    assert x.dtype in [np.int32, np.float32, np.float64], 'invalid dtype'
    assert len(x.shape) in [1, 2], 'invalid number of dims'
    mask = gpuarray.empty_like(x)
    random_func = random_ker[len(x.shape)][x.dtype.type]
    if len(x.shape) == 1:
        random_func(mask, np.float32(p), block=(x.shape[0], 1, 1), grid=(1, 1, 1))
    else:
        # random_func(mask, np.int32(mask.shape[1]), np.float32(p), block=(*x.shape, 1), grid=(1, 1, 1))
        random_func(mask, np.int32(mask.shape[1]), np.float32(p), block=(32, 32, 1), grid=(x.shape[0]//32, x.shape[1] // 32, 1))
    return mask


def get_mask_gpu(shape, dtype=np.float32, p=0.):
    assert len(shape) in [1, 2], 'invalid number of dims'
    mask = gpuarray.empty(shape=shape, dtype=dtype)
    random_func = random_ker[len(shape)][dtype]
    if len(shape) == 1:
        random_func(mask, np.float32(p), block=(shape[0], 1, 1), grid=(1, 1, 1))
    else:
        # random_func(mask, np.int32(shape[1]), np.float32(p), block=(*shape, 1), grid=(1, 1, 1))
        random_func(mask, np.int32(shape[1]), np.float32(p), block=(32, 32, 1), grid=(shape[0] // 32, shape[1] // 32, 1))
    return mask


lstm_ker = {
    np.int32: random_lstm_int_func,
    np.float32: random_lstm_float_func,
    np.float64: random_lstm_double_func,
}


def dropout_mask_lstm_gpu(x: gpuarray.GPUArray, p=0.):
    assert x.dtype in [np.int32, np.float32, np.float64], 'invalid dtype'
    assert len(x.shape) == 2, 'x must have 2 dims'
    mask = gpuarray.empty_like(x)
    lstm_func = lstm_ker[x.dtype.type]
    # lstm_func(mask, np.int32(x.shape[1]), np.float32(p), block=(*x.shape, 1), grid=(1, 1, 1))
    lstm_func(mask, np.int32(x.shape[1]), np.float32(p), block=(32, 32, 1), grid=(x.shape[0]//32, x.shape[1]//32, 1))
    return mask


# def from_one_gpu(x: gpuarray.GPUArray) -> gpuarray.GPUArray:
#     ctype = 'float' if x.dtype == np.float32 else 'double'
#     from_one = ElementwiseKernel(
#         f"{ctype} *Y, {ctype} *x",
#         "Y[i] = 1.0 - x[i]",
#         "from_one")
#     y = pycuda.gpuarray.empty_like(x)
#     from_one(y, x)
#     return y


# CPU Support


def sigmoid(x: np.ndarray) -> np.ndarray:
    func = np.vectorize(lambda x: 1 / (1 + np.exp(-x)))
    return func(x)


def tanh(x: np.ndarray) -> np.ndarray:
    func = np.vectorize(lambda x: np.tanh(x))
    return func(x)


def relu(x: np.ndarray):
    func = np.vectorize(lambda x: max(x, 0))
    return func(x)


def relu_grad(x: np.ndarray):
    func = np.vectorize(lambda x: 1 if x > 0 else 0)
    return func(x)


def softmax(x: np.ndarray) -> np.ndarray:
    smax = np.empty_like(x)
    for i in range(x.shape[1]):
        exps = np.exp(x[:, i] - np.max(x[:, i]))
        smax[:, i] = exps / np.sum(exps)
    return smax


def dropout_mask(x: np.ndarray, p=0.):
    mask = np.random.choice([0, 1], size=x.shape, p=[p, 1-p]).astype(x.dtype)
    return mask


def get_mask(shape, dtype=np.float32, p=0.):
    mask = np.random.choice([0, 1], size=shape, p=[p, 1-p]).astype(dtype)
    return mask


def dropout_mask_lstm(x: np.ndarray, p=0.):
    mask = np.random.choice([0, 1], size=(x.shape[0],), p=[p, 1-p]).astype(x.dtype)
    return mask


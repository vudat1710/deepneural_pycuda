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
    """
    Y[i] = tanh(x[i])
    """,
    "tanh_float")

tanh_double_ker = ElementwiseKernel(
    f"double *Y, double *x",
    """
    double pos_exp = exp (x[i]);
    double neg_exp = exp (-x[i]);
    Y[i] = (pos_exp - neg_exp) / (pos_exp + neg_exp)
    """,
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

random_ker_template = """
#include <curand_kernel.h>
#define ULL unsigned long long
extern "C" {
    __global__ void random_%(p)s_array(%(p)s *arr, int array_len, %(p)s p) {
        curandState cr_state; 
        int tid = blockIdx.x * blockDim.x + threadIdx.x;
        int num_iters = (array_len - 1) / blockDim.x + 1;
        curand_init( (ULL) clock() + (ULL) tid, (ULL) 0, (ULL) 0, &cr_state);
        %(p)s x;
        for (int j = 0; j < num_iters; j++) {
            int i = j * blockDim.x + tid;
            if (i < array_len) {
                x = curand_uniform%(p_curand)s(&cr_state);
                if (x < p) arr[i] = 0;
                else arr[i] = 1.0 / (1 - p);
            }
        }
    }
}
"""

random_float_func = SourceModule(
    no_extern_c=True,
    source=random_ker_template % {'p': 'float', 'p_curand': ''}
).get_function('random_float_array')

random_double_func = SourceModule(
    no_extern_c=True,
    source=random_ker_template % {'p': 'double', 'p_curand': '_double'}
).get_function('random_double_array')

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

norm_float_gpu = ReductionKernel(
    dtype_out=np.float32,
    neutral='0',
    reduce_expr="a + b",
    map_expr="x[i] * x[i]",
    arguments="float *x",
    name='norm',
)

norm_double_gpu = ReductionKernel(
    dtype_out=np.float64,
    neutral='0',
    reduce_expr="a + b",
    map_expr="x[i] * x[i]",
    arguments="double *x",
    name='norm',
)

norm_int_gpu = ReductionKernel(
    dtype_out=np.int32,
    neutral='0',
    reduce_expr="a + b",
    map_expr="x[i] * x[i]",
    arguments="int *x",
    name='norm',
)


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


# expand gpu
expand_dim1_template = "%(p)s *in, %(p)s *out, int copies"
expand_dim1_float_ker = ElementwiseKernel(
    arguments=expand_dim1_template % {'p': 'float'},
    operation="for (int n = 0; n < copies; n++) out[i * copies + n] = in[i]",
    name="expand_dim1_float_ker",
)
expand_dim1_double_ker = ElementwiseKernel(
    arguments=expand_dim1_template % {'p': 'double'},
    operation="for (int n = 0; n < copies; n++) out[i * copies + n] = in[i]",
    name="expand_dim1_double_ker",
)

expand_dim1_v2_template = "%(p)s *out, %(p)s *inp, int copies"
expand_dim1_float_v2_ker = ElementwiseKernel(
    arguments=expand_dim1_v2_template % {'p': 'float'},
    operation="out[i] = inp[i / copies]",
    name="expand_dim1_float_v2",
)

expand_dim1_double_v2_ker = ElementwiseKernel(
    arguments=expand_dim1_v2_template % {'p': 'double'},
    operation="out[i] = inp[i / copies]",
    name="expand_dim1_double_v2",
)

expand_dim0_template = "%(p)s *in, %(p)s *out, int copies, int width"
expand_dim0_float_ker = ElementwiseKernel(
    arguments=expand_dim0_template % {'p': 'float'},
    operation="for (int n = 0; n < copies; n++) out[n * width + i] = in[i]",
    name="expand_dim0_float_ker",
)
expand_dim0_double_ker = ElementwiseKernel(
    arguments=expand_dim0_template % {'p': 'double'},
    operation="for (int n = 0; n < copies; n++) out[n * width + i] = in[i]",
    name="expand_dim0_double_ker",
)

expand_dim0_v2_template = "%(p)s *out, %(p)s *inp, int width"
expand_dim0_float_v2_ker = ElementwiseKernel(
    arguments=expand_dim0_v2_template % {'p': 'float'},
    operation="out[i] = inp[i % width]",
    name="expand_dim0_float_v2",
)
expand_dim0_double_v2_ker = ElementwiseKernel(
    arguments=expand_dim0_v2_template % {'p': 'double'},
    operation="out[i] = inp[i % width]",
    name="expand_dim0_double_v2",
)

expand_ker = {
    "v1": {
        0: {
            np.float32: expand_dim0_float_ker,
            np.float64: expand_dim0_double_ker,
        },
        1: {
            np.float32: expand_dim1_float_ker,
            np.float64: expand_dim1_double_ker,
        }
    },
    "v2": {
        0: {
            np.float32: expand_dim0_float_v2_ker,
            np.float64: expand_dim0_double_v2_ker,
        },
        1: {
            np.float32: expand_dim1_float_v2_ker,
            np.float64: expand_dim1_double_v2_ker,
        }
    }
}


def expand_gpu_v1(x: gpuarray.GPUArray, dim: int, copies):
    expand_func = expand_ker["v1"][dim][x.dtype.type]
    if dim == 0:
        y = gpuarray.empty((copies, x.shape[0]), dtype=x.dtype)
        expand_func(x, y, np.int32(copies), np.int32(x.shape[0]))
    else:
        y = gpuarray.empty((x.shape[0], copies), dtype=x.dtype)
        expand_func(x, y, np.int32(copies))
    return y


def expand_gpu(x: gpuarray.GPUArray, dim: int, copies):
    expand_func = expand_ker["v2"][dim][x.dtype.type]
    if dim == 0:
        y = gpuarray.empty((copies, x.shape[0]), dtype=x.dtype)
        expand_func(y, x, np.int32(x.shape[0]))
    else:
        y = gpuarray.empty((x.shape[0], copies), dtype=x.dtype)
        expand_func(y, x, np.int32(copies))
    return y


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


def dropout_mask_gpu(x: gpuarray.GPUArray, p=0.):
    assert x.dtype in [np.int32, np.float32, np.float64], 'invalid dtype'
    assert len(x.shape) in [1, 2], 'invalid number of dims'
    mask = gpuarray.empty_like(x)
    random_func = random_float_func if x.dtype == np.float32 else random_double_func
    random_func(mask, np.int32(mask.size), x.dtype.type(p), block=(mask.size if mask.size < 1024 else 1024, 1, 1), grid=(1, 1, 1))
    return mask


def get_mask_gpu(shape, dtype=np.float32, p=0.):
    assert len(shape) in [1, 2], 'invalid number of dims'
    mask = gpuarray.empty(shape=shape, dtype=dtype)
    random_func = random_float_func if x.dtype == np.float32 else random_double_func
    random_func(mask, np.int32(mask.size), dtype(p), block=(mask.size if mask.size < 1024 else 1024, 1, 1), grid=(1, 1, 1))
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
    lstm_func(mask, np.int32(x.shape[1]), np.float32(p), block=(32, 32, 1),
              grid=((x.shape[0] - 1) // 32 + 1, (x.shape[1] - 1) // 32 + 1, 1))
    return mask


norm_ker = {
    np.int32: norm_int_gpu,
    np.float32: norm_float_gpu,
    np.float64: norm_double_gpu,
}


def norm_gpu(x: gpuarray.GPUArray):
    norm_func = norm_ker[x.dtype.type]
    return norm_func(x).get() ** 0.5


# embedding indices select
indices_select_ker_template = "%(p)s *out, int *indices, %(p)s *embedding, int embedding_dim"
indices_select_float_ker = ElementwiseKernel(
    arguments=indices_select_ker_template % {'p': 'float'},
    operation="""
        int vec_i = i / embedding_dim;
        int pos_in_vec = i % embedding_dim;
        out[i] = embedding[indices[vec_i] * embedding_dim + pos_in_vec];
        """,
    name="index_select",
)

indices_select_double_ker = ElementwiseKernel(
    arguments=indices_select_ker_template % {'p': 'double'},
    operation="""
        int vec_i = i / embedding_dim;
        int pos_in_vec = i % embedding_dim;
        out[i] = embedding[indices[vec_i] * embedding_dim + pos_in_vec];
        """,
    name="index_select_double_ker",
)


def indices_select(embedding: gpuarray.GPUArray, indices: gpuarray.GPUArray):
    indices_select_func = indices_select_float_ker if embedding.dtype == np.float32 else indices_select_double_ker
    y = gpuarray.empty((*indices.shape, embedding.shape[1]), dtype=embedding.dtype)
    indices_select_func(y, indices, embedding, np.int32(embedding.shape[1]))
    return y


# loss
def bce_with_logits(predicted, target):
    if predicted.shape != target.shape:
        raise ValueError("logits and labels must have the same shape ({} vs {})".format
                         (predicted.shape, target.shape))

    """
        Logistic loss formula is x - x * z + log(1 + exp(-x))
        For x < 0, a more stable formula is -x * z + log(1 + exp(x))
        Generally the formula we need is max(x, 0) - x * z + log(1 + exp(-abs(x)))
        """

    zeros = np.zeros_like(predicted, dtype=predicted.dtype)
    cond = (predicted >= zeros)
    relu_pred = np.where(cond, predicted, zeros)
    neg_abs_pred = np.where(cond, -predicted, predicted)

    return np.add(relu_pred - predicted * target, np.log1p(np.exp(neg_abs_pred)))


# adam support func gpu
adam_mean_float_ker = ElementwiseKernel(
    arguments="float *out, float *mean, float *grad, float d",
    operation="""
    out[i] = d * mean[i] + (1 - d) * grad[i]
    """,
    name="adam_mean_float_ker",
)
adam_mean_double_ker = ElementwiseKernel(
    arguments="double *out, double *mean, double *grad, double d",
    operation="""
    out[i] = d * mean[i] + (1 - d) * grad[i]
    """,
    name="adam_mean_double_ker",
)

adam_var_float_ker = ElementwiseKernel(
    arguments="float *out, float *var, float * grad, float d",
    operation="out[i] = d * var[i] + (1 - d) * grad[i] * grad[i]",
    name="adam_var_float_ker",
)

adam_var_double_ker = ElementwiseKernel(
    arguments="double *out, double *var, double * grad, double d",
    operation="out[i] = d * var[i] + (1 - d) * grad[i] * grad[i]",
    name="adam_var_double_ker",
)

adam_grad_float_ker = ElementwiseKernel(
    arguments="float *out, float *mean, float *var, float d1_t, float d2_t, float eps",
    operation="out[i] = (mean[i] / (1 - d1_t)) / (sqrt(var[i] / (1 - d2_t)) + eps)",
    name="adam_grad_float_ker"
)

adam_grad_double_ker = ElementwiseKernel(
    arguments="double *out, double *mean, double *var, double d1_t, double d2_t, double eps",
    operation="out[i] = (mean[i] / (1 - d1_t)) / (sqrt(var[i] / (1 - d2_t)) + eps)",
    name="adam_grad_double_ker"
)


def adam_mean(mean: gpuarray.GPUArray, grad: gpuarray.GPUArray, d1, out: gpuarray.GPUArray = None):
    adam_mean_func = adam_mean_float_ker if mean.dtype == np.float32 else adam_mean_double_ker
    if out is None:
        out = gpuarray.empty_like(mean)
    adam_mean_func(out, mean, grad, mean.dtype.type(d1))
    return out


def adam_var(var: gpuarray.GPUArray, grad: gpuarray.GPUArray, d2, out: gpuarray.GPUArray = None):
    adam_var_func = adam_var_float_ker if var.dtype == np.float32 else adam_var_double_ker
    if out is None:
        out = gpuarray.empty_like(var)
    adam_var_func(out, var, grad, var.dtype.type(d2))
    return out


def adam_grad(mean: gpuarray.GPUArray, var: gpuarray.GPUArray, d1, d2, eps, t, out: gpuarray.GPUArray = None):
    adam_grad_func = adam_grad_float_ker if mean.dtype == np.float32 else adam_grad_double_ker
    if out is None:
        out = gpuarray.empty_like(var)
    adam_grad_func(out, mean, var, mean.dtype.type(d1 ** t), mean.dtype.type(d2 ** t), mean.dtype.type(eps))
    return out


# add inplace
add_inplace_float_ker = ElementwiseKernel(
    arguments="float *x, float *y",
    operation="x[i] += y[i]",
    name="add_inplace_float",
)

add_inplace_double_ker = ElementwiseKernel(
    arguments="double *x, double *y",
    operation="x[i] += y[i]",
    name="add_inplace_double",
)


def add_(x: gpuarray.GPUArray, y: gpuarray.GPUArray):
    assert x.shape == y.shape, f"x and y must have same shape"
    assert x.dtype == y.dtype, f"x and y must have same dtype"
    add_func = add_inplace_float_ker if x.dtype == np.float32 else add_inplace_double_ker
    add_func(x, y)
    return x


# sub inplace
sub_inplace_float_ker = ElementwiseKernel(
    arguments="float *x, float *y, float c",
    operation="x[i] -= c * y[i]",
    name="sub_inplace_float"
)

sub_inplace_double_ker = ElementwiseKernel(
    arguments="double *x, double *y, double c",
    operation="x[i] -= c * y[i]",
    name="sub_inplace_double"
)


def sub_(x: gpuarray.GPUArray, y: gpuarray.GPUArray, c):
    assert x.shape == y.shape, f"x and y must have same shape"
    assert x.dtype == y.dtype, f"x and y must have same dtype"
    sub_func = sub_inplace_float_ker if x.dtype == np.float32 else sub_inplace_double_ker
    sub_func(x, y, x.dtype.type(c))
    return x


# zero inplace
zero_inplace_float_ker = ElementwiseKernel(
    arguments="float *x",
    operation="x[i] = 0.0",
    name="zero_float"
)

zero_inplace_double_ker = ElementwiseKernel(
    arguments="double *x",
    operation="x[i] = 0.0",
    name="zero_double"
)


def zero_(x: gpuarray.GPUArray):
    zero_func = zero_inplace_float_ker if x.dtype == np.float32 else zero_inplace_double_ker
    zero_func(x)
    return x


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
    temp = np.exp(x)
    softmax_output = temp / np.sum(temp,
                                   axis=len(x.shape) - 1,
                                   keepdims=True)
    return softmax_output


def dropout_mask(x: np.ndarray, p=0.):
    mask = np.random.choice([0, 1], size=x.shape, p=[p, 1 - p]).astype(x.dtype)
    return mask


def get_mask(shape, dtype=np.float32, p=0.):
    mask = np.random.choice([0, 1], size=shape, p=[p, 1 - p]).astype(dtype)
    return mask


def dropout_mask_lstm(x: np.ndarray, p=0.):
    mask = np.random.choice([0, 1], size=(x.shape[0],), p=[p, 1 - p]).astype(x.dtype)
    return mask

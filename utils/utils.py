import pycuda.autoinit
import pycuda.gpuarray as gpuarray
import pycuda.elementwise as elementwise
import pycuda.reduction as reduction
from pycuda.tools import context_dependent_memoize, dtype_to_ctype
from pycuda.compiler import SourceModule
import pycuda.driver as drv
from string import Template
from ..tensor import Tensor
from skcuda import cublas
import numbers
import numpy as np
from typing import Tuple


global _global_cublas_allocator, _global_cublas_handle
_global_cublas_allocator = drv.mem_alloc
_global_cublas_handle = cublas.cublasCreate()



def _get_binaryop_vecmat_kernel(dtype: np.dtype, binary_op: str) -> Tuple[SourceModule, SourceModule]:
    template = Template("""
    #include <pycuda-complex.hpp>

    __global__ void opColVecToMat(const ${type} *mat, const ${type} *vec, ${type} *out,
                                   const int n, const int m){
        const int tx = threadIdx.x;
        const int ty = threadIdx.y;
        const int tidx = blockIdx.x * blockDim.x + threadIdx.x;
        const int tidy = blockIdx.y * blockDim.y + threadIdx.y;

        extern __shared__ ${type} shared_vec[];

        if ((ty == 0) & (tidx < n))
            shared_vec[tx] = vec[tidx];
        __syncthreads();

        if ((tidy < m) & (tidx < n)) {
            out[tidx*m+tidy] = mat[tidx*m+tidy] ${binary_op} shared_vec[tx];
        }
    }

    __global__ void opRowVecToMat(const ${type}* mat, const ${type}* vec, ${type}* out,
                                   const int n, const int m){
        const int tx = threadIdx.x;
        const int ty = threadIdx.y;
        const int tidx = blockIdx.x * blockDim.x + threadIdx.x;
        const int tidy = blockIdx.y * blockDim.y + threadIdx.y;

        extern __shared__ ${type} shared_vec[];

        if ((tx == 0) & (tidy < m))
            shared_vec[ty] = vec[tidy];
        __syncthreads();

        if ((tidy < m) & (tidx < n)) {
            out[tidx*m+tidy] = mat[tidx*m+tidy] ${binary_op} shared_vec[ty];
        }
    }""")
    cache_dir = None
    ctype = dtype_to_ctype(dtype)
    tmpl = template.substitute(type=ctype, binary_op=binary_op)
    mod = SourceModule(tmpl)

    add_row_vec_kernel = mod.get_function('opRowVecToMat')
    add_col_vec_kernel = mod.get_function('opColVecToMat')
    return add_row_vec_kernel, add_col_vec_kernel


def binaryop_matvec(binary_op: str, x_gpu: Tensor, a_gpu: Tensor, axis: int = None, out: Tensor = None, stream: drv.Stream = None) -> Tensor:
    """
    Applies a binary operation to a vector and each column/row of a matrix.

    The numpy broadcasting rules apply so this would yield the same result
    as `x_gpu.get()` op `a_gpu.get()` in host-code.

    Parameters
    ----------
    binary_op : string, ['+', '-', '/', '*' '%']
        The operator to apply
    x_gpu : pycuda.gpuarray.GPUArray
        Matrix to which to add the vector.
    a_gpu : pycuda.gpuarray.GPUArray
        Vector to add to `x_gpu`.
    axis : int (optional)
        The axis onto which the vector is added. By default this is
        determined automatically by using the first axis with the correct
        dimensionality.
    out : pycuda.gpuarray.GPUArray (optional)
        Optional destination matrix.
    stream : pycuda.driver.Stream (optional)
        Optional Stream in which to perform this calculation.

    Returns
    -------
    out : pycuda.gpuarray.GPUArray
        result of `x_gpu` + `a_gpu`
    """
    if axis is None:
        if len(a_gpu.shape) == 1:
            if a_gpu.shape[0] == x_gpu.shape[1]:
                axis = 1
            else:
                raise ValueError(
                    "operands could not be broadcast together "
                    "with shapes %s %s" % (x_gpu.shape, a_gpu.shape))
        elif a_gpu.shape[1] == x_gpu.shape[1]:  # numpy matches inner axes first
            axis = 1
        elif a_gpu.shape[0] == x_gpu.shape[0]:
            axis = 0
        else:
            raise ValueError(
                "operands could not be broadcast together "
                "with shapes %s %s" % (x_gpu.shape, a_gpu.shape))
    else:
        if axis < 0:
            axis += 2
        if axis > 1:
            raise ValueError('invalid axis')

    if binary_op not in ['+', '-', '/', '*', '%']:
        raise ValueError('invalid operator')

    row_kernel, col_kernel = _get_binaryop_vecmat_kernel(
        x_gpu.dtype, binary_op)
    n, m = np.int32(x_gpu.shape[0]), np.int32(x_gpu.shape[1])

    block = (24, 24, 1)
    gridx = int(n // block[0] + 1 * (n % block[0] != 0))
    gridy = int(m // block[1] + 1 * (m % block[1] != 0))
    grid = (gridx, gridy, 1)

    if out is None:
        alloc = _global_cublas_allocator
        out = gpuarray.empty_like(x_gpu)
    else:
        assert out.dtype == x_gpu.dtype
        assert out.shape == x_gpu.shape

    if x_gpu.flags.c_contiguous:
        if axis == 0:
            col_kernel(x_gpu, a_gpu, out, n, m,
                       block=block, grid=grid, stream=stream,
                       shared=24*x_gpu.dtype.itemsize)
        elif axis == 1:
            row_kernel(x_gpu, a_gpu, out, n, m,
                       block=block, grid=grid, stream=stream,
                       shared=24*x_gpu.dtype.itemsize)
    else:
        if axis == 0:
            row_kernel(x_gpu, a_gpu, out, m, n,
                       block=block, grid=grid, stream=stream,
                       shared=24*x_gpu.dtype.itemsize)
        elif axis == 1:
            col_kernel(x_gpu, a_gpu, out, m, n,
                       block=block, grid=grid, stream=stream,
                       shared=24*x_gpu.dtype.itemsize)
    return out


def add_dot(a_gpu: Tensor, b_gpu: Tensor, c_gpu: Tensor, transa: str = 'N', transb: str = 'N', alpha: float = 1.0, beta: float = 1.0, handle: int =None) -> Tensor:
    """
    Calculates the dot product of two arrays and adds it to a third matrix.

    In essence, this computes

    C =  alpha * (A B) + beta * C

    For 2D arrays of shapes `(m, k)` and `(k, n)`, it computes the matrix
    product; the result has shape `(m, n)`.

    Parameters
    ----------
    a_gpu : pycuda.gpuarray.GPUArray
        Input array.
    b_gpu : pycuda.gpuarray.GPUArray
        Input array.
    c_gpu : pycuda.gpuarray.GPUArray
        Cumulative array.
    transa : char
        If 'T', compute the product of the transpose of `a_gpu`.
        If 'C', compute the product of the Hermitian of `a_gpu`.
    transb : char
        If 'T', compute the product of the transpose of `b_gpu`.
        If 'C', compute the product of the Hermitian of `b_gpu`.
    handle : int (optional)
        CUBLAS context. If no context is specified, the default handle from
        `skcuda.misc._global_cublas_handle` is used.

    Returns
    -------
    c_gpu : pycuda.gpuarray.GPUArray

    Notes
    -----
    The matrices must all contain elements of the same data type.
    """

    if handle is None:
        handle = _global_cublas_handle

    # Get the shapes of the arguments (accounting for the
    # possibility that one of them may only have one dimension):
    a_shape = a_gpu.shape
    b_shape = b_gpu.shape
    if len(a_shape) == 1:
        a_shape = (1, a_shape[0])
    if len(b_shape) == 1:
        b_shape = (1, b_shape[0])

    # Perform matrix multiplication for 2D arrays:
    if (a_gpu.dtype == np.complex64 and b_gpu.dtype == np.complex64):
        cublas_func = cublas.cublasCgemm
        alpha = np.complex64(alpha)
        beta = np.complex64(beta)
    elif (a_gpu.dtype == np.float32 and b_gpu.dtype == np.float32):
        cublas_func = cublas.cublasSgemm
        alpha = np.float32(alpha)
        beta = np.float32(beta)
    elif (a_gpu.dtype == np.complex128 and b_gpu.dtype == np.complex128):
        cublas_func = cublas.cublasZgemm
        alpha = np.complex128(alpha)
        beta = np.complex128(beta)
    elif (a_gpu.dtype == np.float64 and b_gpu.dtype == np.float64):
        cublas_func = cublas.cublasDgemm
        alpha = np.float64(alpha)
        beta = np.float64(beta)
    else:
        raise ValueError('unsupported combination of input types')

    transa = transa.lower()
    transb = transb.lower()

    a_f_order = a_gpu.strides[1] > a_gpu.strides[0]
    b_f_order = b_gpu.strides[1] > b_gpu.strides[0]
    c_f_order = c_gpu.strides[1] > c_gpu.strides[0]

    if a_f_order != b_f_order:
        raise ValueError('unsupported combination of input order')
    if a_f_order != c_f_order:
        raise ValueError('invalid order for c_gpu')

    if a_f_order:  # F order array
        if transa in ['t', 'c']:
            k, m = a_shape
        elif transa in ['n']:
            m, k = a_shape
        else:
            raise ValueError('invalid value for transa')

        if transb in ['t', 'c']:
            n, l = b_shape
        elif transb in ['n']:
            l, n = b_shape
        else:
            raise ValueError('invalid value for transb')
        
        if l != k:
            raise ValueError('objects are not aligned')

        lda = max(1, a_gpu.strides[1] // a_gpu.dtype.itemsize)
        ldb = max(1, b_gpu.strides[1] // b_gpu.dtype.itemsize)
        ldc = max(1, c_gpu.strides[1] // c_gpu.dtype.itemsize)

        if c_gpu.shape != (m, n) or c_gpu.dtype != a_gpu.dtype:
            raise ValueError('invalid value for c_gpu')
        cublas_func(handle, transa, transb, m, n, k, alpha, a_gpu.gpudata,
                lda, b_gpu.gpudata, ldb, beta, c_gpu.gpudata, ldc)
    else:
        if transb in ['t', 'c']:
            m, k = b_shape
        elif transb in ['n']:
            k, m = b_shape
        else:
            raise ValueError('invalid value for transb')

        if transa in ['t', 'c']:
            l, n = a_shape
        elif transa in ['n']:
            n, l = a_shape
        else:
            raise ValueError('invalid value for transa')

        if l != k:
            raise ValueError('objects are not aligned')

        lda = max(1, a_gpu.strides[0] // a_gpu.dtype.itemsize)
        ldb = max(1, b_gpu.strides[0] // b_gpu.dtype.itemsize)
        ldc = max(1, c_gpu.strides[0] // c_gpu.dtype.itemsize)

        # Note that the desired shape of the output matrix is the transpose
        # of what CUBLAS assumes:
        if c_gpu.shape != (n, m) or c_gpu.dtype != a_gpu.dtype:
            raise ValueError('invalid value for c_gpu')
        cublas_func(handle, transb, transa, m, n, k, alpha, b_gpu.gpudata,
                ldb, a_gpu.gpudata, lda, beta, c_gpu.gpudata, ldc)
    return c_gpu

def dot(x_gpu: Tensor, y_gpu: Tensor, transa: str = 'N', transb: str = 'N', handle: int = None, out: Tensor = None):
    """
    Dot product of two arrays.

    For 1D arrays, this function computes the inner product. For 2D
    arrays of shapes `(m, k)` and `(k, n)`, it computes the matrix
    product; the result has shape `(m, n)`.

    Parameters
    ----------
    x_gpu : pycuda.gpuarray.GPUArray
        Input array.
    y_gpu : pycuda.gpuarray.GPUArray
        Input array.
    transa : char
        If 'T', compute the product of the transpose of `x_gpu`.
        If 'C', compute the product of the Hermitian of `x_gpu`.
    transb : char
        If 'T', compute the product of the transpose of `y_gpu`.
        If 'C', compute the product of the Hermitian of `y_gpu`.
    handle : int
        CUBLAS context. If no context is specified, the default handle from
        `skcuda.misc._global_cublas_handle` is used.
    out : pycuda.gpuarray.GPUArray, optional
        Output argument. Will be used to store the result.

    Returns
    -------
    c_gpu : pycuda.gpuarray.GPUArray, float{32,64}, or complex{64,128}
        Inner product of `x_gpu` and `y_gpu`. When the inputs are 1D
        arrays, the result will be returned as a scalar.

    Notes
    -----
    The input matrices must all contain elements of the same data type.

    Examples
    --------
    >>> import pycuda.autoinit
    >>> import pycuda.gpuarray as gpuarray
    >>> import numpy as np
    >>> import skcuda.linalg as linalg
    >>> import skcuda.misc as misc
    >>> linalg.init()
    >>> a = np.asarray(np.random.rand(4, 2), np.float32)
    >>> b = np.asarray(np.random.rand(2, 2), np.float32)
    >>> a_gpu = gpuarray.to_gpu(a)
    >>> b_gpu = gpuarray.to_gpu(b)
    >>> c_gpu = linalg.dot(a_gpu, b_gpu)
    >>> np.allclose(np.dot(a, b), c_gpu.get())
    True
    >>> d = np.asarray(np.random.rand(5), np.float32)
    >>> e = np.asarray(np.random.rand(5), np.float32)
    >>> d_gpu = gpuarray.to_gpu(d)
    >>> e_gpu = gpuarray.to_gpu(e)
    >>> f = linalg.dot(d_gpu, e_gpu)
    >>> np.allclose(np.dot(d, e), f)
    True

    """
    if handle is None:
        handle = _global_cublas_handle

    x_shape = x_gpu.shape
    y_shape = y_gpu.shape

    # When one argument is a vector and the other a matrix, increase the number
    # of dimensions of the vector to 2 so that they can be multiplied using
    # GEMM, but also set the shape of the output to 1 dimension to conform with
    # the behavior of numpy.dot:
    if len(x_shape) == 1 and len(y_shape) > 1:
        out_shape = (y_shape[1],)
        x_shape = (1, x_shape[0])
        x_gpu = x_gpu.reshape(x_shape)
    elif len(x_shape) > 1 and len(y_shape) == 1:
        out_shape = (x_shape[0],)
        y_shape = (y_shape[0], 1)
        y_gpu = y_gpu.reshape(y_shape)

    if len(x_gpu.shape) == 1 and len(y_gpu.shape) == 1:
        if x_gpu.size != y_gpu.size:
            raise ValueError('arrays must be of same length')

        # Compute inner product for 1D arrays:
        if (x_gpu.dtype == np.complex64 and y_gpu.dtype == np.complex64):
            cublas_func = cublas.cublasCdotu
        elif (x_gpu.dtype == np.float32 and y_gpu.dtype == np.float32):
            cublas_func = cublas.cublasSdot
        elif (x_gpu.dtype == np.complex128 and y_gpu.dtype == np.complex128):
            cublas_func = cublas.cublasZdotu
        elif (x_gpu.dtype == np.float64 and y_gpu.dtype == np.float64):
            cublas_func = cublas.cublasDdot
        else:
            raise ValueError('unsupported combination of input types')

        return cublas_func(handle, x_gpu.size, x_gpu.gpudata, 1,
                           y_gpu.gpudata, 1)
    else:
        transa = transa.lower()
        transb = transb.lower()
        if out is None:
            if transa in ['t', 'c']:
                k, m = x_shape
            else:
                m, k = x_shape

            if transb in ['t', 'c']:
                n, l = y_shape
            else:
                l, n = y_shape

            alloc = _global_cublas_allocator
            if x_gpu.strides[1] > x_gpu.strides[0]: # F order
                out = gpuarray.empty((m, n), x_gpu.dtype, order="F", allocator=alloc)
            else:
                out = gpuarray.empty((m, n), x_gpu.dtype, order="C", allocator=alloc)

    add_dot(x_gpu, y_gpu, out, transa, transb, 1.0, 0.0, handle)
    if 'out_shape' in locals():
        return out.reshape(out_shape)
    else:
        return out


def transpose(a_gpu: Tensor, conj: bool = False, handle: int = None) -> Tensor:
    """
    Matrix transpose.

    Transpose a matrix in device memory and return an object
    representing the transposed matrix.

    Parameters
    ----------
    a_gpu : pycuda.gpuarray.GPUArray
        Input matrix of shape `(m, n)`.

    Returns
    -------
    at_gpu : pycuda.gpuarray.GPUArray
        Transposed matrix of shape `(n, m)`.
    handle : int
        CUBLAS context. If no context is specified, the default handle from
        `skcuda.misc._global_cublas_handle` is used.

    Examples
    --------
    >>> import pycuda.autoinit
    >>> import pycuda.driver as drv
    >>> import pycuda.gpuarray as gpuarray
    >>> import numpy as np
    >>> import skcuda.linalg as linalg
    >>> linalg.init()
    >>> a = np.array([[1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12]], np.float32)
    >>> a_gpu = gpuarray.to_gpu(a)
    >>> at_gpu = linalg.transpose(a_gpu)
    >>> np.all(a.T == at_gpu.get())
    True
    >>> b = np.array([[1j, 2j, 3j, 4j, 5j, 6j], [7j, 8j, 9j, 10j, 11j, 12j]], np.complex64)
    >>> b_gpu = gpuarray.to_gpu(b)
    >>> bt_gpu = linalg.transpose(b_gpu)
    >>> np.all(b.T == bt_gpu.get())
    True
    """

    if handle is None:
        handle = _global_cublas_handle

    if len(a_gpu.shape) != 2:
        raise ValueError('a_gpu must be a matrix')

    if (a_gpu.dtype == np.complex64):
        func = cublas.cublasCgeam
    elif (a_gpu.dtype == np.float32):
        func = cublas.cublasSgeam
    elif (a_gpu.dtype == np.complex128):
        func = cublas.cublasZgeam
    elif (a_gpu.dtype == np.float64):
        func = cublas.cublasDgeam
    else:
        raise ValueError('unsupported input type')

    if conj:
        transa = 'c'
    else:
        transa = 't'
    M, N = a_gpu.shape
    at_gpu = gpuarray.empty((N, M), a_gpu.dtype)
    func(handle, transa, 't', M, N,
         1.0, a_gpu.gpudata, N, 0.0, a_gpu.gpudata, N,
         at_gpu.gpudata, M)
    return at_gpu

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

    out = Tensor(shape, dtype, allocator, order=order)
    o = np.ones((), dtype)
    out.fill(o)
    return out

def sum_gpu(x_gpu: Tensor, axis: int = None, out: Tensor = None, keepdims: bool = False, calc_mean: bool = False, ddof: int = 0) -> Tensor:
    """
    Compute the sum along the specified axis.

    Parameters
    ----------
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
    global _global_cublas_allocator
    assert isinstance(ddof, numbers.Integral)

    if axis is None or len(x_gpu.shape) <= 1:
        out_shape = (1,)*len(x_gpu.shape) if keepdims else ()
        if calc_mean == False:
            return gpuarray.sum(x_gpu).reshape(out_shape)
        else:
            return gpuarray.sum(x_gpu).reshape(out_shape) / (x_gpu.dtype.type(x_gpu.size-ddof))

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
        alpha = (1.0 / (sum_axis-ddof))
    else:
        alpha = 1.0
    if (x_gpu.dtype == np.complex64):
        gemv = cublas.cublasCgemv
    elif (x_gpu.dtype == np.float32):
        gemv = cublas.cublasSgemv
    elif (x_gpu.dtype == np.complex128):
        gemv = cublas.cublasZgemv
    elif (x_gpu.dtype == np.float64):
        gemv = cublas.cublasDgemv

    alloc = _global_cublas_allocator
    ons = ones((sum_axis, ), x_gpu.dtype, allocator=alloc)

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

if __name__=="__main__":
    mat1 = gpuarray.to_gpu(np.random.rand(1,5).astype(np.float32))
    mat2 = gpuarray.to_gpu(np.random.rand(5,1).astype(np.float32))
    print(transpose(mat1))
from utils.utils import *
from utils.activation import *
from loss import BCEWithLogits


def test_utils():
    mat1 = gpuarray.to_gpu(np.random.rand(1, 5).astype(np.float32))
    # mat2 = gpuarray.to_gpu(np.random.rand(5, 1).astype(np.float32))
    print('before transpose\n', mat1)
    print('after transpose\n', transpose(mat1))


def test_bce_loss():
    loss_ = BCEWithLogits(
        predicted=np.array([0.1, 0.6, 0.8, 1]),
        target=np.array([0, 1, 1, 0]),
    )
    print(loss_)


def test_sum():
    x = gpuarray.to_gpu(np.array([
        [1, 2, 3, 4],
        [5, 6, 7, 8],
        [1, 5, 2, 7],
    ], dtype=np.float32))
    sum_1 = sum_gpu(x, axis=1, keepdims=True)
    print(sum_1.get())


def test_sigmoid():
    x_cpu = np.array([1, 2, 3], dtype=np.float64)
    x_gpu = gpuarray.to_gpu(x_cpu)
    print(sigmoid_gpu(x_gpu))
    print(sigmoid(x_cpu))


def test_softmax():
    x_cpu = np.array([[1, 2, 3], [1, 2, 3]], dtype=np.float64)
    x_gpu = gpuarray.to_gpu(x_cpu)
    print(softmax_gpu(x_gpu))


if __name__ == '__main__':
    # test_utils()
    # test_bce_loss()
    # test_sum()
    # test_sigmoid()
    test_softmax()

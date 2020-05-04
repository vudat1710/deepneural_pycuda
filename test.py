from utils.utils import *
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


if __name__ == '__main__':
    test_utils()
    # test_bce_loss()

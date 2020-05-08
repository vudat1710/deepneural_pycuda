""" Loss function is called take the input as numpy array
    instead of directly compute on GPU
"""

from numpy import ndarray
import numpy as np


def BCEWithLogits(predicted: ndarray, target: ndarray) -> ndarray:
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

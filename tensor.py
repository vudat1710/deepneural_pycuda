"""
Make sure that a tensor is a GPUArray type of pycuda.gpuarray
"""

import numpy as np
import pycuda.autoinit
from pycuda.gpuarray import GPUArray as Tensor
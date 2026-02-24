import numpy as np
from numpy.typing import NDArray

def layer_computation(inputs: NDArray, weights: NDArray, biases: NDArray):
    weights_transpose = weights.T
    dot_product = np.dot(inputs, weights_transpose)
    return dot_product + biases

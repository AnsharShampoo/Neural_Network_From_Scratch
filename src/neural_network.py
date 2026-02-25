import numpy as np
from numpy.typing import NDArray

def layer_computation(inputs: NDArray, weights: NDArray, biases: NDArray):
    weights_transpose = weights.T
    dot_product = np.dot(inputs, weights_transpose)
    return dot_product + biases

class fully_connected_layer():

    def __init__(self, input_size: int, n_neurons: int):
        self.weights = np.random.randn(input_size, n_neurons) * 0.1
        self.biases = np.zeros(n_neurons)

    def forward_computation(self, inputs):
        if not isinstance(inputs, np.ndarray):
            inputs = np.array(inputs)
        dot_product = np.dot(inputs, self.weights)
        return dot_product + self.biases
    
    def relu_activation(self, layer_outputs: NDArray):
        return np.maximum(0, layer_outputs)
    
    def softmax_activation(self, layer_outputs: NDArray):
        exponential = np.exp(layer_outputs)
        exponential_normalized = exponential/np.sum(exponential, axis=2, keepdims=True)
        return exponential_normalized
        
        

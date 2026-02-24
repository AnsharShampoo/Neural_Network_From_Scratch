import unittest
import numpy as np
from src.neural_network import layer_computation

class tests_neural_network(unittest.TestCase):

    def test_layer_computation(self):
        inputs = [
            [1.0, 2.0, 3.0, 2.5],
            [2.0, 5.0, -1.0, 2.0],
            [-1.5, 2.7, 3.3, -0.8],
        ]
        weights = [
            [0.2, 0.8, -0.5, 1.0],
            [0.5, -0.91, 0.26, -0.5],
            [-0.26, -0.27, 0.17, 0.87],
        ]
        biases = [2.0, 3.0, 0.5]

        computation = layer_computation(np.array(inputs), np.array(weights), np.array(biases))
        expected_result = np.array(
            [
                [4.8, 1.21, 2.385],
                [8.9, -1.81, 0.2],
                [1.41, 1.051, 0.026],
            ]
        )
        print(computation)
        assert np.allclose(expected_result, computation)
import numpy as np

from MultiLayeredPerceptron import MLP

NN = MLP(2, 4, 1)
NN.randomize

XOR_inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
XOR_desired_output = np.array([[0], [1], [1], [0]])

# Pretraining testing
for i in XOR_inputs:
    NN.forward(i, False)
    print(f"Desired: {XOR_desired_output}")
    print(f"Genetrated: {str(NN.outputs)}")

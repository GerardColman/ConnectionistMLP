import random

import numpy as np

from MultiLayeredPerceptron import MLP

# Initializing the MLP
NN = MLP(2, 4, 1)
NN.randomize()

# Declaring input, output, max_epochs, and learning rate
XOR_inputs = [[0, 0], [0, 1], [1, 0], [1, 1]]
XOR_desired_output = [[0], [1], [1], [0]]
max_epochs = 10000
learning_rate = 0.1

# Pretraining testing
print("==========PRE-TRAINING TESTING==========")
for i in range(len(XOR_inputs)):
    NN.forward(XOR_inputs[i], True)
    print(f"Target: {XOR_desired_output[i]} Output: {NN.outputs}")

# Training
print("==========TRAINING==========")
random.seed(1)
for e in range(max_epochs):
    error = []
    for i in range(len(XOR_inputs)):
        NN.forward(XOR_inputs[i], True)
        error.append(NN.backwards(XOR_desired_output[i], use_sigmoid= True))
        chance = random.randint(0,100)
        NN.update_weights(learning_rate)
        # if chance >= 0 and chance <= 40:
        #     NN.update_weights(learning_rate)
    print(f"Error at epoch {e} is {np.mean(error)}")
# NN.toString(1)


# Testing
print("==========TESTING==========")
for i in range(len(XOR_inputs)):
    NN.forward(XOR_inputs[i], True)
    print(f"Target: {XOR_desired_output[i]} Output: {NN.outputs}")
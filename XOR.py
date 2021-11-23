import numpy as np
from random import seed
from random import randint

from MultiLayeredPerceptron import MLP

# Initializing the MLP
NN = MLP(2, 4, 1)
NN.randomize()

# Declaring input, output, max_epochs, and learning rate
XOR_inputs = [[0, 0], [0, 1], [1, 0], [1, 1]]
XOR_desired_output = [[0], [1], [1], [0]]
max_epochs = 500
learning_rate = 1

# Pretraining testing
for i in XOR_inputs:
    error = 0
    NN.forward(i, False)
    NN.backwards(target= XOR_desired_output, use_sigmoid=True)
    # print(f"Desired: {XOR_desired_output}")
    # print(f"Genetrated: {NN.outputs}")

# Training
# seed()
# for e in range(max_epochs):
#     error = 0
#     for p in range(XOR_inputs):
#         NN.forward(XOR_inputs[i], False)
#         error += NN.backwards(NN.outputs)

#         chance = randint(0,100)
#         if chance >= 0 and chance <= 10:
#             NN.update_weights(learning_rate)
#     print(f"Error at epoch {e} is {error}")
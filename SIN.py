import math
import random

import numpy as np

from MultiLayeredPerceptron import MLP

SIN_Inputs = []
SIN_Desired_Outputs = []
max_epochs = 10000
learning_rate = 0.1

f = open("SINoutput.txt","w")

# Producing inputs
random.seed()
for i in range(500):
    temp = []
    for j in range(4):
        temp.append(random.randint(-1,1))
    SIN_Inputs.append(temp)

# Producing desired_outputs
for i in SIN_Inputs:
    sin_sum = 0
    sin_sum = math.sin((i[0]-(i[1]+i[2])-i[3]))
    temp = []
    temp.append(sin_sum)
    SIN_Desired_Outputs.append(temp)

# Initializing MLP
NN = MLP(4, 5, 1)
NN.randomize()

f.write("==========SETTINGS==========\n")
f.write("Input units = 4\n")
f.write("hidden units = 5\n")
f.write("output units = 1\n")
f.write(f"max epochs = {max_epochs}\n")
f.write(f"learning_rate = {learning_rate}\n")

# Pretraining testing
print("==========PRE-TRAINING TESTING==========")
f.write("==========PRE-TRAINING TESTING==========\n")
for i in range(len(SIN_Inputs)):
    NN.forward(SIN_Inputs[i], False)
    f.write(f"Target: {SIN_Desired_Outputs[i]} Output: {NN.outputs}\n")
    print(f"Target: {SIN_Desired_Outputs[i]} Output: {NN.outputs}")


# Training
print("==========TRAINING==========")
f.write(f"==========TRAINING==========\n")
random.seed(1)
for e in range(max_epochs):
    error = []
    for i in range(len(SIN_Inputs)):
        NN.forward(SIN_Inputs[i], False)
        error.append(NN.backwards(SIN_Desired_Outputs[i], use_sigmoid= False))
        NN.update_weights(learning_rate)
        # if chance >= 0 and chance <= 40:
        #     NN.update_weights(learning_rate)
    f.write(f"Error at epoch {e} is {np.mean(error)}\n")
    print(f"Error at epoch {e} is {np.mean(error)}")
# NN.toString(1)

# Testing
print("==========TESTING==========")
f.write("==========TESTING==========\n")
for i in range(len(SIN_Inputs)):
    NN.forward(SIN_Inputs[i], False)
    f.write(f"Target: {SIN_Desired_Outputs[i]} Output: {NN.outputs}\n")
    print(f"Target: {SIN_Desired_Outputs[i]} Output: {NN.outputs}")

f.close()
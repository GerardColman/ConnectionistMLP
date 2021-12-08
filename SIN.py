import math
import random

import numpy as np

from MultiLayeredPerceptron import MLP

SIN_Inputs = []
SIN_Desired_Outputs = []
SIN_Testing_Inputs = []
SIN_Testing_Outputs = []
max_epochs = 1000
learning_rate = 1

f = open("SIN Experiments/SINexperiment9.txt","w")

# Producing inputs
random.seed()
for i in range(500):
    temp = []
    for j in range(4):
        temp.append(random.randint(-1,1))
    if i > 400:
        SIN_Testing_Inputs.append(temp)
    else:
        SIN_Inputs.append(temp)

# Producing desired_outputs
for i in range(len(SIN_Inputs)):
    vector = SIN_Inputs[i]
    sin_sum = 0
    sin_sum = math.sin((vector[0]-(vector[1]+vector[2])-vector[3]))
    temp = []
    temp.append(sin_sum)
    SIN_Desired_Outputs.append(temp)

# Producing Testing Outputs:
for i in SIN_Testing_Inputs:
    sin_sum = 0
    sin_sum = math.sin((i[0]-(i[1]+i[2])-i[3]))
    temp = []
    temp.append(sin_sum)
    SIN_Testing_Outputs.append(temp)
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
latest_error = 0
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
    latest_error = np.mean(error)
# NN.toString(1)

# Testing
print("==========TESTING==========")
f.write("==========TESTING==========\n")
testing_error = 0
for i in range(len(SIN_Testing_Inputs)):
    NN.forward(SIN_Testing_Inputs[i], False)
    testing_error += NN.GetError(SIN_Testing_Outputs[i])
testing_error = np.mean(testing_error)
f.write(f"Testing Error: {testing_error}, Training Error: {latest_error}")
print(f"Testing Error: {testing_error}, Training Error: {latest_error}")

f.close()
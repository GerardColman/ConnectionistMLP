import math
import random as rand

import numpy as np

import MultiLayeredPerceptron as MLP

SIN_Inputs = []
SIN_Desired_Outputs = []

# Producing inputs
rand.seed()
for i in range(500):
    temp = []
    for j in range(4):
        temp.append(rand.randint(-1,1))
    SIN_Inputs.append(temp)

# Producing desired_outputs
for i in SIN_Inputs:
    sin_sum = 0
    sin_sum = math.sin((i[0]-(i[1]+i[2])-i[3]))
    SIN_Desired_Outputs.append(sin_sum)


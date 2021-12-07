import math
import statistics as stat
import random as rand

import numpy as np


class MLP:

    number_of_inputs = 0
    number_of_hidden_units = 0
    number_of_outputs = 0
    
    weights_lower = [] # W1 (Output)
    weights_upper = [] # W2 (Input)

    weight_changes_lower = [] # dw1
    weight_changes_upper = [] # dw2

    activations_lower = [] # z1 (Output)
    activations_upper = [] # z2 (Input)

    input_neurons = []
    hidden_neurons = [] # h
    outputs = [] # o

    def __init__(self, num_inputs, num_hidden, num_output):
        self.number_of_inputs = num_inputs
        self.number_of_hidden_units = num_hidden
        self.number_of_outputs = num_output

        self.input_neurons = [0]*num_inputs
        self.outputs = [0]*num_output
        self.hidden_neurons = [0]*num_hidden

        self.activations_lower = [0]*num_hidden
        self.activations_upper = [0]*num_output

        self.weights_upper = [] # Hidden x Output
        self.weights_lower = [] # Inputs x Hidden
        self.weight_changes_upper = [] # Hidden x Output
        self.weight_changes_lower = [] # Inputs x Hidden

    def randomize(self):
        rand.seed(1)
        
        # Initializing weights to random values
        for i in range(0, self.number_of_hidden_units):
            temp = []
            temp2 = []
            for j in range(0, self.number_of_outputs):
                temp.append(rand.uniform(0,1))
                temp2.append(0)
            self.weight_changes_upper.append(temp2)
            self.weights_upper.append(temp)

        # Initializing weights to random values
        for i in range(self.number_of_inputs):
            temp = []
            temp2 = []
            for j in range(self.number_of_hidden_units):
                temp.append(rand.uniform(0,1))
                temp2.append(0)
            self.weight_changes_lower.append(temp2)
            self.weights_lower.append(temp)

    def forward(self, Input, use_sigmoid):
        self.input_neurons = Input
        
        for i in range(self.number_of_hidden_units):
            neuron_activation = 0
            for j in range(self.number_of_inputs):
                neuron_activation += self.input_neurons[j] * self.weights_lower[j][i]
            if use_sigmoid:
                neuron_activation = self.sigmoid(neuron_activation)
            else:
                neuron_activation = self.hyperbolic_tangent(neuron_activation)
            self.activations_lower[i] = neuron_activation # NOTE: THIS MIGHT GO WRONG
            self.hidden_neurons[i] = neuron_activation

        for i in range(self.number_of_outputs):
            output_activation = 0
            for j in range(self.number_of_hidden_units):
                output_activation += self.hidden_neurons[j] * self.weights_upper[j][i]
            if use_sigmoid:
                output_activation = self.sigmoid(output_activation)
            else:
                output_activation = self.hyperbolic_tangent(output_activation)
            self.activations_upper[i] = output_activation # NOTE: THIS MIGHT GO WRONG
            self.outputs[i] = output_activation
        
        return 0
        
    def backwards(self, target, use_sigmoid):
        hidden_delta = [0.0] * self.number_of_hidden_units

        for i in range(self.number_of_hidden_units):
            err = 0
            for j in range(self.number_of_outputs):
                if use_sigmoid:
                    currentDelta = self.sigmoid(self.outputs[j], True) * (target[j] - self.outputs[j])
                else:
                    currentDelta = self.hyperbolic_tangent(self.outputs[j], True) * (target[j] - self.outputs[j])
                err += currentDelta * self.weights_upper[i][j] # Calculating Error
                self.weight_changes_upper[i][j] = currentDelta * self.hidden_neurons[i] # Calculating upper layer weight changes
            
            if use_sigmoid:
                hidden_delta[i] = self.sigmoid(self.hidden_neurons[i], True) * err
            else:
                hidden_delta[i] = self.hyperbolic_tangent(self.hidden_neurons[i], True) * err
        
        for i in range(self.number_of_inputs):
            for j in range(self.number_of_hidden_units):
                self.weight_changes_lower[i][j] = hidden_delta[j] * self.input_neurons[i]

        # print("Target: " + str(target))
        # print("outputs: " + str(self.outputs))
        return self.GetError(target)

    
    def update_weights(self, learning_rate):
        for i in range(self.number_of_inputs):
            for j in range(self.number_of_hidden_units):
                self.weights_lower[i][j] += self.weight_changes_lower[i][j] * learning_rate
                self.weight_changes_lower[i][j] = 0

        for i in range(self.number_of_hidden_units):
            for j in range(self.number_of_outputs):
                self.weights_upper[i][j] += self.weight_changes_upper[i][j] * learning_rate
                self.weight_changes_upper[i][j] = 0

    def sigmoid(self, sig, is_backward=False):
        if is_backward:
            return sig * (1-sig)
        else:
            return 1.0 / (1.0 + math.exp(-sig))

    def hyperbolic_tangent(self, tanh, is_backward=False):
        if is_backward:
            # print(f"derivitave: {1 - (math.tanh(tanh) ** 2)}")
            return 1 - ((math.tanh(tanh)) ** 2)
        else:
            # print(f"Regular tanh: {math.tanh(tanh)}")
            return math.tanh(tanh)

    def GetError(self, target):
        error = 0
        for n in range(0, self.number_of_outputs):
            error += ((target[n] - self.outputs[n]) ** 2) / 2
        return error
           
    def toString(self, epoch_number):
        print(f"Epoch Number: {epoch_number}")
        print("--WEIGHTS--")
        print(f"lower-weights: {self.weights_lower}")
        print(f"upper-weights: {self.weights_upper}")
        print("--WEIGHT UPDATES--")
        print(f"lower-weight-changes: {self.weight_changes_lower}")
        print(f"upper-weight-changes: {self.weight_changes_upper}")
        print("--INPUT--")
        print(f"input: {self.input_neurons}")
        print("--HIDDEN NEURONS--")
        print(f"hidden: {self.hidden_neurons}")
        print("--OUTPUT--")
        print(f"input: {self.outputs}")
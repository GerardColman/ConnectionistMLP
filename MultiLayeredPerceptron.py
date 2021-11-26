import math
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

        # Initialize weight arrays and weight_change arrays to correct size
        self.weights_upper = [[0]*self.number_of_hidden_units]*self.number_of_inputs
        self.weights_lower = [[0]*self.number_of_hidden_units]*self.number_of_outputs

        # Also sets weight change arrays to 0
        self.weight_changes_upper = [[0]*self.number_of_hidden_units]*self.number_of_inputs
        self.weight_changes_lower = [[0]*self.number_of_hidden_units]*self.number_of_outputs

    def randomize(self):
        rand.seed()
        
        # Initializing weights to random values
        for i in range(self.number_of_inputs):
            for j in range(self.number_of_hidden_units):
                self.weights_upper[i][j] = rand.uniform(0.0, 1.0)

        # Initializing weights to random values
        for i in range(self.number_of_outputs):
            for j in range(self.number_of_hidden_units):
                self.weights_lower[i][j] = rand.uniform(0.0, 1.0)

    def forward(self, Input, sigmoid):
        for i in range(self.number_of_inputs - 1):
            self.input_neurons[i] = Input[i]

        for i in range(self.number_of_hidden_units):
            hidden = 0.0
            for j in range(self.number_of_inputs):
                hidden += self.input_neurons[j] * self.weights_upper[j][i]
            self.activations_lower[i] = hidden
            if sigmoid:
                hidden = self.sigmoid(self.activations_lower[i])
            else:
                hidden = self.hyperbolic_tangent(self.activations_lower[i])
            self.hidden_neurons[i] = hidden

        for i in range(self.number_of_outputs):
            output = 0.0
            for j in range(self.number_of_hidden_units):
               output += self.hidden_neurons[j] * self.weights_lower[i][j]
            self.activations_upper[i] = output
            if sigmoid:
                output = self.sigmoid(self.activations_upper[i])
            else:
                output = self.hyperbolic_tangent(self.activations_upper[i])
            self.outputs[i] = output
    
    def backwards(self, target, use_sigmoid):
        output_delta = [0.0] * self.number_of_outputs
        hidden_delta = [0.0] * self.number_of_hidden_units

        # Computing Output Delta
        for i in range(self.number_of_outputs):
            err = target[i] - self.outputs[i]
            # print(self.outputs[i])
            if use_sigmoid:
                output_delta[i] = self.sigmoid(self.outputs[i], True) * err
            else:
                output_delta[i] = self.sigmoid(self.outputs[i], True) * err
        # print(f"output_delta[i] = " + str(output_delta))

        # Computing Hidden Delta
        for i in range(self.number_of_hidden_units):
            error = 0.0
            for j in range(self.number_of_outputs-1):
                error += output_delta[j] * self.weights_lower[i][j]
                # print(f"output delta: {output_delta[j]}, hidden_neuron: {self.hidden_neurons[i]}")
                self.weight_changes_lower[i][j] = output_delta[j] * self.hidden_neurons[i]
        if use_sigmoid:
            hidden_delta[i] = self.sigmoid(self.activations_lower[i], True) * err
        else:
            hidden_delta[i] = self.sigmoid(self.activations_lower[i], True) * err

        for i in range(self.number_of_inputs):
            for j in range(self.number_of_hidden_units):
                self.weight_changes_upper[i][j] = hidden_delta[j] * self.input_neurons[i]
        
        return np.mean(np.abs(np.subtract(target, self.outputs)))
        

            



    def update_weights(self, learning_rate):
        print("awadwa")
        for i in range(self.number_of_hidden_units):
            for j in range(self.number_of_outputs):
                self.weights_lower[i][j] += self.weight_changes_lower[i][j] * learning_rate

        for i in range(self.number_of_hidden_units):
            for j in range(self.number_of_inputs):
                self.weights_upper[i][j] += self.weight_changes_upper[i][j] * learning_rate

        self.weight_changes_upper = [[0]*self.number_of_hidden_units]*self.number_of_inputs
        self.weight_changes_lower = [[0]*self.number_of_hidden_units]*self.number_of_outputs

    def sigmoid(self, sig, is_backward=False):
        if is_backward:
            return (1.0 / 1.0 + math.exp(-sig)) * (1-(1.0 / 1.0 + math.exp(-sig)))
        else:
            return 1.0 / 1.0 + math.exp(-sig)

    def hyperbolic_tangent(self, tanh, is_backward=False):
        if is_backward:
            return 1 - (np.power(self.hyperbolic_tangent(tanh), 2))
        else:
            return (2 / (1 + np.exp(tanh * - 2))) - 1
                
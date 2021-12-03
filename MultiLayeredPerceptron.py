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
        self.weights_upper = [[0]*self.number_of_outputs]*self.number_of_hidden_units
        self.weights_lower = [[0]*self.number_of_hidden_units]*self.number_of_inputs

        # Also sets weight change arrays to 0
        self.weight_changes_upper = [[0]*self.number_of_outputs]*self.number_of_hidden_units
        self.weight_changes_lower = [[0]*self.number_of_hidden_units]*self.number_of_inputs

    def randomize(self):
        rand.seed()
        
        # Initializing weights to random values
        for i in range(self.number_of_hidden_units):
            for j in range(self.number_of_outputs):
                self.weights_upper[i][j] = rand.uniform(0.0, 1.0)

        # Initializing weights to random values
        for i in range(self.number_of_inputs):
            for j in range(self.number_of_hidden_units):
                self.weights_lower[i][j] = rand.uniform(0.0, 1.0)

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


    def forwardOLD(self, Input, sigmoid):
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

        return np.mean(np.abs(np.subtract(target, self.outputs)))
    
    def update_weights(self, learning_rate):
        for i in range(self.number_of_inputs):
            for j in range(self.number_of_hidden_units):
                self.weights_lower[i][j] += self.weight_changes_lower[i][j] * learning_rate

        for i in range(self.number_of_hidden_units):
            for j in range(self.number_of_outputs):
                self.weights_upper[i][j] += self.weight_changes_upper[i][j] * learning_rate

        self.weight_changes_upper = [[0]*self.number_of_outputs]*self.number_of_hidden_units
        self.weight_changes_lower = [[0]*self.number_of_hidden_units]*self.number_of_inputs

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
                
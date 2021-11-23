import numpy as np
import random as rand


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


        # Initialize weight arrays and weight_change arrays to correct size
        self.weights_upper = [[0]*self.number_of_hidden_units]*self.number_of_inputs
        self.weights_lower = [[0]*self.number_of_hidden_units]*self.number_of_inputs

        # Also sets weight change arrays to 0
        self.weight_changes_lower = [[0]*self.number_of_hidden_units]*self.number_of_inputs
        self.weight_changes_upper = [[0]*self.number_of_hidden_units]*self.number_of_inputs


    def randomize(self):
        rand.seed()
        
        # Initializing weights to random values
        for i in range(self.number_of_inputs):
            for j in range(self.number_of_hidden_units):
                self.weights_lower[i][j] = rand.uniform(0.0, 1.0)
                self.weights_upper[i][j] = rand.uniform(0.0, 1.0)
        # print(self.weights_lower)
        # print(self.weights_upper)

    def forward(self, Input, sigmoid):
        for i in range(self.number_of_inputs - 1):
            self.input_neurons[i] = Input[i]

        for i in range(self.number_of_hidden_units):
            hidden = 0.0
            for j in range(self.number_of_inputs):
                hidden += self.input_neurons[j] * self.weights_upper[j][i]
            if sigmoid == True:
                hidden = self.sigmoid(hidden)
            else:
                hidden = self.hyperbolic_tangent(hidden)
            self.hidden_neurons[i] = hidden

        for i in range(self.number_of_outputs):
            output_sum = 0.0
            for j in range(self.number_of_hidden_units):
                output_sum += self.hidden_neurons[j] * self.outputs[j][i]
            
            if sigmoid == True:
                output_sum = self.sigmoid(output_sum)
            else:
                output_sum = self.hyperbolic_tangent(output_sum)
            self.outputs[i] = output_sum

        # self.activations_lower = np.dot(I, self.weights_lower)
        # if sin:
        #     self.hidden_neurons = self.hyperbolic_tangent(self.activations_lower)
        # else:
        #     self.hidden_neurons = self.sigmoid(self.activations_lower)
        
        # self.activations_upper = np.dot(self.hidden_neurons, self.weights_upper)

        # if sin:
        #     self.outputs = self.hyperbolic_tangent(self.activations_upper)
        # else:
        #     self.outputs = self.sigmoid(self.activations_upper)
    
    def backwards(self, T, target, sin):
        err = np.subtract(target, self.outputs)
        
        for i in range(self.outputs):
            err[i] = target[i] - self.outputs[i]

        if sin:
            activation_lower_layer = self.hyperbolic_tangent(self.activations_lower)
            activation_upper_layer = self.hyperbolic_tangent(self.activations_upper)
        else:
            activation_lower_layer = self.sigmoid(self.activations_lower)
            activation_upper_layer = self.sigmoid(self.activations_upper)

        weight_changes_upper = np.multiply(err, activation_upper_layer)
        self.weight_changes_upper = np.dot(self.hidden_neurons, weight_changes_upper)

        weight_changes_lower = np.multiply(np.dot(weight_changes_upper, self.weights_upper), activation_lower_layer)
        self.weight_changes_lower = np.dot(T, weight_changes_lower)

        return np.mean(np.abs(err))
    
    def update_weights(self, learning_rate):
        self.weights_lower = np.add(self.weights_lower, learning_rate * self.weight_changes_lower)
        self.weights_upper = np.add(self.weights_upper, learning_rate * self.weight_changes_upper)
        self.weight_changes_lower = np.array
        self.weight_changes_upper = np.array


    def sigmoid(self, sig, is_backward=False):
        if is_backward:
            return np.exp(-sig) / (1 + np.exp(-sig)) ** 2
        else:
            return 1 / (1 + np.exp(-sig))

    def hyperbolic_tangent(self, tanh, is_backward=False):
        if is_backward:
            return 1 - (np.power(self.hyperbolic_tangent(tanh), 2))
        else:
            return (2 / (1 + np.exp(tanh * - 2))) - 1
                
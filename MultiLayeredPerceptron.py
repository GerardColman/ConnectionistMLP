import numpy as np


class MLP:

    number_of_inputs = 0
    number_of_hidden_units = 0
    number_of_outputs = 0
    weights_lower = np.array # W1
    weights_upper = np.array # W2
    weight_changes_lower = np.array # dw1
    weight_changes_upper = np.array # dw2
    activations_lower = np.array # z1
    activations_upper = np.array # z2
    hidden_neurons = np.array # h
    outputs = np.array # o

    def __init__(self, num_inputs, num_hidden, num_output):
        self.number_of_inputs = num_inputs
        self.number_of_hidden_units = num_hidden
        self.number_of_outputs = num_output

        # Setting weight change arrays to 0
        self.weight_changes_lower = np.full((self.number_of_inputs, self.number_of_hidden_units), 0)
        self.weight_changes_upper = np.full((self.number_of_inputs, self.number_of_hidden_units), 0)

    def randomize(self):

        # Initializing weights to random values
        self.weights_lower = np.array((np.random.uniform(0.0, 1, (self.number_of_inputs, self.number_of_hidden_units))))
        self.weights_upper = np.array((np.random.uniform(0.0, 1, (self.number_of_inputs, self.number_of_hidden_units))))

    def forward(self, I, sin):
        self.activations_lower = np.dot(I, self.weights_lower)
        if sin:
            self.hidden_neurons = self.hyperbolic_tangent(self.activations_lower)
        else:
            self.hidden_neurons = self.sigmoid(self.activations_lower)
        
        self.activations_upper = np.dot(self.hidden_neurons, self.weights_upper)
        if sin:
            self.outputs = self.hyperbolic_tangent(self.activations_upper)
        else:
            self.outputs = self.sigmoid(self.activations_upper)
    
    def backwards(self, T, target, sin):
        err = np.subtract(target, self.outputs)

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
    
    def updateWeights(self, learning_rate):
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
            return (2 / (1 + np.exp(tanh * -2))) - 1
            
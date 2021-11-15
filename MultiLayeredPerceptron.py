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

    def sigmoid()

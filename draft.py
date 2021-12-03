import MultiLayeredPerceptron as MLP

NN = MLP(2, 4, 1)
NN.randomize()

# Declaring input, output, max_epochs, and learning rate
XOR_inputs = [[0, 0], [0, 1], [1, 0], [1, 1]]
XOR_desired_output = [[0], [1], [1], [0]]
max_epochs = 100
learning_rate = 1

NN.fo
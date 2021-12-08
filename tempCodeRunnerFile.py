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
# Testing
print("==========TESTING==========")
for i in range(len(XOR_inputs)):
    NN.forward(XOR_inputs[i], True)
    print(f"Target: {XOR_desired_output[i]} Output: {NN.outputs}")
'''
pip install numpy
'''
import numpy as np

# Input data and normalization
X = np.array(([2, 9], [1, 5], [3, 6]), dtype=float)
y = np.array(([92], [86], [89]), dtype=float)
X = X / np.amax(X, axis=0)  # normalize X by its max value in each column
y = y / 100  # normalize y to range between 0 and 1

# Sigmoid function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Derivative of the sigmoid function
def derivatives_sigmoid(x):
    return x * (1 - x)

# Variable initialization
epoch = 7000       # number of training iterations
lr = 0.1           # learning rate
inputlayer_neurons = 2  # number of features in dataset
hiddenlayer_neurons = 3 # number of hidden layer neurons
output_neurons = 1      # number of neurons in output layer

# Weight and bias initialization
wh = np.random.uniform(size=(inputlayer_neurons, hiddenlayer_neurons))
bh = np.random.uniform(size=(1, hiddenlayer_neurons))
wout = np.random.uniform(size=(hiddenlayer_neurons, output_neurons))
bout = np.random.uniform(size=(1, output_neurons))

# Training loop
for i in range(epoch):
    # Forward propagation
    hinp1 = np.dot(X, wh)
    hinp = hinp1 + bh
    hlayer_act = sigmoid(hinp)

    outinp1 = np.dot(hlayer_act, wout)
    outinp = outinp1 + bout
    output = sigmoid(outinp)

    # Backpropagation
    EO = y - output           # error at the output
    outgrad = derivatives_sigmoid(output)  # gradient at output layer
    d_output = EO * outgrad

    EH = d_output.dot(wout.T)               # error at hidden layer
    hiddengrad = derivatives_sigmoid(hlayer_act)  # gradient at hidden layer
    d_hiddenlayer = EH * hiddengrad

    # Updating weights and biases
    wout += hlayer_act.T.dot(d_output) * lr
    bout += np.sum(d_output, axis=0, keepdims=True) * lr
    wh += X.T.dot(d_hiddenlayer) * lr
    bh += np.sum(d_hiddenlayer, axis=0, keepdims=True) * lr

# Display results
print("Input: \n" + str(X))
print("Actual Output: \n" + str(y))
print("Predicted Output: \n" + str(output))

# This program implements a simple feedforward neural network with one hidden layer to predict output values from normalized input data using forward propagation and backpropagation for training.
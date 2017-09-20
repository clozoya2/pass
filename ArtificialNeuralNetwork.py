"""
Demo NN program from https://youtu.be/h3l4qz76JhQ.
Creates an neural network that simulates the exclusive OR function with two inputs and one output.
"""


import numpy as np


def sigmoid(x, derivative=False):
    if (derivative == True):
        return (x * (1 - x))

    return 1 / (1 + np.exp(-x))


def train(iterations, synapses, inputData, outputData):
    synapse0 = synapses[0]
    synapse1 = synapses[1]
    layer2 = None
    for j in range(iterations+1):

        # Calculate forward through the network.
        layer0 = inputData
        layer1 = sigmoid(np.dot(layer0, synapse0))
        layer2 = sigmoid(np.dot(layer1, synapse1))

        # Back propagation of errors using the chain rule.
        layer2_error = outputData - layer2
        if (j % (iterations / (10 ** (len(str(iterations)) - 3)))) == 0 or j == iterations:
            print("Iteration " + str(j) + " error: " + str(np.mean(np.abs(layer2_error))))

        layer2_delta = layer2_error * sigmoid(layer2, derivative=True)
        layer1_error = layer2_delta.dot(synapse1.T)
        layer1_delta = layer1_error * sigmoid(layer1, derivative=True)

        # update weights (no learning rate term)
        synapse1 += layer1.T.dot(layer2_delta)
        synapse0 += layer0.T.dot(layer1_delta)

    return layer2


# Third input column accommodates bias term
inputData = np.array([[0, 0, 1],
              [0, 1, 1],
              [1, 0, 1],
              [1, 1, 1]])

outputData = np.array([[0],
              [1],
              [1],
              [0]])


"""
Randomly generated synapses
synapse0 between input and hidden layer (3x4 matrix: 2 inputs, 1 bias, 1 hidden layer)
synapse1 between hidden and output layer (4x1 matrix: 4 nodes in hidden layer, one output
"""
synapse0 = 2 * np.random.random((3, 4)) - 1
synapse1 = 2 * np.random.random(
    (4, 1)) - 1


print("Prediction: " + str(train(100000, (synapse0, synapse1), inputData, outputData)))

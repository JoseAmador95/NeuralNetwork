"""
**Multiple-Layer Perceptron Neural Network**

A library to implement an MLP Neural Network with flexible architecture. The activation
function is the Sigmoid function, that adds a layer of subtle non-linearity in comparison
to the perceptron network. The class uses backpropagation and Stochastic Gradient Descend
to train the network an random initial weights and biases

The file also includes some helper functions like:
* The Sigmoid function and its derivative
* Output translator for the handwritten network example
* Save/Load features to use the trained network in other files

This file is based on the work in http://neuralnetworksanddeeplearning.com/ by Michael
Nielsen, and the hadwritten numbers data set was obtained from his Github repository.
"""

import random
import numpy as np
import pickle


def sigmoid(x):
    """ Sigmoid function for neuron's activation level """
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    """ Derivative of the sigmoid function"""
    return sigmoid(x) * (1 - sigmoid(x))


def outputLayer2number(x):
    """ Converts the output of the NN for classifying handwritten numbers into a single
        digit. The output will be a number from 0 to 9 if the activation level of the
        ouput layer is above 0.5. If not confident enough, the function will output None."""
    if np.max(x) > 0.5:
        return np.argmax(x)
    else:
        return None

def saveNetwork(NN, filename = None):
    """ Saves the NN parameters (Layers, weights and biases) into a file for later use"""
    if not filename or type(filename) != str:
        filename = 'parameters.pkl'
    with open(filename, 'wb') as f:
        pickle.dump([NN.layers, NN.weights, NN.biases], f)

def loadNetwork(filename):
    """ Restores a NN object by importing the layer architecture, weights and biases
        from a file"""
    with open(filename, 'rb') as f:
        layers, weights, biases = pickle.load(f)
    NN = NeuralNetwork(layers)
    NN.weights = weights
    NN.biases = biases
    return NN

def importDataset():
    """ Loads the sample dataset from obtained from the authors Github repo and returns
        it in a tuple containing the training, testing and validation sets."""
    import os
    os.chdir('DeepLearningPython35')
    print(os.getcwd())
    import DeepLearningPython35.mnist_loader as mnist_loader
    training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
    os.chdir('..')
    return (training_data, validation_data, test_data)

class NeuralNetwork:
    def __init__(self, layers):
        """ Class constructor that initializes the layers with the correspondent neuron
            numbers. The argument layers must be a list containing the number of neurons
            for each layer, being the number of elements the number of layers in the NN."""
        self.n_layers = len(layers)
        self.layers = layers
        self.biases = [np.random.randn(y, 1) for y in layers[1:]]
        self.weights = [np.random.randn(y, x) for x, y in zip(layers[:-1], layers[1:])]
        pass

    def predict(self, x):
        """ Returns the output of the NN using the current weights and biases """
        for b, w in zip(self.biases, self.weights):
            x = sigmoid(np.dot(w, x) + b)
        return x

    def SGD(self, data, epochs, batch_size, eta, test_data=None):
        """ Trains the NN using the Stochastic Gradient Descend algorithm. The training
            data is split into multiple batches consisting on several tuples containing
            (input, label) elements. The batch size, number of iterations and learning
            rate (eta) are given as function arguments. Test data may be provided to
            track progress."""
        data = list(data)
        if test_data:
            test_data = list(test_data)
            n_test = len(test_data)
        n_training = len(data)
        for i in range(epochs):
            random.shuffle(data)
            batches = [
                data[j:j + batch_size]
                for j in range(0, n_training, batch_size)
            ]
            for batch in batches:
                self.update_batch(batch, eta)
            if test_data:
                print('Epoch {0}: {1} / {2}'.format(
                    i, self.evaluate(test_data), n_test
                ))
            else:
                print('Epoch {0} complete!'.format(i))

    def update_batch(self, batch, eta):
        """ Updates the weights and biases by using the backpropagation algorithm to
            estimate the gradient vector. The learning rate can be adjusted by using
            the eta parameter"""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        for x, y in batch:
            delta_nabla_b, delta_nabla_w = self.backpropagation(x, y)
            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
            self.weights = [w - eta / len(batch) * nw for w, nw in zip(self.weights, nabla_w)]
            self.biases = [b - eta / len(batch) * nb for b, nb in zip(self.biases, nabla_b)]

    def evaluate(self, test_data):
        """ Returns the number of correct predictions by the NN using the test data set """
        test_results = [(np.argmax(self.predict(x)), y) for x, y in test_data]
        return sum(int(x == y) for x, y in test_results)

    def backpropagation(self, x, y):
        """ Executes the backpropagation algorithm that returns the gradient vector of
            the error cost function. Arguments 'x' and 'y' are the input and output of
            the NN, respectively."""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        activation = x
        activations = [activation]
        zs = []
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation) + b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)

        delta = self.cost_derivative(activations[-1], y) * sigmoid_derivative(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].T)

        for l in range(2, self.n_layers):
            z = zs[-l]
            sp = sigmoid_derivative(z)
            delta = np.dot(self.weights[-l + 1].T, delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l - 1].T)

        return (nabla_b, nabla_w)

    def cost_derivative(self, activations, y):
        """ Returns the derivative of the cost function given the activation level of the
            output layer and the true label"""
        return (activations - y)


if __name__ == '__main__':
    """ Handwritten classifier example using the data set from the Github repo described above, which
        contains the grayscale value of every pixel of 10K 27x27 images and its corresponding label. 
        These images are the scan of a handwritten number from 0 to 9."""

    # Import the data set
    training_data, validation_data, test_data = importDataset()

    # Define the Network architecture (Input layer, Hidden Layer(s), Output Layer)
    # (27x27 pixels input layer, single hidden layer of 30 neurons, and a 10-element output layer)
    NN = NeuralNetwork([784, 30, 10])

    # Print the Network description
    print('Number of layers: %d\nNeurons per layer: %s\nBiases:\n%s\nWeights:\n%s\n' %
          (NN.n_layers, str(NN.layers), str([b.shape for b in NN.biases]), str([w.shape for w in NN.weights])))

    # Train the network using Stochastic Gradient Descend method
    NN.SGD(training_data, 10, 10, 3, test_data=test_data)

    # Save the network parameters to be used in another file
    saveNetwork(NN)

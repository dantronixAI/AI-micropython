from .numpy_package import numpy as np
from .functions import sigmoid
from .optimizations import rmsprop
import random
class FFN:
    def __init__(self, params):
        shape = params['shape']

        # Layer info
        self.numLayers = len(shape) - 1
        self.shape = shape
        self.activation={}
        self.optimization={}

        if 'activation' in params.keys():
            activationFunction = params['activation']
            for layer in range(self.numLayers):
                self.activation[layer] = activationFunction['layer'+str(layer+1)]
        else:
            for layer in range(self.numLayers):
                self.activation[layer] = sigmoid

        if 'optimization' in params.keys():
            optimizer = params['optimization']
            for layer in range(self.numLayers):
                self.optimization[layer] = optimizer.get_layer()
        else:
            for layer in range(self.numLayers):
                self.optimization[layer] = rmsprop().get_layer()

        # Input/Output data from last run
        self._layerInput = []
        self._layerOutput = []
        # Initialize the weights
        self.weights = []
        for (l1, l2) in zip(shape[:-1], shape[1:]):
            # Ideally:
            # self.weights.append((np.random.random(size=(l2, l1 + 1))-0.5)*2)
            values = np.zeros((l2, l1 + 1))
            for row in range(l2):
                for col in range(l1+1):
                    values[row][col]=(random.random()-0.5)*2
            self.weights.append(values)



    def forwardProp(self, input):
        numSamples = input.shape[0]

        self._layerInput = []
        self._layerOutput = []
        for n_layer in range(self.numLayers):
            # Get input to the layer
            if n_layer == 0:
                numNeurons = input.T.shape[0]
                M = np.ones((numNeurons+1,numSamples))
                M[:-1,:] = input.T
            else:
                numNeurons = self._layerOutput[-1].shape[0]
                M = np.ones((numNeurons+1,self._layerOutput[- 1].shape[1]))
                M[:-1,:] = self._layerOutput[-1]
            layerInput = np.dot(self.weights[n_layer],M)

            self._layerInput.append(layerInput)
            self._layerOutput.append(self.activation[n_layer](layerInput))

        return self._layerOutput[-1].T

    def eval(self, input):
        numSamples = input.shape[0]
        for n_layer in range(self.numLayers):
            if n_layer == 0:
                numNeurons = input.T.shape[0]
                M = np.ones((numNeurons+1,numSamples))
                M[:-1,:] = input.T
            else:
                numNeurons = layerOutput.shape[0]
                M = np.ones((numNeurons+1,layerOutput.shape[1]))
                M[:-1,:] = layerOutput
            layerInput = np.dot(self.weights[n_layer],M)
            layerOutput = self.activation[n_layer](layerInput)

        return  layerOutput.T
    def backwardProp(self, input, target, learningRate=0.2):
        delta = []
        numSamples = input.shape[0]

        # First run the network
        self.forwardProp(input)

        # Calculate the deltas for each node
        for n_layer in reversed(range(self.numLayers)):
            if n_layer == self.numLayers - 1:
                # If the output layer, then compare to the target values
                output_delta = self._layerOutput[n_layer] - target.T
                error = np.sum(output_delta ** 2)
                delta.append(output_delta * self.activation[n_layer](self._layerInput[n_layer], True))
            else:
                # If a hidden layer. compare to the following layer's delta
                delta_pullback = np.dot(self.weights[n_layer + 1].T,delta[-1])
                delta.append(delta_pullback[:-1, :] * self.activation[n_layer](self._layerInput[n_layer], True))

        # Compute updates to each weight
        for n_layer in range(self.numLayers):
            delta_n_layer = self.numLayers - 1 - n_layer

            if n_layer == 0:
                numNeurons = input.T.shape[0]
                layerOutput = np.ones((numNeurons + 1, numSamples))
                layerOutput[:-1, :] = input.T
            else:
                numNeurons = self._layerOutput[n_layer - 1].shape[0]
                layerOutput = np.ones((numNeurons + 1, self._layerOutput[n_layer - 1].shape[1]))
                layerOutput[:-1, :] = self._layerOutput[n_layer-1]

            currentWeightDelta = np.dot(layerOutput, delta[delta_n_layer].T).T
            self.weights[n_layer] = self.optimization[n_layer].optimize(currentWeightDelta)
        return error

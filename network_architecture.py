import numpy as np
import random

class Network:
    def __init__(self, sizes):
        self.numberOfLayers = len(sizes)
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y,x) for x,y in zip(sizes[:-1], sizes[1:])]
    
    def feedForward(self, a):
        for bias, weight in zip(self.biases, self.weights):
            a = sigmoid(np.dot(weight, a) + bias)
        return a
    
    def SGD(self, trainingData, epochs, batchSize, eta, testData=None):
        if testData:
            testDataSize = len(testData)

        trainingDataSize = len(trainingData)
        for i in range(epochs):
            random.shuffle(trainingData)
            miniBatches = [trainingData[k:k+batchSize] for k in range(0, trainingDataSize, batchSize)]
            for miniBatch in miniBatches:
                self.updateMiniBatch(miniBatch, eta)
            
            if testData:
                print("Epoch {0}: {1} / {2}".format(i, self.evaluate(testData), testDataSize))
    
    def updateMiniBatch(self, miniBatch, eta):
        nabla_b = [np.zeros(bias.shape) for bias in self.biases]
        nabla_w = [np.zeros(weight.shape) for weight in self.weights]
        
        for x, y in miniBatch:
            delta_nabla_b, delta_nabla_w = self.backPropagation(x, y) 
            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        
        self.weights = [w - (eta/len(miniBatch))*nw for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b - (eta/len(miniBatch))*nb for b, nb in zip(self.biases, nabla_b)]
    
    def backPropagation(self, x, y):
        nabla_b = [np.zeros(bias.shape) for bias in self.biases]
        nabla_w = [np.zeros(weight.shape) for weight in self.weights]
        zs = []
        activation = x;
        activations = [x]
        
        for bias, weight in zip(self.biases, self.weights):
            z = np.dot(weight, activation)+bias
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        
        delta = self.costDerivative(activations[-1], y) * sigmoidPrime(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        
        for l in range(2, self.numberOfLayers):
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sigmoidPrime(zs[-l])
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        return(nabla_b, nabla_w)
    
    def costDerivative(self, outputActivations, y):
        return (outputActivations - y)
    
    def evaluate(self, testData):
        testResults = [(np.argmax(self.feedForward(x)), y) for (x,y) in testData]
        return sum(int(yHat == y) for (yHat, y) in testResults)

def sigmoid(z):
    return 1.0/(1 + np.exp(-z))

def sigmoidPrime(z):
    return sigmoid(z)*(1-sigmoid(z))


import math
import numpy as np
import random


def read_input(inputfile):
    with open(inputfile, "r") as f:
        nr_features = int(f.readline()) - 1
        n = int(f.readline())
        date = []
        for linie in f.readlines():
            linie = linie.split(',')
            date.append([float(ex) for ex in linie])
        return nr_features, n, date


class Neuron:
    def __init__(self, w=[], out=None, delta=0.0):
        self.w = w
        self.out = out
        self.delta = delta


def netInit(nrInputs, nrOutputs, nrHidden):
    net = []
    hiddenLayer = [Neuron([random.random() for i in range(nrInputs + 1)]) for h in range(nrHidden)]
    net.append(hiddenLayer)
    outputLayer = [Neuron([random.random() for i in range(nrHidden + 1)]) for o in range(nrOutputs)]
    net.append(outputLayer)
    return net


def calculate(x, w):
    result = 0.0
    for i in range(len(x)):
        result += x[i] * w[i]
        result += w[len(x)]
    return result


def activation(x, inverse=False):
    if inverse == False:
        return x
    return x


def deactivation(x):
    return math.log(x / (1 - x))


def forward(net, inputs):
    for layer in net:
        newInputs = []
        for neuron in layer:
            result = calculate(inputs, neuron.w)
            neuron.out = activation(result)
            newInputs.append(neuron.out)
        inputs = newInputs
    return inputs


def back(net, expected):
    for i in range(len(net) - 1, 0, -1):
        layer = net[i]
        errors = []
        if i == (len(net) - 1):
            for j in range(len(layer)):
                neuron = layer[j]
                errors.append(expected[j] - neuron.out)
        else:
            for j in range(len(layer)):
                error = 0.0
                nextLayer = net[i + 1]
                for neuron in nextLayer:
                    error += neuron.w[j] * neuron.delta
                errors.append(error)
        for j in range(len(layer)):
            layer[j].delta = errors[j] * activation(layer[j].out, inverse=True)


def computeError(computedOutputs, realOutputs):
    error = sum([(computedOutputs[i] - realOutputs[i]) ** 2 for i in range(len(computedOutputs))])
    return error


def updateWeights(net, example, learningRate):
    for i in range(len(net)):
        inputs = example[:-1]
        if (i > 0):
            inputs = [neuron.out for neuron in net[i - 1]]
        for neuron in net[i]:
            for j in range(len(inputs)):
                neuron.w[j] += learningRate * neuron.delta * inputs[j]
            neuron.w[-1] += learningRate * neuron.delta


def train(net, data, learningRate, epoci):
    for epoch in range(epoci):
        sumError = 0.0
        for example in data:
            computedOutputs = forward(net, example[:-1])
            expected = [example[-1]]
            error = sum([(expected[i] - computedOutputs[i]) ** 2 for i in range(len(expected))])
            sumError += error
            back(net, expected)
            updateWeights(net, example, learningRate)
        print(sumError)


def test(net, data):
    computedOutputs = []
    for example in data:
        computedOutput = forward(net, example[:-1])
        computedOutputs.append(computedOutput[0])
    return computedOutputs


def run(trainData, testData, learningRate, noEpochs):
    nrInputs = len(trainData[0]) - 1

    nrOutputs = 1
    net = netInit(nrInputs, nrOutputs, 15)

    train(net, trainData, learningRate, noEpochs)
    realOutputs = [(trainData[i][j]) for j in range(len(trainData[0]) - 1, len(trainData[0])) for i in range(len(trainData))]
    computedOutputs = test(net, trainData)
    print("train SRE: ", computeError(computedOutputs, realOutputs) / len(trainData))

    realOutputs = [(testData[i][j]) for j in range(len(testData[0]) - 1, len(testData[0])) for i in range(len(testData))]
    computedOutputs = test(net, testData)
    print("test SRE: ", computeError(realOutputs, [(x) for x in computedOutputs]) / len(testData))

trainfile = "hard_parkinson_train"
testfile = "hard_parkinson_test"

_, _, date_train = read_input(trainfile)
_, _, date_test = read_input(testfile)

run(date_train, date_test, 0.00000000001, 30)

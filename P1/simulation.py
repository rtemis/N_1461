import os
import random

from neural_network import NeuralNetwork, Layer
from neuron import Neuron, Connection

class Data():

    def __init__(self):
        self.train_in = []
        self.train_out = []
        self.test_in = []
        self.test_out = []

        self.attributes = 0
        self.classes = 0        
        
        self.attributes_test = 0
        self.classes_test = 0

    def load_data_proportional(self, filename, prop=0.7):

        file = open(filename, "r")
        data = file.readlines()
        file.close()

        config = data[0].split(' ')
        self.attributes = config[0]
        self.classes = config[1]
        
        data = data[1:]

        random.shuffle(data)

        length = len(data)
        threshold = length * prop

        for i in range(threshold):
            self.train_in.append(data[i][0:attributes])
            self.train_out.append(data[i][attributes:])
        
        for i in range(threshold, length):
            self.test_in.append(data[i][0:attributes])
            self.test_out.append(data[i][attributes:])

        return self.train_in, self.train_out, self.test_in, self.test_out

    def load_data_file(self, filename):

        file = open(filename, "r")
        data = file.readlines()
        file.close()

        config = data[0].split(' ')
        self.attributes = config[0]
        self.classes = config[1]
        
        data = data[1:]

        for line in data:
            self.train_in.append(line[0:attributes])
            self.test_in.append(line[0:attributes])
            self.train_out.append(line[attributes:])
            self.test_out.append(line[attributes:])

        return self.train_in, self.train_out

    def load_data_files(self, file_train, file_test):
        file = open(file_train, "r")
        data1 = file.readlines()
        file.close()
        
        config = data1[0].split(' ')
        self.attributes = config[0]
        self.classes = config[1]
        
        data1 = data1[1:]

        for i in data1:
            self.train_in.append(data1[i][0:attributes])
            self.train_out.append(data1[i][attributes:])
        
        file = open(file_train, "r")
        data2 = file.readlines()
        file.close()
        
        config = data2[0].split(' ')
        self.attributes_test = config[0]
        self.classes_test = config[1]

        for i in data2
            self.test_in.append(data2[i][0:attributes])
            self.test_out.append(data2[i][attributes:])

        return self.train_in, self.train_out, self.test_in, self.test_out



def simulation_test():
    x_1 = Neuron(0, "Directa")
    x_2 = Neuron(0, "Directa")
    x_3 = Neuron(0, "Directa")

    a12 = Neuron(2, "McCulloch", active=1, inactive=0)
    a13 = Neuron(2, "McCulloch", active=1, inactive=0)
    a23 = Neuron(2, "McCulloch", active=1, inactive=0)

    y = Neuron(2, "McCulloch", active=1, inactive=0)

    x_1.connect(a12, 1)
    x_2.connect(a12, 1)

    x_1.connect(a13, 1)
    x_3.connect(a13, 1)

    x_2.connect(a23, 1)
    x_3.connect(a23, 1)

    a12.connect(y, 2)
    a13.connect(y, 2)
    a23.connect(y, 2)

    layer_and = Layer()
    layer_and.add_neuron(x_1)
    layer_and.add_neuron(x_2)
    layer_and.add_neuron(x_3)

    layer_or = Layer()
    layer_or.add_neuron(a12)
    layer_or.add_neuron(a13)
    layer_or.add_neuron(a23)

    layer_f = Layer()
    layer_f.add_neuron(y)

    network = NeuralNetwork()
    network.add_layer(layer_and)
    network.add_layer(layer_or)
    network.add_layer(layer_f)

    data = [[0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1], [1, 0, 0], [1, 0, 1]]

    for d in data:
        x_1.initialize(d[0])
        x_2.initialize(d[1])
        x_3.initialize(d[2])

        network.fire()
        network.initialize()
        network.propagate()
        print(d, a12.f_x, a13.f_x, a23.f_x, y.f_x)

    network.initialize()
    network.fire()
    network.initialize()
    network.propagate()
    print(d, a12.f_x, a13.f_x, a23.f_x, y.f_x)

    network.initialize()
    network.fire()
    network.initialize()
    network.propagate()
    print(d, a12.f_x, a13.f_x, a23.f_x, y.f_x)

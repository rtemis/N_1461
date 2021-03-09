import os
import random

from neural_network import NeuralNetwork, Layer
from neuron import Neuron, Connection
from perceptron import perceptron
from adaline import adaline

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

        config = data[0].split()
        self.attributes = int(config[0])
        self.classes = int(config[1])
        
        data = data[1:]
        #random.shuffle(data)

        ls_in = []
        ls_out = []
        for i in data:
            line = [float(x) for x in i.split()]
            ls_in.append(line[:self.attributes])
            ls_out.append(line[self.attributes:])

        length = len(data)
        threshold = int(length * prop)

        self.train_in = ls_in[:threshold]
        self.train_out = ls_out[:threshold]      
        
        self.test_in = ls_in[threshold:]
        self.test_out = ls_out[threshold:]

        return self.train_in, self.train_out, self.test_in, self.test_out

    def load_data_file(self, filename):

        file = open(filename, "r")
        data = file.readlines()
        file.close()

        config = data[0].split()
        self.attributes = int(config[0])
        self.classes = int(config[1])
        
        data = data[1:]
        #random.shuffle(data)

        for i in data:
            line = [float(x) for x in i.split()]
            self.train_in.append(line[:self.attributes])
            self.train_out.append(line[self.attributes:])

        return self.train_in, self.train_out

    def load_data_files(self, file_train, file_test):
        file = open(file_train, "r")
        data = file.readlines()
        file.close()
        
        config = data[0].split()
        self.attributes = config[0]
        self.classes = config[1]
        
        data = data[1:]
        random.shuffle(data)

        for i in data:
            line = [float(x) for x in i.split()]
            self.train_in.append(line[:self.attributes])
            self.train_out.append(line[self.attributes:])
        
        file = open(file_train, "r")
        data = file.readlines()
        file.close()
        
        config = data[0].split()
        if self.attributes != config[0] or self.classes != config[1]:
            raise Exception('Files of different sizes')

        data = data[1:]
        random.shuffle(data)

        for i in data:
            line = [float(x) for x in i.split()]
            self.test_in.append(line[:self.attributes])
            self.test_out.append(line[self.attributes:])

        return self.train_in, self.train_out, self.test_in, self.test_out



def mcculloch():
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


def main():
    data = Data()

    tr_in, tr_out = data.load_data_file('data/problema_real1.txt')

    net = perceptron(tr_in, tr_out, tr_in, tr_out, data.attributes, data.classes)

    net.print_network()

main()
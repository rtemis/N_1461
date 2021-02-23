import os
import random

from neural_network import NeuralNetwork, Layer
from neuron import Neuron, Connection

class Data():

    def __init__(self):
        self.train = []
        self.test = []

    def load_data_proportional(self, filename, prop=0.7):

        file = open(filename, "r")
        data = file.readlines()
        file.close()

        random.shuffle(data)

        length = len(data)
        threshold = length * prop

        for i in range(threshold):
            self.train.append(data[i])
        
        for i in range(threshold, length):
            self.test.append(data[i])
            
        return 

    def load_data_file(self, filename):
        pass

    def load_data_files(self, file_train, file_test):
        pass



def main():
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

    x_1.initialize(1)
    x_2.initialize(1)
    x_3.initialize(1)

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

    network.fire()
    print(a12.value)
    print(a13.value)
    print(a23.value)

    print(y.f_x)
    print(y.value)

if __name__ == "__main__":
    main()



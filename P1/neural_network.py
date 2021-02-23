###############################################################################
# FILE: neural_network.py
# AUTHORS: LEAH HADEED & EVA GUTIERREZ
# 
# BRIEF:
# Contains the class for the creation of the neural network. Abstract class?

class NeuralNetwork():

    def __init__(self):
        self.layers = []
        
    def free(self):
        pass

    def initialize(self):
        pass

    def add_layer(self, layer):
        self.layers.append(layer)         

    def fire(self):
        for layer in self.layers:
            layer.fire()

    def propagate(self):
        for layer in self.layers:
            layer.propagate()
        


class Layer():
    
    def __init__(self):
        self.neurons = []
        
    def free(self):
        pass

    def initialize(self):
        pass

    def add_neuron(self, neuron):
        self.neurons.append(neuron)
        
    def connect(self, weight, layer=None, neuron=None):
        pass 

    def fire(self):
        for neuron in self.neurons:
            neuron.fire()
        self.propagate()

    def propagate(self):
        for neuron in self.neurons:
            neuron.propagate()
        return 


class Perceptron(NeuralNetwork):
    def __init__(self):
        pass
    def back_propagation(self):
        # if prev weight == weight, stop
        # else propagate
        pass


class Adaline(NeuralNetwork):
    def __init__(self):
        pass
    def back_propagation(self):
        # if prev weight == weight, stop
        # else propagate
        pass
    
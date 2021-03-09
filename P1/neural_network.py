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
        for layer in self.layers:
            layer.initialize()

    def add_layer(self, layer):
        self.layers.append(layer)         

    def fire(self):
        for layer in self.layers:
            layer.fire()

    def propagate(self):
        for layer in self.layers:
            layer.propagate()
    
    def print_network(self):
        for layer in self.layers:
            print([[c.weight for c in x.connections] for x in layer.neurons])

        


class Layer():
    
    def __init__(self):
        self.neurons = []
        
    def free(self):
        pass

    def initialize(self):
        for neuron in self.neurons:
            neuron.initialize(0)

    def add_neuron(self, neuron):
        self.neurons.append(neuron)
        
    def connect(self, weight, layer=None, neuron=None):
        if layer is None:
            for n in self.neurons:
                n.connect(neuron, weight)
        else:
            for n in self.neurons:
                for m in layer.neurons:
                    n.connect(m, weight)

    def fire(self):
        for neuron in self.neurons:
            neuron.fire()

    def propagate(self):
        for neuron in self.neurons:
            neuron.propagate()
         

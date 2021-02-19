###############################################################################
# FILE: neural_network.py
# AUTHORS: LEAH HADEED & EVA GUTIERREZ
# 
# BRIEF:
# Contains the class for the creation of the neural network. Abstract class?

import neuron 

class NeuralNetwork():

    def __init__(self):
        self.layers = []
        

    def create(self):
        pass

    def free(self):
        pass

    def initialize(self):
        pass

    def add_layer(self, layer):
        pass

    def fire(self):
        pass

    def propagate(self):
        pass


class Layer():
    
    def __init__(self):
        self.neurons = []
        

    def create(self):
        pass

    def free(self):
        pass

    def initialize(self):
        pass

    def add_layer(self, neuron):
        pass

    def connect(self, weight, layer=None, neuron=None):
        pass

    def propagate(self):
        pass
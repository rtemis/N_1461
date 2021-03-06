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
    
    def error(self, target):
        return [(target[i] - n.value)**2 for i, n in enumerate(self.layers[-1].neurons)]
    
    def print_network(self):
        for layer in self.layers:
            print([[c.weight for c in x.connections] for x in layer.neurons])

    def feedforward(self, ins):
        temp_in = ins + [1] 
        for i , m in enumerate(self.layers[0].neurons):
            m.initialize(temp_in[i])

        self.fire()
        self.initialize()
        self.propagate()
        
        return [n.value for n in self.layers[-1].neurons]

    def f(y_in):
        fy_in = []
        for y in y_in:
            if y.value > y.umbral:
                fy_in.append(1)
            elif y.value < - y.umbral:
                fy_in.append(-1)
            else: 
                fy_in.append(0)
        return fy_in


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
         

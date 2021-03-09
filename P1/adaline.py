from neural_network import NeuralNetwork, Layer
from neuron import Neuron, Connection

import itertools

def adaline(train_in, train_out, test_in, test_out, attributes, classes):
    flag = True
    tolerance = 1
    maxiter = 0
    alfa = 1

    layer_in = Layer()
    for att in range(attributes):
        layer_in.add_neuron(Neuron(0.2, "Directa"))
    
    # Bias 
    layer_in.add_neuron(Neuron(0.2, "Sesgo"))

    layer_out = Layer()    
    for cl in range(classes):
        layer_out.add_neuron(Neuron(0.2, "Directa"))

    for out in layer_out.neurons:
        layer_in.connect(1, neuron=out)
  

    network = NeuralNetwork()
    network.add_layer(layer_in)
    network.add_layer(layer_out)

    while flag == True and maxiter < 50: 
        for (s, t) in zip(train_in, train_out):
            temp_s = s + [1]
            for i , m in enumerate(network.layers[0].neurons):
                m.initialize(temp_s[i])

            network.fire()
            network.initialize()
            network.propagate()

            for i , y in enumerate(network.layers[-1].neurons):
                if y.value >= 0:
                    y.f_x = 1
                else: 
                    y.f_x = -1

                change = 0
                for c in y.inputs: 
                    c.w_prev = c.weight
                    c.weight += alfa * (t[i] - y.value) * c.input.f_x
                    if c.weight - c.w_prev > change:
                        change = c.weight - c.w_prev 
                if change < tolerance:
                    flag = False
                        
        maxiter += 1
        print('epoch ' , maxiter)

    return network


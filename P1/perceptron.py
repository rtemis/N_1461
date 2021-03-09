from neural_network import NeuralNetwork, Layer
from neuron import Neuron, Connection

import itertools

def perceptron(train_in, train_out, test_in, test_out, attributes, classes):
    flag = True
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

    while flag == True and maxiter < 120:
        flag = False
        for (s, t) in zip(train_in, train_out):
            temp_s = s + [1]
            for i , m in enumerate(network.layers[0].neurons):
                m.initialize(temp_s[i])

            network.fire()
            network.initialize()
            network.propagate()

            for y, ti in zip(network.layers[-1].neurons, t):
                if y.value > y.umbral:
                    y.f_x = 1
                elif y.value < - y.umbral:
                    y.f_x = -1
                else: 
                    y.f_x = 0

                # print (y.f_x, int(ti))
                if y.f_x != int(ti):
                    for c in y.inputs: 
                        c.w_prev = c.weight
                        c.weight += alfa * ti * c.input.f_x
                    #print([c.weight for c in y.inputs])
                    flag = True

        maxiter += 1
        print('epoch ' , maxiter)

    return network


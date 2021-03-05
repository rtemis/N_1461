from neural_network import NeuralNetwork, Layer
from neuron imoprt Neuron, Connection
from simulation import Data

import itertools

def perceptron():
    flag = False
    alfa = 0.05

    data = Data()

    train_in, train_out, test_in, test_out = data.load_data_proportional("and.txt", 0.7)

    layer_in = Layer()
    for att in data.attributes:
        layer_in.add_neuron(Neuron(0, "Directa"))
    
    # Bias 
    layer_in.add_neuron(Neuron(0, "Sesgo"))

    layer_out = Layer()    
    for cl in data.classes:
        layer_out.add_neuron(Neuron(0, "Directa"))


    for m in layer_in.neurons:
        m.initialize(0)
        for n in layer_out.neurons:
            m.connect(n, 1)


    network = NeuralNetwork()
    network.add_layer(layer_in)
    network.add_layer(layer_out)

    while flag == True:
        flag = False 
        for (s, t) in itertools.zip(train_in, train_out):

            for m in layer_in.neurons:
                m.initialize(s)

            network.fire()
            network.initialize()
            network.propagate()

            # Calculate the f_x 
            y_in = layer_in.neurons[-1].f_x

            for m in layer_in.neurons:
                for c in m.connections:
                    y_in += m.f_x * c.weight
                    c.w_prev = c.weight 

                if y_in != t:
                    c.weight = alfa * t * xi
                    bias += alfa * t 
                    flag = True

    return 

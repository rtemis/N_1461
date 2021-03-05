from neural_network import NeuralNetwork, Layer
from neuron imoprt Neuron, Connection
from simulation import Data

import math 

def adaline():
    flag = False
    
    data = Data()

    train_in, train_out, test_in, test_out = data.load_data_proportional("and.txt", 0.7)

    layer_in = Layer()
    for att in data.attributes:
        layer_in.add_neuron(Neuron(0, "Directa"))

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

    while(flag == False):
        for s, t in train_in:
            x_1.initialize(d[0])
            x_2.initialize(d[1])
            x_3.initialize(d[2])

            network.fire()
            network.initialize()
            network.propagate()
            print(d, a12.f_x, a13.f_x, a23.f_x, y.f_x)

            c.w_prev = c.weight 
            c.b_prev = c.bias 

            if y_in != t: 
                c.weight = alfa * (t-y_in) * xi
                bias += alfa * (t-y_in) 
            
        if math.sqrt((c.weight - c.w_prev)^2) < tolerance:
            flag = True

    return 


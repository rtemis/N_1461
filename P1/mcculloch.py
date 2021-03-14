from neuron import Neuron, Connection
from neural_network import NeuralNetwork, Layer

class McCulloch():
    def testConfig():
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
        x_2.initialize(0)
        x_3.initialize(1)

        x_1.fire()
        x_2.fire()
        x_3.fire()

        x_1.propagate()
        x_2.propagate()
        x_3.propagate()

        a12.fire()
        a13.fire()
        a23.fire()

        a12.propagate()
        a13.propagate()
        a23.propagate()

        y.fire()
        y.propagate()

        print(a12.f_x)
        print(a13.f_x)
        print(a23.f_x)

        print(y.f_x)


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

    data = [[0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1], [1, 0, 0], [1, 0, 1], [1,1,0], [1,1,1]]

    for d in data:
        x_1.initialize(d[0])
        x_2.initialize(d[1])
        x_3.initialize(d[2])

        network.fire()
        network.initialize()
        network.propagate()
        print(d, a12.f_x, a13.f_x, a23.f_x, y.f_x)
    
    network.fire()
    network.initialize()
    network.propagate()
    print(d, a12.f_x, a13.f_x, a23.f_x, y.f_x)

    network.fire()
    network.initialize()
    network.propagate()
    print(d, a12.f_x, a13.f_x, a23.f_x, y.f_x)

    network.fire()
    network.initialize()
    network.propagate()
    print(d, a12.f_x, a13.f_x, a23.f_x, y.f_x)

    # network.initialize()
    # network.fire()
    # network.initialize()
    # network.propagate()
    # print(d, a12.f_x, a13.f_x, a23.f_x, y.f_x)

    # network.initialize()
    # network.fire()
    # network.initialize()
    # network.propagate()
    # print(d, a12.f_x, a13.f_x, a23.f_x, y.f_x)

main()
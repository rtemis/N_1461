from neuron import Neuron, Connection

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


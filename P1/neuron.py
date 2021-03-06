from enum import Enum

class NeuronType(Enum):
    DIRECT = 0
    MCCULLOCH = 1
    SESGO = 2
    SIGMOIDBIPOLAR = 3
    SIGMOIDPERSONALIZED = 4
    
class Neuron():

    # Create function for the neuron
    def __init__(self, umbral, type, active=None, inactive=None):
        # Threshold for activation
        self.umbral = umbral 
        # Determinates for whether the neuron is activated or not
        if active == None and inactive == None:
            self.output_active = 0.0 
            self.output_inactive = 0.0 
        else:
            self.output_active = active
            self.output_inactive = inactive
        # The type of neuron 
        self.type = type
        # Value of the neuron
        self.value = 0.0
        self.f_x = 0.0
        # List of connections with the neuron
        self.connections = []
        self.inputs = []

    def free(self):
        pass

    def initialize(self, val):
        # Sets the neuron value 
        self.value = val
         

    def connect(self, output, weight):
        c = Connection(weight=weight, input=self, output=output)
        self.connections.append(c)
        output.inputs.append(c)
        
    
    def fire(self):
        # Typical input neurons 0 or 1
        if self.type == "Directa":
            self.f_x = self.value
        elif self.type == "Sesgo":
            self.f_x = 1.0
        # Typical output neuron 
        elif self.type == "McCulloch":
            if self.value >= self.umbral:
                self.f_x = self.output_active
            else:
                self.f_x = self.output_inactive
        else:
            self.f_x = self.output_inactive

        for cxn in self.connections:
            cxn.propagate(self.f_x)
        

    def propagate(self):
        for cxn in self.connections:
            cxn.output.value += cxn.weight * cxn.val_received

        


class Connection():

    def __init__(self, weight, input, output):
        self.weight = weight
        self.w_prev = 0.0
        self.val_received = 0.0
        self.output = output
        self.input = input

    def free(self):
        pass

    def propagate(self, f_x):
        self.val_received = f_x




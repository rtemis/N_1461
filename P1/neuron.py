class Neuron():

    def __init__(self, umbral, typeN, active=None, inactive=None):
        # Threshold for activation
        self.umbral = umbral 
        
        # Determinates for 
        if active == None and inactive == None:
            self.output_active = 0.0 
            self.output_inactive = 0.0 
        else:
            self.output_active = active
            self.output_inactive = inactive
        # ?
        self.type = typeN

        #
        self.value = 0.0
        self.f_x = 0.0

        self.connections = []

    def free(self):
        pass

    def initialize(self, val):
        self.value = val
        return 

    def connect(self, neuron, weight):
        self.connections.append(Connection(weight=weight, neuron=neuron))
        return
    
    def fire(self):
        pass

    def propagate(self):
        pass 


class Connection():

    def __init__(self, weight, neuron):
        self.w = weight
        self.w_prev = 0.0
        self.val_received = 0.0
        self.neurons = []

    def free(self):
        pass

    def propagate(self):
        pass 




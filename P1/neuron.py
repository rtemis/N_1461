class Neuron():

    def __init__(self, umbral, typeN, active=0.0, inactive=0.0):
        # Threshold for activation
        self.umbral = umbral 
        
        # Determinates for 
        self.output_active = active 
        self.output_inactive = inactive 

        # ?
        self.type = typeN

        #
        self.value = 0.0
        self.f_x = 0.0

        self.connections = []

    def create(self, umbral, typeN, active=None, inactive=None):
        if active == None and inactive == None:
            self.__init__(umbral=umbral, typeN=typeN)
        else:
            self.__init__(umbral=umbral, typeN=typeN, active=active, inactive=inactive)
        return  

    def free(self):
        pass

    def initialize(self, val):
        self.value = val
        return 

    def connect(self, neuron):
        self.connections.append(neuron)
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
        for i in self.neurons:
            i.free
        return

    def propagate(self):
        pass 




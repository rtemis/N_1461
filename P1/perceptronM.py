import random
from data import Data
from math import exp
from matplotlib import pyplot

class PerceptronM():
    # layers = [10, 20, 5, 3]
    # val per layer
    def __init__(self, layers, bipolar=True, seed=None):
        random.seed(seed)
        self.weights = []
        self.bipolar = bipolar

        for i in range(len(layers)-1):
            self.weights.append([])
            # All neurons plus bias = layers[i] + 1l
            for j in range(layers[i + 1]):
                self.weights[i].append([random.uniform(-1,1) for k in range(layers[i] + 1)])

    def sigmoid(self, value):
        return ((2.0 / (1.0 + exp(-value))) - 1) if self.bipolar else (1.0 / (1 + exp(-value)))

    def sigmoidD(self, value):
        return ((1.0 + value)*(1.0 - value)) / 2.0 if self.bipolar else (value * (1.0 - value))

    # b + Sum(wi * si) = y_in
    def activate(weights, source):
        outs = []
        for i, outW in enumerate(weights):
            outs.append(outW[0])
            for j, w in enumerate(outW[1:]):
                outs[i] += w * source[j]

        return outs

    def exploit(self, source):
        for w in self.weights:
            neurons = PerceptronM.activate(w, source)
            source = [self.sigmoid(n) for n in neurons]
        return source

    def propagate(self, source, target, numLayers, currentLayer):
        currentWeight = self.weights[currentLayer]
        
        if currentLayer < numLayers:
            # Propagacion capa oculta
            nextLayer = PerceptronM.activate(currentWeight, source)
            nextLayerSig = [self.sigmoid(n) for n in nextLayer]
            outs, d_in = self.propagate(nextLayerSig, target, numLayers, currentLayer+1)
            # Error = d_in * sigD(y_in)
            error = [d_in[i] * self.sigmoidD(y_in) for i, y_in in enumerate(nextLayerSig)]
            d_in = [0]*(len(source) + 1)

            # Retropropagation capa oculta
            for i, dj in enumerate(error):
                # Calculo error de entradas
                d_in[0] += currentWeight[i][0] * dj
                for j, zi in enumerate(source):
                    d_in[j+1] += currentWeight[i][j+1] * dj
                    
                    # Actualizacion de pesos
                    currentWeight[i][j+1] += self.alpha * dj * zi 
                currentWeight[i][0] += self.alpha * dj 
            
            return [outs, d_in]

        else:
            # propagate last layer
            endLayer = PerceptronM.activate(currentWeight, source)
            endLayerSig = [self.sigmoid(n) for n in endLayer]
            # Error = (ti - y_in) * sigD(y_in)
            error = [(target[i] - y_in) * self.sigmoidD(y_in) for i, y_in in enumerate(endLayerSig)]
            d_in = [0]*(len(source) + 1)
                    
            # Retropropagacion ultima capa 
            for i, dj in enumerate(error):
                # Calculo error de entradas
                d_in[0] += currentWeight[i][0] * dj
                for j, zi in enumerate(source):
                    d_in[j+1] += currentWeight[i][j+1] * dj
                    
                    # Actualizacion de pesos
                    currentWeight[i][j+1] += self.alpha * dj * zi 
                currentWeight[i][0] += self.alpha * dj 
            
            return [endLayerSig, d_in]

    def train(self, train_in, train_out, epochs=5000, alpha=0.2, tolerance=0.0001):
        self.epochs = epochs
        self.alpha = alpha

        ECMs = []
        for i in range(epochs):
            epochECM = 0
            for (s, t) in zip(train_in, train_out):
                output = self.propagate(s, t, len(self.weights)-1, 0)[0]
                epochECM += sum([(t[i] - oi)**2 for i, oi in enumerate(output)])
            ECMs.append(epochECM)
            if epochECM < tolerance:
                break

        return ECMs


data = Data()
p = PerceptronM([2, 5, 1])
tr_in, tr_out = data.load_data_file('./data/and.txt')
ecms = p.train(tr_in, tr_out)
print(p.exploit(tr_in[0]), tr_in[0], tr_out[0])
pyplot.plot(ecms)
pyplot.savefig('plot.png')


        

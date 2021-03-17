from neural_network import NeuralNetwork, Layer
from neuron import Neuron, Connection
from data import Data

import itertools
import sys 
from parser import parse
from matplotlib import pyplot as plt
import numpy as np
from sklearn import metrics

class Perceptron():
    # Configure the network with the structure:
    #   x1--w1--
    #           -
    #   x2--w2-----Yi
    #   ...     -
    #   1---b---
    def __init__(self, attributes, classes, umbral=0.9, alpha=0.5, patience=200):
        # Create input layer
        layer_in = Layer()
        # Add all neurons to input layer based on number of attributes in the file
        for att in range(attributes):
            layer_in.add_neuron(Neuron(umbral, "Directa"))
        
        # Add bias as final neuron in layer
        layer_in.add_neuron(Neuron(1, "Sesgo"))

        # Create output layer
        layer_out = Layer()  
        # Add all the output neurons to the final layer   
        for cl in range(classes):
            layer_out.add_neuron(Neuron(umbral, "Directa"))

        # Adjust initial weights for each connection
        for out in layer_out.neurons:
            layer_in.connect(1, neuron=out)
    
        # Set up the network
        self.network = NeuralNetwork()
        # Add the layers to the network
        self.network.add_layer(layer_in)
        self.network.add_layer(layer_out)
        # Create variable for Delta rule / ECM
        self.ecm = []
        # Configure alfa value (moderates the change in learning)
        self.alpha = alpha
        # Configure patience value (maximum iterations over algorithm)
        self.patience = patience

    def train(self, train_in, train_out, verbose=False):
        flag = True
        maxiter = 0
        weights = {}

        while flag == True and maxiter < self.patience:
            flag = False
            eq = [0]*len(self.network.layers[-1].neurons)
            for (s, t) in zip(train_in, train_out):
                self.network.feedforward(s)

                eq = [eq[i] + x for i, x in enumerate(self.network.error(t))]

                i = 0
                for y, ti in zip(self.network.layers[-1].neurons, t):
                    if y.value > y.umbral:
                        y.f_x = 1
                    elif y.value < - y.umbral:
                        y.f_x = -1
                    else: 
                        y.f_x = 0

                    weights[i] = {}
                    if verbose:
                        print (y.f_x, int(ti))
                    if y.f_x != int(ti):
                        for c in y.inputs: 
                            c.w_prev = c.weight
                            c.weight += self.alpha * ti * c.input.f_x
                        if verbose:
                            print([c.weight for c in y.inputs])
                        flag = True

                    for j, c in enumerate(y.inputs):
                        weights[i][j] = c.weight
                    i += 1


            maxiter += 1
            self.ecm.append([e/len(train_in) for e in eq])

            if verbose:
                print('epoch ' , maxiter, self.ecm)
        return weights

    def test(self, weights, test_in, test_out, filename='prediccion_perceptron.txt'):
        fp = open(filename, 'w')

        for i , y in enumerate(self.network.layers[-1].neurons):
            for j, c in enumerate(y.inputs): 
                c.weight = weights[i][j]
        
        for source in test_in:
            # Initialize, fire, and propagate 
            self.network.feedforward(source)
            # Write prediction outputs to file
            for y in self.network.layers[-1].neurons:
                if y.value > y.umbral:
                    fp.write(' 1\t')
                elif y.value < -y.umbral: 
                    fp.write('-1\t')
                else:
                    fp.write(' 0\t')
            fp.write('\n')

        fp.close()
    
    def accuracy(self, t_out):
        predict_correct = 0
        total = 0
        verbose = False

        # Compare predictions to values
        fp = open('prediccion_perceptron.txt', 'r')
        for i, line in enumerate(fp.readlines()):
            predictions = line.split()
            if verbose:
                print('pred',predictions)
                print('test',t_out[i])
            for j, p in enumerate(predictions):
                if verbose:
                    print('cmp:',p, int(t_out[i][j]))
                if int(p) == int(t_out[i][j]):
                    predict_correct += 1
                total += 1
        
        print('Correct Predictions:',predict_correct, 'Total predictions:',total)
        print('Accuracy:',predict_correct/total)

        fp.close()

    def plot(self, filename):
        plotx = []
        ploty = []
        for x, y in enumerate(self.ecm):
            plotx.append(x)
            ploty.append(y)
        
        plt.plot(plotx, ploty)
        plt.savefig(filename)



def main():
    # Create the data shelter object
    data = Data()

    if len(sys.argv) - 1 == 0:
        # Read data from files
        tr_in, tr_out = data.load_data_file('data/and.txt')

        per_logic = Perceptron(attributes=data.attributes, classes=data.classes)

        # Create and train the adaline network
        weights = per_logic.train(tr_in, tr_out)
        # Print network
        per_logic.network.print_network()

        # Train and test
        tr_in, tr_out, t_in, t_out = data.load_data_proportional('data/problema_real1.txt')
        perceptron = Perceptron(attributes=data.attributes, classes=data.classes)
        weights = perceptron.train(tr_in, tr_out)
        # Print the network
        perceptron.network.print_network()
        # Test predictions and write to file
        perceptron.test(weights=weights, test_in=t_in, test_out=t_out)

        perceptron.accuracy(t_out)

    else: 
        params = parse(sys.argv[1:])

        alpha = float(params['--alpha']) if params.get('--alpha') != None else 1
        umbral = float(params['--umbral']) if params.get('--umbral') != None else 0.2
        tolerance = float(params['--tolerance']) if params.get('--tolerance') != None else 0.0001
        patience = int(params['--patience']) if params.get('--patience') != None else 200
        file1 = params['--train'] if params.get('--train') != None else 'data/problema_real1.txt'
        file2 = params['--test'] if params.get('--test') != None else 'data/problema_real1.txt'
        prop = float(params['--prop']) if params.get('--prop') != None else 0.7
       
        if params.get('--data'):
            if params['--data'] == '1':
                # Train and test
                tr_in, tr_out, t_in, t_out = data.load_data_proportional(file1, prop)
            elif params['--data'] == '2':
                # Train and test
                tr_in, tr_out = data.load_data_file(file1)
                t_in = tr_in
                t_out = tr_out
            elif params['--data'] == '3':
                # Train and test
                tr_in, tr_out, t_in, t_out = data.load_data_files(file1, file2)
        else:
            # Train and test
            tr_in, tr_out = data.load_data_file(file1)
            t_in = tr_in
            t_out = tr_out

        perceptron = Perceptron(attributes=data.attributes, classes=data.classes, umbral=umbral, alpha=alpha, patience=patience)
        weights = perceptron.train(tr_in, tr_out)
        # Print the network
        perceptron.network.print_network()
        # Test predictions and write to file
        perceptron.test(weights=weights, test_in=t_in, test_out=t_out)

        perceptron.accuracy(t_out)

        perceptron.plot('perceptron_ecm_'+str(alpha)+'_'+str(umbral)+'.png') 

main()
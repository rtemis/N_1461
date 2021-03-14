import os
import random

class Data():

    def __init__(self):
        self.train_in = []
        self.train_out = []
        self.test_in = []
        self.test_out = []

        self.attributes = 0
        self.classes = 0        
        
        self.attributes_test = 0
        self.classes_test = 0

    def load_data_proportional(self, filename, prop=0.7):

        file = open(filename, "r")
        data = file.readlines()
        file.close()

        config = data[0].split()
        self.attributes = int(config[0])
        self.classes = int(config[1])
        
        data = data[1:]
        #random.shuffle(data)

        ls_in = []
        ls_out = []
        for i in data:
            line = [float(x) for x in i.split()]
            ls_in.append(line[:self.attributes])
            ls_out.append(line[self.attributes:])

        length = len(data)
        threshold = int(length * prop)

        self.train_in = ls_in[:threshold]
        self.train_out = ls_out[:threshold]      
        
        self.test_in = ls_in[threshold:]
        self.test_out = ls_out[threshold:]

        return self.train_in, self.train_out, self.test_in, self.test_out

    def load_data_file(self, filename):

        file = open(filename, "r")
        data = file.readlines()
        file.close()

        config = data[0].split()
        self.attributes = int(config[0])
        self.classes = int(config[1])
        
        data = data[1:]
        #random.shuffle(data)

        for i in data:
            line = [float(x) for x in i.split()]
            self.train_in.append(line[:self.attributes])
            self.train_out.append(line[self.attributes:])

        return self.train_in, self.train_out

    def load_data_files(self, file_train, file_test):
        file = open(file_train, "r")
        data = file.readlines()
        file.close()
        
        config = data[0].split()
        self.attributes = config[0]
        self.classes = config[1]
        
        data = data[1:]
        random.shuffle(data)

        for i in data:
            line = [float(x) for x in i.split()]
            self.train_in.append(line[:self.attributes])
            self.train_out.append(line[self.attributes:])
        
        file = open(file_train, "r")
        data = file.readlines()
        file.close()
        
        config = data[0].split()
        if self.attributes != config[0] or self.classes != config[1]:
            raise Exception('Files of different sizes')

        data = data[1:]
        random.shuffle(data)

        for i in data:
            line = [float(x) for x in i.split()]
            self.test_in.append(line[:self.attributes])
            self.test_out.append(line[self.attributes:])

        return self.train_in, self.train_out, self.test_in, self.test_out

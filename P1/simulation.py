import os
import neural_network
import random

class Data():

    def __init__(self):
        self.train = []
        self.test = []

    def load_data_proportional(self, filename, prop=0.7):

        file = open(filename, "r")
        data = file.readlines()
        file.close()

        random.shuffle(data)

        length = len(data)
        threshold = length * prop

        for i in range(threshold):
            self.train.append(data[i])
        
        for i in range(threshold, length):
            self.test.append(data[i])
            
        return 

    def load_data_file(self, filename):
        pass

    def load_data_files(self, file_train, file_test):
        pass



def main():
    pass

if __name__ == "__main__":
    main()



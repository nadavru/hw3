# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

class KNN(object):
    def __init__(self, k=5):
        self.k=k
    
    def train(self, data):
        self.data = data
    
    def test(self, data):
        right = 0
        wrong = 0
        for i in range(data.shape[0]):
            if self.classify(data[i]) == data[i][0]:
                right += 1
            else:
                wrong += 1
                #print('wrongly classified ', i)
        accuracy = right/(right+wrong)*100
        print('right answers: ', right)
        print('wrong answers: ',  wrong)
        print('accuracy: ', accuracy, '%')
        return accuracy
    
    def classify(self, x):
        length = len(self.data)
        if length<=self.k:
            num_of_ones = np.count_nonzero(self.data[:,0])
            if num_of_ones >= length/2:
                return 1
            else:
                return 0
        distances = [[sum(abs(self.data[i][1:]-x[1:])), self.data[i][0]] for i in range(length)]
        best_indexes = sorted(range(length), key = lambda sub: distances[sub][0])[:self.k]
        best_results = []
        for i in range(self.k):
            best_results .append( distances[best_indexes[i]][1] )
        #print(best_results)
        num_of_ones = best_results.count(1)
        if num_of_ones > self.k/2:
            return 1
        else:
            return 0

def prepare_data():
    train_data = np.loadtxt('train.csv', delimiter=',', skiprows=1)
    test_data =  np.loadtxt('test.csv', delimiter=',', skiprows=1)
    min_value = [min(train_data[:,index]) for index in range(1,len(train_data[0]))]
    max_value = [max(train_data[:,index]) for index in range(1,len(train_data[0]))]
    
    '''for index in range(1,len(train_data[0])):
        min_value[index-1] = min(train_data[:,index])
        max_value[index-1] = max(train_data[:,index])
        #print(max_value[index-1]-min_value[index-1])'''
    
        
    for x in train_data:
        for index, feature in enumerate(x[1:]):
            diff = max_value[index]-min_value[index]
            if diff==0.0:
                x[index+1] = 0.5
            else:
                x[index+1] = (feature-min_value[index])/diff
                
    for x in test_data:
        for index, feature in enumerate(x[1:]):
            diff = max_value[index]-min_value[index]
            if diff==0.0:
                x[index+1] = 0.5
            else:
                x[index+1] = (feature-min_value[index])/diff
        
    return train_data, test_data

def main():
    x = [1,3,9,27] 
    y = []
    train, test = prepare_data()
    for i in range(len(x)):
        knn = KNN(k=x[i])
        knn.train(train)
        #print("testing")
        y.append(knn.test(test))
    
    # plotting the points  
    plt.plot(x, y) 
      
    # naming the x axis 
    plt.xlabel('K values') 
    # naming the y axis 
    plt.ylabel('accuracy') 
      
    # function to show the plot 
    plt.show() 

if __name__ == '__main__':
    main()

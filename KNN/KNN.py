import numpy as np
import pandas as pd
from collections import defaultdict

class KNN:
    #constuctor
    def __init__(self, metric, k):
        self.metric = metric
        self.k = k
    
    #call fit to create data fields x and y
    def fit(self,x,y):
        self.x = np.array(x)
        self.y = np.array(y)

    #actualy predict the data but only 1 point at a time.
    def predict(self, x):
        #ensure data is actually numpy array
        x = np.array(x)

        #check which distance calculation to do
        if self.metric == 'L2':
            distance = np.linalg.norm(self.x - x, axis=1)
        else:
            norm_x = np.linalg.norm(x)
            norm_self_x = np.linalg.norm(self.x, axis=1)
            cosine_sim = np.dot(self.x, x) / (norm_self_x * norm_x)
            distance = 1 - cosine_sim
        #combine x and y data so that when sorted it easy to grab relevant labels
        xy = np.column_stack((distance,self.y))

        #sort matrix
        xy = xy[xy[:,0].argsort()]
        
        #grab k samples with lowest distance
        neighbors = xy[:self.k]

        #count labels of top k dynamically
        labels = defaultdict(int)
        for label in neighbors[:,1]:
            labels[label] += 1

        #return the label with largest count
        return max(labels, key = labels.get)
        


    

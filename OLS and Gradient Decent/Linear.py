import numpy as np
import pandas as pd

class LinearRegression():
    def __init__(self, method='OLS'):
        self.method = method
    
    def fit(self,x,y):
        self.x = np.array(x,dtype='float64')
        x_0 = np.ones(self.x.shape[0],dtype='float64')
        self.x = np.column_stack([x_0,self.x])
        self.y = np.array(y,dtype='float64')
        self.xt = np.array(np.transpose(self.x),dtype='float64')
        self.b = np.array(np.linalg.inv(np.dot(self.xt,self.x))@(self.xt@self.y),dtype='float64')

    def predict(self, x):
        x = np.array(x,dtype='float64')
        x_0 = np.ones(x.shape[0],dtype='float64')
        x = np.column_stack([x_0,x])
        return x@self.b
    
    def returnB(self):
        return self.b
    
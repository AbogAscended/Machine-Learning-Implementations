import numpy as np
import pandas as pd
from Linear import LinearRegression

data = pd.read_table(
    "data/auto-mpg.txt",
    sep="\s+",
    quotechar='"',
    names=['mpg','cylinders','displacement','horsepower','weight',
           'acceleration','model_year','origin','car_name'],
    na_values='?',
    dtype={
        'horsepower': float,
        'car_name': str
    }
)
data = data.drop('car_name', axis=1)
data = data.dropna()
y = data.pop('mpg')
x = data

sampleSize = x.shape[0]
xtrain = x[0:int(np.floor(sampleSize*.8))]
xtest = x[int(np.floor(sampleSize*.8))+1:]
ytrain = y[0:int(np.floor(sampleSize*.8))]
ytest = y[int(np.floor(sampleSize*.8))+1:]

foldSize = np.floor(sampleSize/10)
xdataFolds, ydataFolds, foldModels, foldPredicts, foldRMSE = [], [], [], [], []
for i in range(10):
    foldModels.append(LinearRegression())
    foldModels[i].fit(xtrain[int(i*foldSize):int((i+1)*foldSize)], ytrain[int(i*foldSize):int((i+1)*foldSize)])
    foldPredicts.append(foldModels[i].predict(xtest))
    foldRMSE.append(np.sqrt((foldPredicts[i]-ytest)**2))

print(foldRMSE)
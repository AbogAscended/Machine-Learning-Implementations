import pandas as pd
from KNN import KNN
from sklearn.model_selection import train_test_split 

#Read in data from csv files
training = pd.read_csv('Data/MNIST_training.csv')
test = pd.read_csv('Data/MNIST_test.csv')

#assign y as label variable and x as features
y_test = test['label']
X_test = test.drop(['label'], axis = 1)

y_train = training['label']
X_train = training.drop(['label'], axis = 1)

#create model with knn and then fit with training data
model = KNN("L2",3)
model.fit(X_train,y_train)

#check each point in test for prediction and check if correct
correct, false = 0, 0
for i in range(len(X_test)):
    predicted = model.predict(X_test.iloc[i])
    true = y_test.iloc[i]
    if predicted == true:
        correct += 1
    else:
        false += 1

#print total correct and false and output accuracy
print(f"The modeL got {correct} correct and {false} false overall,")
print(f"accuracy is {correct/(correct+false)}")

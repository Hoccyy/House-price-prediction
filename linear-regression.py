import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

import matplotlib.pyplot as pyplot
import pickle
from matplotlib import style

# Load the data
data = pd.read_csv('train.csv', sep=',')

# Last value included to be dropped but also for test set
data = data[['LotArea', 'YearBuilt', 'TotRmsAbvGrd', 'Fireplaces', 'GarageCars', 'GarageArea', 'MSSubClass', 'OverallQual', 'OverallCond', 'YearRemodAdd', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF', 'GrLivArea', 'GarageCars', 'GarageArea', 'YrSold', 'WoodDeckSF', 'SalePrice']]
# print(data.head)  - maybe remove last 3

# Value to predict
predict = "SalePrice"

X = np.array(data.drop([predict], axis=1))
y = np.array(data[predict])


x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


pickle_in = open('housemodel.pickle', 'rb')
model = pickle.load(pickle_in)

predictions = model.predict(x_test)

accuracy = model.score(x_test, y_test)

for i in range (len(predictions)):
    print ("Guess $" + str(predictions[i]) + "    |     Real $" + str(y_test[i]))

print (accuracy)
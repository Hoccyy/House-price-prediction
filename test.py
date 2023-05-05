import pickle
import pandas as pd
import csv


data = pd.read_csv('train.csv')
data = data[['LotArea', 'YearBuilt', 'TotRmsAbvGrd', 'Fireplaces', 'GarageCars', 'GarageArea', 'MSSubClass', 'OverallQual', 'OverallCond', 'YearRemodAdd', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF', 'GrLivArea', 'GarageCars', 'GarageArea', 'YrSold', 'WoodDeckSF']]


pickle_in = open('housemodel.pickle', 'rb')
model = pickle.load(pickle_in)


predictions = model.predict(data)

# outputting predictions
with open('predictions.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Predicted', 'Actual'])
    for i in range(len(predictions)):
        writer.writerow([predictions[i], data])
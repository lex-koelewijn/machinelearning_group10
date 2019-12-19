import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# %matplotlib inline

#This function returns slices, currently only allows for an array as dataset. (Indexing should be altered when data is matrix.)
def create_dataset(dataset, look_back=1):
    X, Y = [], []
    for i in range(len(dataset)-look_back-1):
        a = dataset[i:(i+look_back)]
        X.append(a)
        Y.append(dataset[i + look_back])
    return np.array(X), np.array(Y)


# +
dataset = pd.read_csv('F.txt', sep='\t', header=None)
dataset.head()
dataset.describe()

#Create data matrix with slices of 20 training data points of training data and 1 test point.
X, y = create_dataset(dataset.loc[:,2].to_numpy(), 20) 
#TODO: Vector encoding moet worden aangepast naar een formaat wat OK is. 
print(X)

# +
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

#Train a linear model
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

#Show the predicted coefficients
coeff_df = pd.DataFrame(regressor.coef_, columns=['Coefficient'])
#Coeffecients currently represent which of the past 20 notes in the window is most influential for the prediction.
print(coeff_df)

#Make a prediction
y_pred = regressor.predict(X_test)

# +
#Show The difference between actual values and predicted values
df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
print(df)

from sklearn import metrics
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
# -



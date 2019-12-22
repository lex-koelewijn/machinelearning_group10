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
window_size = 20

#Create data matrix with slices of 20 training data points of training data and 1 test point.
X, y = create_dataset(dataset.loc[:,2].to_numpy(), window_size) 
#TODO: Vector encoding moet worden aangepast naar een formaat wat OK is. 

# +
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

#Train a linear model
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

#Show the predicted coefficients
coeff_df = pd.DataFrame(regressor.coef_, columns=['Coefficient'])
# #Coeffecients currently represent which of the past 20 notes in the window is most influential for the prediction.
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
# ## One Hot Vectors
# Train a linear regression again, but this time use One Hot encodings. 

# +
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

#Create One hot encodings of X and y data. 
onehot_encoder = OneHotEncoder(sparse=False)
y_onehot = onehot_encoder.fit_transform(y.reshape(len(y),1))
X_onehot = onehot_encoder.fit_transform(X)

#Reshape X cause all training one hot vectors in the window get concatenated by sklearn. 
X_onehot = X_onehot.reshape(X_onehot.shape[0], int(X_onehot.shape[1]/window_size), window_size)


# +
X_train, X_test, y_train, y_test = train_test_split(X_onehot, y_onehot, test_size=0.2, random_state=0)

#Train a linear model by going through each window of training data seperately. 
regressor = LinearRegression()
for i in range(0,X_train.shape[0]):
    regressor.fit(X_train[i,:,:], y_train[i])

#Show the predicted coefficients
coeff_df = pd.DataFrame(regressor.coef_, columns=['Coefficient'])
# #Coeffecients currently represent which of the past 20 notes in the window is most influential for the prediction.
print(coeff_df)
# -

#Make predictions of our test data based on trained model
y_pred = np.empty((X_test.shape[0], X_test.shape[1]))
print(X_test[0,:,:])
#Go through each testing window and predict. 
for i in range(0,X_test.shape[0]):
    y_pred[i] = regressor.predict(X_test[i,:,:])
print((y_pred[0])))
test = 0 
for val in y_pred[0]:
    test += abs(val)
print(test)


#Return the original integer note with the highest probability from the prediction. 
def one_hot_to_original_data(y_onehot,y):
    y_orig = np.empty(y_onehot.shape[0])
    for i in range(0,y_onehot.shape[0]):
        y_orig[i] = np.unique(y)[np.argmax(y_onehot[i,:])]
    return y_orig


# +
#Convert back to integer notes by taking the note with the maximum proability. 
y_test_orig = one_hot_to_original_data(y_test, y)
y_pred_orig = one_hot_to_original_data(y_pred, y)

# Show The difference between actual values and predicted values
df = pd.DataFrame({'Actual': y_test_orig, 'Predicted': y_pred_orig})
print(df)

from sklearn import metrics
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test_orig, y_pred_orig))
print('Mean Squared Error:', metrics.mean_squared_error(y_test_orig, y_pred_orig))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test_orig, y_pred_orig)))
# -



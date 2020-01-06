import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from matplotlib.ticker import MaxNLocator
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
# -

# Create list of unique notes, which is later used for onehot encoding
y_df = pd.DataFrame(y, columns=['notes'])
sorted_notes = sorted(y_df['notes'].unique())
print(sorted_notes)

# We want to find out if for example every four values have a correlation
fig = sm.graphics.tsa.plot_acf(dataset.loc[:,2], lags=5)
# Does not seem that there is a correlation when taking the n-order difference

# +
from sklearn.model_selection import train_test_split
# Shuffle false so that only the last 20% of values are chosen
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0, shuffle=False)

# Train a linear model
from sklearn.linear_model import LinearRegression, Ridge
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Show the predicted coefficients
coeff_df = pd.DataFrame(regressor.coef_, columns=['Coefficient'])
# Coefficients currently represent which of the past 20 notes in the window is most influential for the prediction.
print(coeff_df['Coefficient'])

# Plot the autocorrlation between different lags in the coefficients
# Here we do see that there is some correlation between every fourth value
# But also the third and fifth value
sm.graphics.tsa.plot_acf(coeff_df['Coefficient'], lags=10)

# Now plot the coefficients themselves, there are interesting correlations
# It does not seem that there is a clear correlation between every fourth value,
# But there are correlations found
plt.figure(figsize=(12,8))
plt.plot(coeff_df['Coefficient'])
ax = plt.gca()
ax.xaxis.set_major_locator(MaxNLocator(integer = True))

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
# ## One Hot Vector With Reshaping Of Data
# Train a linear regression again, but this time use One Hot encodings. In this implementation data is restructured but this seems to harm performance as you cannot iteratively train the linear model. 

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
X_train, X_test, y_train, y_test = train_test_split(X_onehot, y_onehot, test_size=0.2, random_state=0, shuffle=False)

#Train a linear model by going through each window of training data seperately. 
regressor = LinearRegression()
for i in range(0,X_train.shape[0]):
    regressor.fit(X_train[i,:,:], y_train[i])

#TODO: I am unsure you can train a model iterativly as I'm doing now.     
#Show the predicted coefficients
coeff_df = pd.DataFrame(regressor.coef_, columns=['Coefficient'])
# #Coeffecients currently represent which of the past 20 notes in the window is most influential for the prediction.
print(coeff_df)

# +
#Make predictions of our test data based on trained model
y_pred = np.empty((X_test.shape[0], X_test.shape[1]))

#Go through each testing window and predict. 
for i in range(0,X_test.shape[0]):
    y_pred[i] = regressor.predict(X_test[i,:,:])


# -

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
# # One Hot Vector Without Reshaping Data
#
# In this version of the one hot vector incoding the data is not being reshaped, so we use the original concatenated input vector for training. This model has similar/slightly better performance than the simple linear regression using the notes as data.

# +
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

#Create One hot encodings of X and y data. 
#categories=np.reshape(len(sorted_notes),1)
onehot_encoder = OneHotEncoder(sparse=False)
y_onehot = onehot_encoder.fit_transform(y.reshape(len(y),1))
X_onehot = onehot_encoder.fit_transform(X)



# +
X_train, X_test, y_train, y_test = train_test_split(X_onehot, y_onehot, test_size=0.2, random_state=0, shuffle=False)

#Train a ridge regression model on training data.  
regressor = Ridge()
regressor.fit(X_train,y_train)

#Make predictions for test data 
y_pred = regressor.predict(X_test)

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
# Predict a new signal based on the last part of the signal and the window size

# +
# Define how many stes we want to predict and init required arrays
steps_to_predict = 100
signal = dataset.loc[:,2].to_numpy() # Get the original signal
new_part = np.empty(steps_to_predict)

# For the amount of steps to predict, predict a new note using the past x notes as defined in the window size. 
# Then append this new note to the signal and predict again using a window over the last notes in the signal.
# So new notes also infuence the prediction.
for i in range(0,steps_to_predict):
    window = signal[len(signal)-window_size-1:len(signal)-1]
    x = onehot_encoder.transform(window.reshape(1,-1))
    next_prob = regressor.predict(x)
    next_note = one_hot_to_original_data(next_prob, y)
    np.append(signal, next_note)
    new_part[i] = next_note
    
print("Newly predicted notes: \n", new_part)
# -








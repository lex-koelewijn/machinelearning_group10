# # Predicting Bach using Window Based Linear Regression
#
# In this notebook we will try to predict Bach, ie we try to solve the time series prediction task of finishing his unfinished fugue. In the notebook we start with a simple regression on the data as is and we gradually move to more complex variations. To give short overview before being overwhelmed by the notebook:
# 1. Simple window based linear regression on raw data 
# 2. Window based linear regression but with one hot encodings of the notes
# 3. Window based linear regression with one hot encodings of both the note and its duration. 
# 4. Window based linear regression where the notes are represented by their chroma representation
#
# This means that large parts of the code get repeated with the adjustments made in order to make the new idea work. The notebook contains everything in order to show the progression we made during the project. 

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from matplotlib.ticker import MaxNLocator
# %matplotlib inline

#This function returns slices, currently only allows for an array (possibly of tuples) as dataset.
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
window_size = 40 #The window size is used throughout the enite notebook

#Create data matrix with slices of window_size training data points of training data and 1 test point. PLease note that only 1 voice is used.
X, y = create_dataset(dataset.loc[:,2].to_numpy(), window_size) 
# -

# Create list of unique notes, which is later used for onehot encoding
y_df = pd.DataFrame(y, columns=['notes'])

# We want to find out if for example every four values have a correlation
fig = sm.graphics.tsa.plot_acf(dataset.loc[:,2], lags=16)

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
# Coefficients currently represent which of the past window_size notes in the window is most influential for the prediction.
# print(coeff_df['Coefficient'])

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
# print(df)

from sklearn import metrics
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))


# -
# # One Hot Vector Encoding of the data
#
# In this version we use one hot vector encodings for each individual note. So the input to the regressor will be a concatenated vector of window_size times a one hot vector representing a note. From hereonout we will use ridge regression rather than the standard linear regression.

#Return the original integer note with the highest probability from the prediction. 
def one_hot_to_original_data(y_onehot,y):
    y_orig = np.empty(y_onehot.shape[0])
    for i in range(0,y_onehot.shape[0]):
        y_orig[i] = np.unique(y)[np.argmax(y_onehot[i,:])]
    return y_orig


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
# print(df)

from sklearn import metrics
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test_orig, y_pred_orig))
print('Mean Squared Error:', metrics.mean_squared_error(y_test_orig, y_pred_orig))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test_orig, y_pred_orig)))
# -
# Predict a new signal based on the last part of the signal and the window size and keep iteratively updating the signal:

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
# # Add note and duration as features to vector encoding
# We will now encode the signal in a different way namely, a note plus its duration. So now 10 repetions of the same note x will result in 1 tuple saying (x, 10). This representation thus adds duration as a feature to the encoding of the signal. The note and the duration will be represented by onehot vectors and for each pair of note plus duration in the window size these onehot vectors will be concatenated. Please note that a window size of x now no longer represent x notes, but x pairs of notes and their duration. Duration per not has a maximum value of 16, or 4 bars, because this is a natural representation in music. So if a note occurs longer than that it will be represented by multiple pairs. 

#Get one voice of the signal
signal = dataset.loc[:,2]


#convert a the raw signal to its notes and durations where durations are split up in max sizes of 16. 
def convert_to_note_and_duration(signal):
    converted_signal = []
    duration = 0
    last_note = 0
    for note in signal:
        if note is last_note and duration < 16:
            duration += 1
        else:
            converted_signal.append((last_note, duration))
            duration = 1
        last_note = note
    return converted_signal


# +
#Rewrite to note + duration
signal_rewritten = convert_to_note_and_duration(signal)

#Create slices, ie. windows of the data
X_d, y_d = create_dataset(signal_rewritten, window_size)

# +
#Reshape data such that note and duration are concatenated. 
X_d_res = X_d.reshape((X_d.shape[0],2*window_size))

#Create one hot encoding
enc_X = OneHotEncoder(sparse = False, drop=None)
enc_y = OneHotEncoder(sparse = False, drop=None)
X_d_one = enc_X.fit_transform(X_d_res)
y_d_one = enc_y.fit_transform(y_d)


# +
#Split data into train and test set
X_train, X_test, y_train, y_test = train_test_split(X_d_one, y_d_one, test_size=0.2, random_state=0, shuffle=False)

#Train a ridge regression model on training data.  
regressor = Ridge()
regressor.fit(X_train,y_train)

#Make predictions for test data 
y_pred = regressor.predict(X_test)

# +
#Convert back to integer notes by taking the note with the maximum proability. 
y_test_orig = enc_y.inverse_transform(y_test)
y_pred_orig =enc_y.inverse_transform(y_pred)

# Show The difference between actual values and predicted values
df = pd.DataFrame({'Actual_Note': y_test_orig[:,0],'Actual_Duration': y_test_orig[:,1], 
                   'Predicted_Note': y_pred_orig[:,0], 'Predicted_Duration': y_test_orig[:,1]})
# print(df)

#Show measures for notes and durations
from sklearn import metrics
print("Error measure for notes: ")
print('Mean Absolute Error:', metrics.mean_absolute_error(df['Actual_Note'], df['Predicted_Note']))
print('Mean Squared Error:', metrics.mean_squared_error(df['Actual_Note'], df['Predicted_Note']))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(df['Actual_Note'], df['Predicted_Note'])))
print('\nError measure for duration: ')
print('Mean Absolute Error:', metrics.mean_absolute_error(df['Actual_Duration'], df['Predicted_Duration']))
print('Mean Squared Error:', metrics.mean_squared_error(df['Actual_Duration'], df['Predicted_Duration']))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(df['Actual_Duration'], df['Predicted_Duration'])))


# -

def to_signal(note_duration):
    return np.repeat(note_duration[0],note_duration[1])


#Convert note and duration to an array representing the signal in terms of integers
def to_signal_long(note_duration):
    signal = []
    for pair in note_duration:
        signal = np.append(signal, to_signal(pair))
    return signal


#Predict the next x steps using the trained regressor
def make_prediction(signal, steps_to_predict=50):
    new_part = np.empty((steps_to_predict, 2))
    for i in range(0,steps_to_predict):
        window = signal[len(signal)-window_size:len(signal)]
        correct_shape = np.reshape(window, (2*window_size))
        x = enc_X.transform(correct_shape.reshape(1, -1))
        pred = regressor.predict(x)
        next_step = enc_y.inverse_transform(pred)
        signal.append((next_step[0,0], next_step[0,1]))
        new_part[i,:] = next_step
    return new_part


# +
#PLEASE NOTE:
#Running the make prediction function more than once will keep appending the signal and generating predictions on these new parts
#rather than the original signal. 

#Make prediction using the new regressor
new = make_prediction(signal_rewritten)
new_signal = to_signal_long(new)
print(new_signal)
# np.savetxt(r'new_signal.txt', new_signal, fmt='%d') #Print the new signal such that it can be converted to an audio file
# -
#Plot Comparisson of actual signal to predicted signal to get a rough visusal representation of how well the prediction is.
plt.figure(figsize=(20,12))
plt.plot(df.Actual_Note, label='Acutal Note')
plt.plot(df.Predicted_Note, label = 'predicted Note')
plt.legend()

# # Find optimal window size
# Up to this point we had been using an arbitrarily chosen window size. In order to find out which size we should use we ran a sweep through the window_sizes, ie we trained a model using one hot vector encodings of both note and durations for a window size randing from 1-100 and plot the performance in order to decide on a good window size.  

max_size = 100
performance = np.empty((max_size+1, 7))
for size in range(1,max_size):
    signal = dataset.loc[:,2]
    #Rewrite to note + duration
    signal_rewritten = convert_to_note_and_duration(signal)
    #Create slices, ie. windows of the data
    X_d, y_d = create_dataset(signal_rewritten, size)
    #Reshape data such that note and duration are concatenated. 
    X_d_res = X_d.reshape((X_d.shape[0],2*size))
    #Create one hot encoding
    enc_X = OneHotEncoder(sparse = False, drop=None)
    enc_y = OneHotEncoder(sparse = False, drop=None)
    X_d_one = enc_X.fit_transform(X_d_res)
    y_d_one = enc_y.fit_transform(y_d)
    #Split data into train and test set
    X_train, X_test, y_train, y_test = train_test_split(X_d_one, y_d_one, test_size=0.2, random_state=0, shuffle=False)
    #Train a ridge regression model on training data.  
    regressor = Ridge()
    regressor.fit(X_train,y_train)
    #Make predictions for test data 
    y_pred = regressor.predict(X_test)
    #Convert back to integer notes by taking the note with the maximum proability. 
    y_test_orig = enc_y.inverse_transform(y_test)
    y_pred_orig =enc_y.inverse_transform(y_pred)

    # Show The difference between actual values and predicted values
    df = pd.DataFrame({'Actual_Note': y_test_orig[:,0],'Actual_Duration': y_test_orig[:,1], 
                       'Predicted_Note': y_pred_orig[:,0], 'Predicted_Duration': y_test_orig[:,1]})

    #Measures for notes
    performance[size,0] = metrics.mean_absolute_error(df['Actual_Note'], df['Predicted_Note'])
    performance[size,1] = metrics.mean_squared_error(df['Actual_Note'], df['Predicted_Note'])
    performance[size,2] = np.sqrt(metrics.mean_squared_error(df['Actual_Note'], df['Predicted_Note']))
    
    #Measures for durations
    performance[size,3] = metrics.mean_absolute_error(df['Actual_Duration'], df['Predicted_Duration'])
    performance[size,4] = metrics.mean_squared_error(df['Actual_Duration'], df['Predicted_Duration'])
    performance[size,5] = np.sqrt(metrics.mean_squared_error(df['Actual_Duration'], df['Predicted_Duration']))
    
    #Corrected for window size
    performance[size,6] = metrics.mean_squared_error(df['Actual_Note'], df['Predicted_Note'])/size

#Plot Error adjusted for window size for the notes. 
plt.plot(performance[:,6])
plt.legend(("MSE_Notes"))

# So based on the plot above a windows size of 30-40 seems to be optimal.

# # Chroma Circle feature representation
# Encode notes using the chroma circle so now notes turn into 5 features rather than a single integer. Regressor is now trained on 6 feautures, namely the chroma representation of a note plus its duration.

# +
import math
#Get one voice of the signal
signal = dataset.loc[:,2]
#Rewrite to note + duration
signal_rewritten = convert_to_note_and_duration(signal)

#Create slices, ie. windows of the data
X_d, y_d = create_dataset(signal_rewritten, window_size)


# +
#Convert a note to a chroma represenation. 
#This code was based on the implementation mentioned in the thesis 'Art in Echo State Networks:Music Generation' by Aulon Kuqi

def convert_to_chroma(midi_note, min_note, max_note):
    chroma = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    radius_chroma = 1
    c5 = [1, 8, 3, 10, 5, 12, 7, 2, 9, 4, 11, 6]
    radius_c5 = 1
    note = (midi_note-55) % 12
    
    chroma_angle = (chroma[note] - 1) * (360/12);
    c5_angle = (c5[note] - 1) * (360/12);
    
    chroma_x = radius_chroma * math.sin(chroma_angle);
    chroma_y = radius_chroma * math.cos(chroma_angle);
    c5_x = radius_c5 * math.sin(c5_angle);
    c5_y = radius_c5 * math.cos(c5_angle);
    
    n = midi_note - 69;
    fx = 2**(n/12)*440;
    
    min_p = 2 * math.log(2**((min_note - 69)/12) * 440) / math.log(2);
    max_p = 2 * math.log(2**((max_note - 69)/12) * 440)/ math.log(2);

    pitch = 2 * math.log(fx)/math.log(2) - max_p + (max_p - min_p)/2;
    y = [pitch, chroma_x, chroma_y, c5_x, c5_y];
    return y


# -

#Convert each note to a chroma represenation
def rewrite_to_chroma_list(X_d, signal):
    rewritten = []
    for idx, note in enumerate(X_d):
        for i in range(0, window_size): 
            chroma_note = convert_to_chroma(X_d[idx,i,0], np.unique(signal)[1], max(signal))
            rewritten.append((chroma_note, X_d[idx,i,1]))
    return rewritten


#Rewrite the notes in chroma format and their duration to the required numpy format for one hot encoding. 
def to_chroma_format(X_d, rewritten):   
    #First get notes and duration seperate from each other
    X_d_notes= np.empty((X_d.shape[0], window_size, 5))
    X_d_duration = np.empty((X_d.shape[0], window_size))
    
   #Create numpy arrays for the notes and the according durations
    for idx in range(0, X_d.shape[0]): #n_samples
        for i in range(0, window_size): 
            note_list = rewritten[idx*window_size+i][0]
            duration = rewritten[idx*window_size+i][1]
            X_d_notes[idx,i,:] = note_list
            X_d_duration[idx,i] = duration
    
    #Get everythin in numpy array in correct format for one hot encoding
    X_d_np = np.empty((X_d.shape[0], window_size*6))
    for idx in range(0, X_d.shape[0]): #n_samples
        for i in range(0, window_size): 
            tmp = np.array(X_d_notes[idx,i,:])
            tmp = np.append(tmp, X_d_duration[idx,i])
            X_d_np[idx, i*6:i*6+6]=tmp
    return X_d_np


# +
#Get signal in the required format. 
rewritten = rewrite_to_chroma_list(X_d, signal)
X_d_np = to_chroma_format(X_d, rewritten)

#Create one hot encoding
enc_X = OneHotEncoder(sparse = False, drop=None)
enc_y = OneHotEncoder(sparse = False, drop=None)
X_d_one = enc_X.fit_transform(X_d_np)
y_d_one = enc_y.fit_transform(y_d)

# +
#Split data into train and test set
X_train, X_test, y_train, y_test = train_test_split(X_d_one, y_d_one, test_size=0.2, random_state=0, shuffle=False)
#Train a ridge regression model on training data.  
regressor = Ridge(alpha=1) #A paramtersweep for the value of alpha is executed later on.
regressor.fit(X_train,y_train)
#Make predictions for test data 
y_pred = regressor.predict(X_test)
#Convert back to integer notes by taking the note with the maximum proability. 
y_test_orig = enc_y.inverse_transform(y_test)
y_pred_orig =enc_y.inverse_transform(y_pred)

# Show The difference between actual values and predicted values
df = pd.DataFrame({'Actual_Note': y_test_orig[:,0],'Actual_Duration': y_test_orig[:,1], 
                   'Predicted_Note': y_pred_orig[:,0], 'Predicted_Duration': y_test_orig[:,1]})

#Show measures for notes and durations
from sklearn import metrics
print("Error measure for notes: ")
print('Mean Absolute Error:', metrics.mean_absolute_error(df['Actual_Note'], df['Predicted_Note']))
print('Mean Squared Error:', metrics.mean_squared_error(df['Actual_Note'], df['Predicted_Note']))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(df['Actual_Note'], df['Predicted_Note'])))
print('\nError measure for duration: ')
print('Mean Absolute Error:', metrics.mean_absolute_error(df['Actual_Duration'], df['Predicted_Duration']))
print('Mean Squared Error:', metrics.mean_squared_error(df['Actual_Duration'], df['Predicted_Duration']))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(df['Actual_Duration'], df['Predicted_Duration'])))


# -

#Predict the next x steps using the trained regressor
def make_prediction_chroma(signal, original_signal, steps_to_predict=50):
    new_part = np.empty((steps_to_predict, 2))
    for i in range(0,steps_to_predict):
        window = signal[len(signal)-window_size:len(signal)]
        #Get list of chroma notes
        rewritten = []
        for idx, pair in enumerate(window):
            note = window[idx][0]
            dur = window[idx][1]
            chroma_note = convert_to_chroma(note, np.unique(original_signal)[1], max(original_signal))
            rewritten.append((chroma_note, dur))
        #Get the corect format as used for one hot encoder
        chroma = to_chroma_format(np.empty((1,240)), rewritten)
        x = enc_X.transform(chroma)
        pred = regressor.predict(x)
        next_step = enc_y.inverse_transform(pred)
        signal.append((next_step[0,0], next_step[0,1]))
        new_part[i,:] = next_step
    return new_part


# +
#Make prediction using the new regressor
new = make_prediction_chroma(signal_rewritten, signal, 100)
new_signal = to_signal_long(new)

print(new_signal)
np.savetxt(r'chroma_prediction.txt', new_signal, fmt='%d') #Save the signal predict using chroma representation to create an audio file
# -
#Plot Comparisson of actual signal to predicted signal to get a rough visusal representation of how well the prediction is.
plt.figure(figsize=(20,12))
plt.plot(df.Actual_Note, label='Acutal Note')
plt.plot(df.Predicted_Note, label = 'predicted Note')
plt.legend()

# # Parameter search for optimal alpha
# Without specifying alpha in the ridge() function, the defautl will be 1. This need not be te best value, so we search for the optimal value for alpha using a Gridsearch. The optimal value turns out to be 1500 for our current model. 

# +
from sklearn import svm, datasets
from sklearn.model_selection import GridSearchCV

parameters = {'alpha':[0.5,1,2,3,4,5,7.5,10,15,20,25,30,35,40,50,100,200,500,750,1000,1250,1500,2000,5000,10000]}
regressor = Ridge()
clf = GridSearchCV(regressor, parameters, scoring ='r2')
clf.fit(X_train,y_train)
GridSearchCV(estimator=Ridge(), param_grid={'alpha': [0.5,1,2,3,4,5], 'kernel': ('linear', 'rbf')})
# print(sorted(clf.cv_results_))

result_frame = pd.DataFrame.from_dict(clf.cv_results_)
print(result_frame)
print(clf.best_estimator_)

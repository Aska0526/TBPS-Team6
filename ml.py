# -*- coding: utf-8 -*-

import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize

#%%

cwd = os.getcwd()  # Get the current working directory (cwd)
files = os.listdir(cwd)  # Get all the files in that directory
print("Files in %r: %s" % (cwd, files))

#%% load the total dataset

# IMPORTANT: this is the unfiltered dataset, we should use the filtered dataset

file_name = 'total_dataset.pkl'
total = pd.read_pickle(os.getcwd() + f'/year3-problem-solving' + f'/{file_name}')

#%% print column names

print(total.columns.values.tolist())

#%% plot B0_MM column for inspection

plt.hist(total["B0_MM"], bins=100)

#%% remove all rows from the total dataset that have B0_MM > 5540

threshold = 5380

noise = total[total["B0_MM"] > threshold]

#%% insert a column called signal_or_noise

# 1 is signal 0 is noise

noise["signal_or_noise"] = 0

#%% plot to check the noise data

plt.hist(noise["B0_MM"], bins=100)

#%% load the signal dataset

file_name = 'signal.pkl'
signal = pd.read_pickle(os.getcwd() + f'/year3-problem-solving' + f'/{file_name}')

#%%

print(signal.columns.values.tolist())

#%%

signal["signal_or_noise"] = 1

#%%

plt.hist(signal["B0_MM"], bins=100)

#%% mix the noise with the signal dataset

# we do this by concatenating the noise and signal datasets

frames  = [signal, noise]

concat = pd.concat(frames, axis=0, join="outer")

#%% let's plot again the B0_MM column

plt.hist(concat["B0_MM"], bins=100)
plt.ylabel('Count')
plt.xlabel('B0 mass (Mev/$c^2$)')

#%%

counts, bins = np.histogram(noise['B0_MM'], bins=100)

#%%

x = bins[:-1] # redefine the x axis, there is one more bin than counts

plt.plot(x, counts)

#%%

def expo(x, a, tau, b, c):
    return a * np.exp(-tau * (x-b)) + c

#%%

# some scaling to help fitting

#x /= (max(total["B0_MM"])  - min(total["B0_MM"]))

#counts = np.array(counts, dtype=np.float64)

#counts /= 7500 # estimate

#%%

# smoothing out our data

window_size = 11

i = 0
moving_averages = []

while i < len(x) - window_size + 1:
    this_window = counts[i : i + window_size]
    window_average = sum(this_window) / window_size
    moving_averages.append(window_average)
    i += 1

print(moving_averages)
    
moving_x = np.linspace(min(x), max(x), len(moving_averages))
    
plt.plot(moving_x, moving_averages)

#%%

p0 = [10000, 1200, 5200, 400]
popt = scipy.optimize.curve_fit(expo, moving_x, moving_averages)

"""
First of all without scaling, it is nearly impossible to fit an exponential curve
because the x axis is so massive.
Second of all even after smoothing out the function, we cannot fit a curve well
because we have too few data points.
Conclusion: fitting an exponential curve and then predicting the combinatorial background
is unrealistic
"""

#%%

import tensorflow as tf
from tensorflow import keras
from keras.datasets import mnist
import matplotlib.pyplot as plt
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, Flatten, MaxPooling1D, Dropout, BatchNormalization, Rescaling, RandomFlip, RandomRotation
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import MinMaxScaler
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
import pickle
import numpy as np

#%% drop the first row because we have odd number of samples

concat.drop(concat.tail(1).index,inplace=True)

#%%

X = concat['B0_MM']

y = concat['signal_or_noise']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.5, random_state = 42) 

#%%

X_train = np.array(X_train)
X_test = np.array(X_test)
y_train = np.array(y_train)
y_test = np.array(y_test)

#%%

# reshape data to fit model
X_train = X_train.reshape(len(X_train), 1)
X_test = X_test.reshape(len(X_test), 1) 

# scale X
cs = MinMaxScaler()
X_train = cs.fit_transform(X_train)
X_test = cs.transform(X_test)

#%%

X_train = X_train.reshape(len(X_train), -1, 1) # number of examples, timesteps, feature
X_test = X_test.reshape(len(X_test), -1, 1) 

#%%

y_train = to_categorical(y_train, 2) # 2 classes: 1 or 0
y_test = to_categorical(y_test, 2)

#%%

# create model
model = Sequential() # build the model layer by layer

# add model layers
# model.add(BatchNormalization())
model.add(Conv1D(32, kernel_size=2, activation='relu', input_shape=(1, 1), padding='same'))
model.add(BatchNormalization())
#model.add(MaxPooling1D((2)))
"""
model.add(Conv1D(64, kernel_size=2, activation='relu', padding='same'))
model.add(MaxPooling1D((2)))
model.add(BatchNormalization())
model.add(Conv1D(64, kernel_size=2, activation='relu', padding='same'))
model.add(BatchNormalization())
"""
# connection the convolution and dense layers
model.add(Flatten())
model.add(Dense(64, activation='relu'))
# randomly drop neurons to improve accuracy
model.add(Dropout(0.5))
# 10 nodes in the output layer
# softmax makes the probability sums up to 1, prediction is based on the option with highest probability
# model.add(BatchNormalization())
model.add(Dense(2, activation='softmax'))

#%%

model.summary()

#%%

#y_train = y_train.reshape(1, 140343)
#y_test = y_test.reshape(1, 140343)

#%%
# compile model using accuracy to measure model performance
# optimizer controls learning rate, adam is self-tuning
# categorical_crossentropy is best for classification
model.compile(optimizer=Adam(learning_rate=1e-3), loss="categorical_crossentropy", metrics=['accuracy'])

#%%

# train the model
# eopchs is how many times we cycle throughh the data
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10)

#%%

#predict first 4 images in the test set
print(model.predict(X[:10]))
#actual results for first 4 images in test set
print(y_test[:10])

#%%

ds_to_clean = np.array(total["B0_MM"])

ds_to_clean = ds_to_clean.reshape(len(ds_to_clean), 1)

ds_to_clean = cs.transform(ds_to_clean)

ds_to_clean = ds_to_clean.reshape(len(ds_to_clean), -1, 1)

#%%

prediction = model.predict(ds_to_clean)

#%% converting probability back to binary numbers

prediction = np.argmax(prediction, axis=-1)

#%%

print(prediction[-100:])

#%% final results stored in a list (probably could be written in a better way)

final = []

for i in range(len(prediction)):
    if prediction[i] == 1:
        final.append(total["B0_MM"].iloc[i])

#%% CNN filtered dataset and original unfiltered total dataset (what we were given initially)

plt.hist(total["B0_MM"], bins=100, alpha=0.4, label="Completely unfiltered total dataset")
plt.hist(final, bins=100, label="Mixed dataset after CNN")
plt.legend()
plt.ylabel('Count')
plt.xlabel('B0 mass (Mev/$c^2$)')

"""
Currently we have only experimented with 1 parameter (B0_MM) as the input
We predict the model's prediction will continue improve after including more parameters
"""

#%% CNN filtered dataset and pure signal dataset (what we want to obtain eventually)

plt.hist(signal["B0_MM"], bins=100, label="Pure signal dataset", alpha=0.8)
plt.hist(final, bins=100, label="Mixed dataset after CNN", alpha=0.4)
plt.legend()
plt.ylabel('Count')
plt.xlabel('B0 mass (Mev/$c^2$)')

"""
Here we see that some signal that we wish to keep is mistakenly removed by the CNN model
It might not be adequate to address our problem as binary classification
because there are regions with signal only, noise only and signal + noise (ie mass between 5200 and 5400)???
"""


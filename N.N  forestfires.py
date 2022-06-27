# -*- coding: utf-8 -*-
"""
Created on Wed Jun  1 12:16:16 2022

@author: lalith kumar
"""
# PREDICTING THE BURNED AREA OF FOREST FIRES WITH NEURAL NETWORKS.

from tensorflow.keras.layers import Dense

from tensorflow.keras.models import Sequential
# importing dataset.

import pandas as pd
df = pd.read_csv('E:\\data science\ASSIGNMENTS\\ASSIGNMENTS\\N.N\\forestfires.csv')
df.head()
df.shape
df.info()
df.describe()
list(df)

# label encoding
from sklearn.preprocessing import LabelEncoder
LE = LabelEncoder()
df['month'] = LE.fit_transform(df['month'])
df['month'].value_counts()

from sklearn.preprocessing import LabelEncoder
LE = LabelEncoder()
df['day'] = LE.fit_transform(df['day'])
df['day'].value_counts()

from sklearn.preprocessing import LabelEncoder
LE = LabelEncoder()
df['size_category'] = LE.fit_transform(df['size_category'])
df['size_category'].value_counts()

# split x&y variables

X = df.iloc[:,0:29]
y = df.iloc[:,30]
list(X)

# create model
model = Sequential()
model.add(Dense(40, input_dim=29,  activation='relu')) #input layer
model.add(Dense(1, activation='sigmoid')) #output layer
#model.add(Dense(8,  activation='relu')) #2nd layer
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])


# Fit the model
history = model.fit(X, y, validation_split=0.25, epochs=250, batch_size=10)

# evaluate the model
scores = model.evaluate(X, y)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

# output:-
# loss: 0.1388 - accuracy: 0.9903
# accuracy: 99.03%

# list all data in history
history.history.keys()


# summarize history for accuracy
import matplotlib.pyplot as plt
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

#======================================================================================



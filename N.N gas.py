# -*- coding: utf-8 -*-
"""
Created on Wed Jun  1 12:52:26 2022

@author: lalith kumar
"""
# predicting turbine energy yield (TEY) using ambient variables as features.

from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential

# importing dataset.
import pandas as pd
df = pd.read_csv('E:\\data science\ASSIGNMENTS\\ASSIGNMENTS\\N.N\\gas_turbines.csv')
df.head()
df.shape
df.info()
df.describe()
list(df)
   
# split x&y variables

X = df.drop(['TEY'],axis=1)
y = df['TEY']
list(X)

# create model
model = Sequential()
model.add(Dense(15, input_dim=10,  activation='relu')) #input layer
model.add(Dense(1, activation='relu')) #output layer
#model.add(Dense(8,  activation='relu')) #2nd layer
model.compile(loss='msle', optimizer='adam', metrics=['msle'])

# Fit the model
history = model.fit(X, y, validation_split=0.33, epochs=500, batch_size=150)

# evaluate the model
scores = model.evaluate(X, y)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

# output
# loss: 1.0931 - accuracy: 0.0000e+00

# summarize history for accuracy
import matplotlib.pyplot as plt
plt.plot(history.history['msle'])
plt.plot(history.history['val_msle'])
plt.title('msle')
plt.ylabel('msle')
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

#---------------------------------------------------------------------------------
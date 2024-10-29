#from tkinter import *
#from tkinter import ttk
#import tkinter as tk
# mlp for multiclass classification

from matplotlib import pyplot
from numpy import argmax
import pandas as pd
from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import plot_model
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Dropout
from tensorflow.keras import layers, regularizers
import numpy as np
import tensorflow as tf




def model():
 read_file = pd.read_excel("cleaned_data_telecom.xlsx")    # чете файла във формат rxcel
 read_file.to_csv("Test.csv",                   # обръща файла от формат xlsx във csv
                 index=None,
                 header=True)

 df = pd.DataFrame(pd.read_csv("Test.csv"))                 # зарежда csv файла като data frame
 df1 = df.iloc[1:]                                          # пропуска първия ред от файла
# split into input and output columns
 X, y = df1.values[:, :-1], df1.values[:,-1]               # на X присвоява първите стълбoве от матрицата с данните, това са параметрите
                                                          # на y присвоява последния стълб от матрицата с данните (това е резултaта)

 X1 = LabelEncoder().fit_transform(X[:,0])
 X[:,0] = X1
 X1 = LabelEncoder().fit_transform(X[:,1])
 X[:,1] = X1
 X1 = LabelEncoder().fit_transform(X[:,2])
 X[:,2] = X1
 X1 = LabelEncoder().fit_transform(X[:,3])
 X[:,3] = X1
 X1 = LabelEncoder().fit_transform(X[:,4])
 X1 = (X1-np.min(X1))/(np.max(X1)-np.min(X1))
 X[:,4] = X1
 X1 = LabelEncoder().fit_transform(X[:,5])
 X1 = (X1-np.min(X1))/(np.max(X1)-np.min(X1))
 X[:,5] = X1
 X1 = LabelEncoder().fit_transform(X[:,6])
 X1 = (X1-np.min(X1))/(np.max(X1)-np.min(X1))
 X[:,6] = X1
 X1 = LabelEncoder().fit_transform(X[:,7])
 X1 = (X1-np.min(X1))/(np.max(X1)-np.min(X1))
 X[:,7] = X1
 X1 = LabelEncoder().fit_transform(X[:,8])
 X1 = (X1-np.min(X1))/(np.max(X1)-np.min(X1))
 X[:,8] = X1
 X1 = LabelEncoder().fit_transform(X[:,9])
 #X1 = (X1-np.min(X1))/(np.max(X1)-np.min(X1))
 X[:,9] = X1
 X1 = LabelEncoder().fit_transform(X[:,10])
 X1 = (X1-np.min(X1))/(np.max(X1)-np.min(X1))
 X[:,10] = X1
 X1 = LabelEncoder().fit_transform(X[:,11])
 X1 = (X1-np.min(X1))/(np.max(X1)-np.min(X1))
 X[:,11] = X1
 X1 = LabelEncoder().fit_transform(X[:,12])
 X1 = (X1-np.min(X1))/(np.max(X1)-np.min(X1))
 X[:,12] = X1

 X1 = LabelEncoder().fit_transform(X[:,13])
 X[:,13] = X1

 X1 = LabelEncoder().fit_transform(X[:,14])
 X1 = (X1-np.min(X1))/(np.max(X1)-np.min(X1))
 X[:,14] = X1

 X1 = X[:,15]
 X1 = X1.astype('float32')
 X1 = (X1-np.min(X1))/(np.max(X1)-np.min(X1))
 X[:,15] = X1

 X1 = X[:,16]
 X1 = X1.astype('float32')
 X1 = (X1-np.min(X1))/(np.max(X1)-np.min(X1))
 X[:,16] = X1

 X1 = X[:,17]
 X1 = X1.astype('float32')
 X1 = (X1-np.min(X1))/(np.max(X1)-np.min(X1))
 X[:,17] = X1

 print(X)
 X = X.astype('float32')
 #print(X[:1])
# print(X[:2])
 #print(X[:3])
 print(X[:6])
#df11 = pd.DataFrame(X)
#print (df11)
#df11.to_csv('haha.csv')
 y = LabelEncoder().fit_transform(y)
 print(y)

 X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.33)
 print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

 n_features = X_train.shape[1]


 model = Sequential()
 model.add(Dense(304, activation='gelu', kernel_initializer='he_normal', input_shape=(n_features,),kernel_regularizer=regularizers.L2(0.01)))
# model.add(BatchNormalization())
#model.add(Dropout(0.5))
 model.add(Dense(152, activation='gelu', kernel_initializer='he_normal',kernel_regularizer=regularizers.L2(0.01) ))
 #model.add(BatchNormalization())
 model.add(Dense(76, activation='gelu', kernel_initializer='he_normal',kernel_regularizer=regularizers.L2(0.01)))
 #model.add(BatchNormalization())
#model.add(Dropout(0.5))
 model.add(Dense(38, activation='gelu', kernel_initializer='he_normal',kernel_regularizer=regularizers.L2(0.01)))
 #model.add(BatchNormalization())
 model.add(Dense(19, activation='gelu', kernel_initializer='he_normal',kernel_regularizer=regularizers.L2(0.01)))
 #model.add(BatchNormalization())
 model.add(Dense(8, activation='gelu', kernel_initializer='he_normal',kernel_regularizer=regularizers.L2(0.01)))
 #model.add(BatchNormalization())
 model.add(Dense(4, activation='gelu', kernel_initializer='he_normal',kernel_regularizer=regularizers.L2(0.01)))
 #model.add(BatchNormalization())
 model.add(Dense(2, activation='softmax'))

 model.compile(optimizer='adam',  loss='sparse_categorical_crossentropy', metrics=['accuracy'])

 history=model.fit(X_train, y_train, epochs=1024, batch_size=256, verbose=1,validation_split=0.44)
 model.save('model.h5')
 loss, acc = model.evaluate(X_test, y_test, verbose=2)
 print('Test Accuracy: %.3f' % acc)
#TOCHNOST.delete(0, tk.END)
#TOCHNOST.insert(0, acc)
 pyplot.title('Learning Curves')
 pyplot.xlabel('Epoch')
 pyplot.ylabel('Cross Entropy')
 pyplot.plot(history.history['loss'], label='train')
#pyplot.plot(history.history['accuracy'], label='train')
 pyplot.plot(history.history['val_loss'], label='val')
#pyplot.plot(history.history['val_accuracy'], label='val')

 pyplot.legend()
 pyplot.show()

def stt():

 model = load_model('model.h5')
 #row = ([1., 0., 0., 0., 0.5, 0., 0.,1.,0.,0.,0.,0.,0.,1.,0.6666667,0.11542289,0.0012751,0.,0.]) #ne
 #row = ([0., 1., 0., 1., 0., 0., 1., 0., 1., 0., 0., 0., 0.5, 0., 1., 0.38507465, 0.2158666, 0.46478873, 0.])# ne
 #row = ([0., 1., 0., 1., 0., 0., 1., 1., 0., 0., 0., 0., 0., 1., 1., 0.35422885, 0.01031041, 0.01408451, 0.]) # da
 #row = ([0., 1., 0., 0., 0.5, 0., 1., 0., 1., 1., 0., 0., 0.5, 0., 0., 0.23930347, 0.21024117, 0.6197183, 0.]) # ne
 row = ([0.0, 0.0, 0.0, 1.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.6666667, 0.5218905, 0.01533002, 0.01408451, 0.])  # da
 row =([0.,0.,0.,1.,1.,0.5,0.,0.,1.,0.,1.,1.,0.,1.,0.6666667,0.8099503,0.09251096,0.09859155,0.]) # da

 yhat = model.predict([row])
 yyy=( (yhat, argmax(yhat)))
 print('Predicted: %s (class=%d)' % (yhat, argmax(yhat)))
 print(yhat[:, argmax(yhat)])

model()
stt()
#%%
import this
import os
import io
import ast
import itertools
import collections
from time import time
from tqdm import tqdm

from itertools import cycle
import pickle
import datetime
import json

import numpy as np 
import pandas as pd

import random as rn
from random import randint


from collections import Counter 

from sklearn.model_selection import ShuffleSplit, train_test_split, StratifiedKFold, StratifiedShuffleSplit, KFold 
from sklearn.preprocessing import StandardScaler, Normalizer, MinMaxScaler
from sklearn.preprocessing import MultiLabelBinarizer, FunctionTransformer, LabelBinarizer
from sklearn.metrics import confusion_matrix, classification_report, f1_score, recall_score, precision_score

from imblearn.under_sampling import NearMiss, CondensedNearestNeighbour, OneSidedSelection


from sklearn import decomposition

from sklearn.feature_selection import SelectKBest
from sklearn.pipeline import Pipeline

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import regularizers
from tensorflow.keras import initializers
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras import layers, metrics
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Concatenate
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Conv1D, MaxPooling1D
from tensorflow.keras.models import model_from_json, load_model
from tensorflow.keras.regularizers import *
from tensorflow.keras.callbacks import CSVLogger
# from tensorflow.keras import backend as K

import matplotlib.pyplot as plt # for making plots
import seaborn as sns
sns.set(context="paper",
        style="whitegrid",
        palette="deep",
        font_scale=2.0,
        color_codes=True,
        rc=None)
# %matplotlib inline
plt.rcParams["figure.figsize"] = [6,4]


#%%
df = pd.read_csv("C:\Mannu\Projects\Sporozoite Spectra for An funestus s.s\Data\sporozoite_full.csv")
df.head()

#%%
# data shape
print(df.shape)

# Checking class distribution abd correlation in the data
Counter(df["Sporozoite"])


#%%
# Select vector of labels and matrix of features

X = df.iloc[:,7:] # matrix of features
y = df["Sporozoite"] # vector of labels
X

#%%
# rescalling the data (undersampling the over respresented class - negative class)

pca_pipe = Pipeline([('scaler', StandardScaler()),
                      ('pca',  decomposition.KernelPCA(n_components = 8, kernel = 'linear'))])


# convert all inputs into array format for simplicity

X = np.asarray(X)
y = np.asarray(y)
print('y labels : {}'.format(np.unique(y)))


rus = NearMiss(version = 1, n_neighbors = 3)
X_res, y_res = rus.fit_resample(X, y)
y_res_count = collections.Counter(y_res)
print(y_res_count)

y_list = LabelBinarizer().fit_transform(y_res)
# y_res_2 = keras.utils.to_categorical(y_list)


# %%

validation_size = 0.2
seed = 42

# y_list_2 = keras.utils.to_categorical(y_list)
X_train, X_val, y_train, y_val = train_test_split(X_res, y_list, test_size = validation_size, random_state = seed)

X_train = X_train.reshape([X_train.shape[0], -1])
X_val = X_val.reshape([X_val.shape[0], -1])

X_train = pca_pipe.fit_transform(X_train)
X_val = pca_pipe.fit_transform(X_val)

#%%


dim = X_train.shape[1]

# parameter rate for l2 regularization
regConst = 0.01
    
# defining a stochastic gradient boosting optimizer
sgd = tf.keras.optimizers.SGD(lr = 0.001, momentum = 0.9, 
                                    nesterov = True, clipnorm = 1.)
    
# define categorical_crossentrophy as the loss function (multi-class problem i.e. 3 age classes)
    
cce = 'categorical_crossentropy'
bce = 'binary_crossentropy'


#Design the deep neural network [Small + 1 layer]
model = Sequential()
model.add(tf.keras.layers.Dense(500, input_dim = dim, activation = "relu"))
model.add(tf.keras.layers.Dense(500, activation = "relu"))
model.add(tf.keras.layers.Dense(500, activation = "relu"))
model.add(tf.keras.layers.Dense(500, activation = "relu"))
model.add(tf.keras.layers.Dense(500, activation = "relu"))
model.add(tf.keras.layers.Dense(1, activation = "sigmoid")) #activation = softmax for multiclass classification

model.compile(optimizer = sgd, loss = bce, 
                metrics=['accuracy'])
model.summary()


model.fit(X_train, y_train, validation_data = (X_val, y_val), 
        epochs = 1000, batch_size = 64)



# %%

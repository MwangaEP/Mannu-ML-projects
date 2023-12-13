#%%
import os
import io
import ast
import json
import itertools
import collections
from time import time
from tqdm import tqdm

from itertools import cycle
import pickle
import random as rn
import datetime

import numpy as np 
import pandas as pd

# from random import randint
from scipy.stats import uniform, randint
from collections import Counter 

from sklearn.model_selection import ShuffleSplit, train_test_split, StratifiedKFold, StratifiedShuffleSplit, KFold 
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, f1_score, recall_score, precision_score, precision_recall_fscore_support

from sklearn import decomposition
from sklearn.cluster import k_means, KMeans

from imblearn.under_sampling import RandomUnderSampler

import matplotlib.pyplot as plt # for making plots
import seaborn as sns
sns.set(context="paper",
        style="white",
        palette="deep",
        font_scale=2.0,
        color_codes=True,
        rc=None)
%matplotlib inline
plt.rcParams["figure.figsize"] = [6, 4]


#%%
# importing dataframe 

# Loading dataset  
# Upload infection data

par_df = pd.read_csv("C:\Mannu\Projects\Mwanga-DBS work\Parasite age\Transformed\Age_spectra_raw_clean.dat", delimiter = '\t')
print(par_df.head())

print(par_df.shape)

# Checking class distribution in the data
print(Counter(par_df["Cat1"]))

# drops columns of no interest
# train_data = [par_age_df].drop(['Unnamed: 0'], axis = 1)
par_df.head(10)


#%%

# rename factors in variable (Cat1) to make it more informative

Treatment = []

for row in par_df['Cat1']:

    if row == 'EC':
        Treatment.append('Early control')
    
    elif row == 'LC':
        Treatment.append('Late control')
    
    elif row == 'ET':
        Treatment.append('Early rings')

    else:
        Treatment.append('Late rings')

# print(Age)

par_df['Treatment'] = Treatment

# drop the column with age as a string and keep the age in intergers

par_df_2 = par_df.drop(['Cat1', 'Cat2', 'Cat3', 'Cat5'], axis = 1) 
par_df_2.head(5)

# Checking class distribution in the data
print(Counter(par_df_2["Treatment"]))

#%%
# Drop control from the data and classify only
 
par_df_2 = par_df_2.loc[par_df_2['Treatment'] != 'Early control']
par_df_2 = par_df_2.loc[par_df_2['Treatment'] != 'Early rings']

print(Counter(par_df_2["Treatment"])) # count the number of levels in the column and their size


# Since the resolution of the spectra data is 2cm, 2300 may not be available in the colummn names, 
# Check whats the closest number

# drop treatment column temporarly
temp_df = par_df_2.drop(['Treatment'], axis = 1)

# make a list containing all column names
col_names = temp_df.columns.tolist()

col_names = [int(x) for x in col_names]

# get the closest wavenumbers
print(list(map(lambda y:min(col_names, key=lambda x:abs(x - y)), [2400])))
print(list(map(lambda y:min(col_names, key=lambda x:abs(x - y)), [1720])))

# set the start and end column names as integers
start_col = 2401 #2301
end_col = 1721 #1801

# get the column names between the start and end column
cols_to_drop = [str(i) for i in range(start_col, end_col - 1, -2)]

par_df_2 = par_df_2.drop(cols_to_drop, axis=1)

par_df_2.head()

#%%

# get the closest wavenumbers to 3855 and 3700

print(list(map(lambda y:min(col_names, key=lambda x:abs(x - y)), [3855])))
print(list(map(lambda y:min(col_names, key=lambda x:abs(x - y)), [3700])))

# set the start and end column names as integers
start_col_2 = 3855
end_col_2 = 3701

# get the column names between the start and end column
cols_to_drop_2 = [str(i) for i in range(start_col_2, end_col_2 - 1, -2)]

par_df_2 = par_df_2.drop(cols_to_drop_2, axis=1)

par_df_2.head()

#%%

# define X (matrix of features) and y (list of labels)

X = par_df_2.iloc[:,:-1] # select all columns except the first one 
y = par_df_2["Treatment"]

print('Original dataset shape %s' % Counter(y))
rus = RandomUnderSampler(random_state = 42)
X_res, y_res = rus.fit_resample(X, y)
print('Resampled dataset shape %s' % Counter(y_res))

print('shape of X resampled : {}'.format(X_res.shape))
print('shape of y resampled : {}'.format(y_res.shape))

#%%

# scale data
scaler = StandardScaler().fit(X = X_res)
scl_features = scaler.transform(X = X_res)

#%%
# Perform PCA

balanced_df = pd.concat([X_res, y_res], axis = 1)


pca = decomposition.PCA(n_components = 3)
X_pca = pca.fit_transform(scl_features)

# pca.explained_variance_ratio_[:15]

from sklearn.preprocessing import LabelEncoder

encoder = LabelEncoder()
numerical_labels = encoder.fit_transform(y_res)

# Plot PCA results
plot = plt.scatter(X_pca[:, 0], X_pca[:, 1], c = numerical_labels)
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.legend()
plt.title('PCA with label coloring')
plt.show()
# handles=plot.legend_elements()[1], labels=list(balanced_df['Treatment'])

# %%

plt.figure(figsize = (8,4))
sns.lineplot(np.arange(1, 8, step = 1), pca.explained_variance_ratio_[:7].cumsum(), marker = 'o', zorder= 2, linestyle = '--')
plt.xlabel('Number of components')
plt.ylabel('Commulative explained variance')


# %%

# The graph shows the amount of variance captured (on the y-axis) depending on the number
# of components we include (the x-axis). A rule of thumb is to preserve around 97 % of the variance.
# So, in this instance, we decide to keep 3 components.

pca = decomposition.PCA(n_components = 3)
pca_scores = pca.fit_transform(scl_features)

#%%

wcss = []

for i in range(1, 7):
    kmeans_pca = KMeans(n_clusters = i, init = 'k-means++', random_state = 42)
    kmeans_pca.fit(pca_scores)
    wcss.append(kmeans_pca.inertia_)

#%%

plt.figure(figsize = (8,4))
sns.lineplot(range(1, 7), wcss, marker = 'o', zorder= 2, linestyle = '--')
plt.xlabel('K-means with PCA clustering')
plt.ylabel('WCSS')

# %%

kmeans_pca = KMeans(n_clusters = 2, init = 'k-means++', random_state = 42)
kmeans_pca.fit(pca_scores)

#%%

balanced_df = pd.concat([X_res, y_res], axis = 1)
pca_kmeans_df = pd.concat([balanced_df.reset_index(drop = True), pd.DataFrame(pca_scores)], axis = 1)
pca_kmeans_df.columns.values[-3:] = ['Component 1', 'Component 2', 'Component 3']

# The last column we add the PCA clusters

pca_kmeans_df['segment_kmeans_PCA'] = kmeans_pca.labels_
pca_kmeans_df 

# %%

plt.figure(figsize = (10, 8))
sns.scatterplot(pca_kmeans_df['Component 2'],pca_kmeans_df['Component 1'], palette = ['r', 'g'])
# %%

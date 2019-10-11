#%%
# This program is built to distinguish between positive and negative malaria infection from 
# the human dried blood spot (DBS)

# The analysis script was adapted from https://github.com/SimonAB/Gonzalez-Jimenez_MIRS; and has been changed to accommodate this specific analysis 

import this

# Importing all packages that my be needed for the analysis
import os
import ast
import itertools
import collections
from time import time
from tqdm import tqdm 

import numpy as np 
import pandas as pd  
import scipy.stats as stats
import statsmodels.api as sm
import statsmodels.formula.api as smf

from random import randint
from collections import Counter 

import pickle

from sklearn.model_selection import KFold
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix, precision_recall_fscore_support, mean_squared_error, r2_score

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import Normalizer

from imblearn.ensemble import EasyEnsemble

from sklearn.linear_model import LogisticRegressionCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest

from sklearn.pipeline import Pipeline
from sklearn.pipeline import FeatureUnion

import xgboost as xgb
from xgboost import XGBClassifier

import matplotlib.pyplot as plt # for making plots
import seaborn as sns
sns.set(context="paper",
        style="whitegrid",
        palette="deep",
        font_scale=1.6,
        color_codes=True,
        rc=None)
%matplotlib inline
plt.rcParams["figure.figsize"] = [8,6]


#%%
# Defining FUNCTIONS (Report classification report) 
def cv_report(results, n_top=3):
    """
    Report classification accuracy
    """
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        for candidate in candidates:
            print("Model with rank: {0}".format(i))
            print("Mean validation score: {0:.3%} ± {1:.3%}".format(
                results['mean_test_score'][candidate],
                results['std_test_score'][candidate]))
            print("Parameters: {0}".format(results['params'][candidate]))
            print("")



# # define a convenient plotting function
def plot_confusion_matrix(cm, classes,
                          normalise=True,
                          text=False,
                          title='Confusion matrix',
                          xrotation=0,
                          yrotation=0,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalisation can be applied by setting 'normalise=True'.
    """

    if normalise:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        title = "{0} (normalised)".format(title)
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix')

    # print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=xrotation)
    plt.yticks(tick_marks, classes, rotation=yrotation)

    if text:
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]),
                                      range(cm.shape[1])):
            plt.text(j, i, "{0:.2f}".format(cm[i, j]), horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label', weight = "bold", color = "red")
    plt.xlabel('Predicted label', weight = "bold", color = "red")

#%%

df = pd.read_csv("D:\OneDrive\Documents\MANNU - Spectra data\DBS for measuring malaria prevalence\Round 2 PCR\Atmos smoothed\PCR vs MIRS + ML - 80 - 20\ML analysis\Cleaned datamosquitos_mwanga.csv")
print(df.head())
df.describe()


#%%
df.shape # shape of the data

#%%
# Checking class distribution
class_counts = df.groupby('Status').size()
class_counts

#%%

df = pd.read_csv("D:\OneDrive\Documents\MANNU - Spectra data\DBS for measuring malaria prevalence\Round 2 PCR\Atmos smoothed\PCR vs MIRS + ML - 80 - 20\ML analysis\Cleaned datamosquitos_mwanga.csv")
X = df.iloc[:,4:] # defining X matrix
y = df["Status"] # defining Y matrix
X

#%%
# Now splitting 20% validation set from the the wholedataset
seed = 4
rs = ShuffleSplit(n_splits=7, test_size=0.2, random_state=seed)
rs.get_n_splits(X)

print(rs)

for train_index, val_index in rs.split(X):
   print("TRAIN:", train_index, "VALIDATION:", val_index)

print(train_index.shape, val_index.shape)

#%%
# Saving a training split to the disk
df.iloc[train_index,:]
training_DBS = df.iloc[train_index,:]
training_DBS.to_csv("D:\OneDrive\Documents\MANNU - Spectra data\DBS for measuring malaria prevalence\Round 2 PCR\Atmos smoothed\PCR vs MIRS + ML - 80 - 20\ML analysis\DBS_train.csv")

#%%
# Saving a validation split to the disk
df.iloc[val_index,:]
validation_DBS = df.iloc[val_index,:]
validation_DBS.to_csv("D:\OneDrive\Documents\MANNU - Spectra data\DBS for measuring malaria prevalence\Round 2 PCR\Atmos smoothed\PCR vs MIRS + ML - 80 - 20\ML analysis\DBS_validation.csv")

#%%
# importing the training split; for ML training
df = pd.read_csv("D:\OneDrive\Documents\MANNU - Spectra data\DBS for measuring malaria prevalence\Round 2 PCR\Atmos smoothed\PCR vs MIRS + ML - 80 - 20\ML analysis\DBS_train.csv")
print(df.head(5))

# df = df.drop(['Species', 'Status', 'Country', 
#                        'RearCnd', 'StoTime'], axis=1)

# df.head(5)

#%%
# Checking class distribution
class_counts = df.groupby('Status').size()
class_counts

#%%

# to have at least same number of observation
X = df.iloc[:,5:] # defining X matrix
y = df["Status"] # defining Y matrix
X


#%%
# Data splitting and defining models
num_folds = 7 # Spliting the training set into 7 parts
validation_size = 0.2 # defining the size of the test set
seed = 4 # you can choose any integer, this ensures reproducibility of the tests
scoring = 'accuracy' # score model accuracy

sss = StratifiedShuffleSplit(
        n_splits=num_folds, test_size=validation_size, random_state=seed)

#%%
models = [] # telling python to create sub names models
models.append(("KNN", KNeighborsClassifier()))
models.append(("LR", LogisticRegressionCV(multi_class = 'auto', cv=sss, random_state=seed, max_iter=2000)))
# models.append(("LDA", LinearDiscriminantAnalysis()))
models.append(("SVM", SVC(random_state=seed, gamma='auto')))
models.append(("NB", GaussianNB()))
models.append(("XGB", XGBClassifier(random_state=seed, nthread=1)))
# models.append(("CART", DecisionTreeClassifier()))
models.append(("RF", RandomForestClassifier(random_state=seed, n_estimators=100)))
# models.append(("GB", GradientBoostingClassifier()))
models.append(("MLP", MLPClassifier(random_state=seed, max_iter=1000)))


#%%
# standardizing the data
# df[df.columns[df.columns != "Status"]]=StandardScaler().fit_transform(df[df.columns[df.columns != "Status"]].values)
# df.head(5)
X_new= StandardScaler().fit_transform(X)
X_new

#%%
# Comparing classifiers with standardized data
results = []
names = []

for name, model in models:
    sss = StratifiedShuffleSplit(
        n_splits=num_folds, test_size=validation_size, random_state=seed)
    cv_results = cross_val_score(
        model, X_new, y, cv=sss, scoring=scoring)
    results.append(cv_results)
    names.append(name)
    msg = "Cross val score for {0}: {1:.2%} ± {2:.2%}".format(
        name, cv_results.mean(), cv_results.std())
    print(msg)

#%%
# results_2 = np.round(100*results)
# results_2

#%%
# plotting the results of algorithm comparison

sns.set(context="paper",
    style="whitegrid",
    font_scale=2.0,
    rc={"font.family": "Dejavu Sans"})

plt.rcParams["figure.figsize"] = [8,6]
sns.boxplot(x=names, y=results)
sns.despine(offset=10, trim=True)
plt.title(" ")
plt.xticks(rotation=90)
plt.ylim(0.50, 1.00)
plt.yticks((0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80,
0.85, 0.90, 0.95, 1.0))
plt.ylabel('Accuracy', weight = "bold");
plt.savefig("Algorithm selection_newR.png", dpi = 300, bbox_inches="tight")


#%%
# importing the training split; for ML training
df = pd.read_csv("D:\OneDrive\Documents\MANNU - Spectra data\DBS for measuring malaria prevalence\Round 2 PCR\Atmos smoothed\PCR vs MIRS + ML - 80 - 20\ML analysis\DBS_train.csv")
print(df.head(5))

df = df.drop(['Species'], axis=1)

df.head(5)


#%%
# standardizing the data
df[df.columns[df.columns != "Status"]]=StandardScaler().fit_transform(df[df.columns[df.columns != "Status"]].values)
df.head(5)


#%%

X = df.iloc[:,4:] # defining X matrix of features
y = df["Status"] # defining Y vector
X

#%%
# BIG LOOP

# Tunning the model
# LR was the best perfomed model; we will further tune the model

## Set validation procedure

num_splits = 7 # split training set into 7 parts 
validation_size = 0.2 # size of the test set
seed = 4 # pick any integer. This ensures reproducibility of the tests
scoring = 'accuracy' # score model accuracy

# preparing the model
classifier =LogisticRegressionCV(Cs = 20,
                                fit_intercept = True, 
                                cv = sss, 
                                dual = False, 
                                penalty = 'l2', 
                                scoring = None, 
                                solver = 'lbfgs', 
                                tol = 1e-4, 
                                max_iter = 2000, 
                                class_weight = 'balanced', 
                                n_jobs = -1, 
                                verbose = 1, 
                                refit = True, 
                                intercept_scaling = 1., 
                                multi_class = 'auto', 
                                random_state = 4)

# Defining hyperparameters

solver = ['lbfgs', 'newton-cg']
param_grid = dict(#Cs = Cs,
                solver = solver)


sss = StratifiedShuffleSplit(
         n_splits=num_splits, test_size=validation_size, random_state=seed)


# prepare matrices of results
sss_results = pd.DataFrame() # model parameters and global accuracy score
sss_coef = pd.DataFrame() 
sss_per_class_results = [] # per class accuracy scores
start = time()

for train_index, test_index in sss.split(X, y):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
 
    # GRID SEARCH
    grid = GridSearchCV(estimator=classifier, param_grid=param_grid, scoring=scoring, cv=sss) 
    grid_result = grid.fit(X_train, y_train)

    # print out results and give hyperparameter settings for best one
    means = grid_result.cv_results_['mean_test_score']
    stds = grid_result.cv_results_['std_test_score']
    params = grid_result.cv_results_['params']
    for mean, stdev, param in zip(means, stds, params):
           print("%.2f (%.2f) with: %r" % (mean, stdev, param))
    
    # # print best parameter settings
    print("Best: %.2f using %s" % (grid_result.best_score_, grid_result.best_params_))

    best_DBS_classifier = LogisticRegressionCV(**grid_result.best_params_)

    # Fitting the best model
    best_DBS_classifier.fit(X_train, y_train)
   
    # testing model 
    y_pred = best_DBS_classifier.predict(X_test)
    local_cm = confusion_matrix(y_test, y_pred)
    local_report = classification_report(y_test, y_pred)

    # append coefficients to dataframe
    coef_table = pd.DataFrame(best_DBS_classifier.coef_, columns=X.columns).T
    # combine outputs
    sss_coef = pd.merge(sss_coef, coef_table, left_index=True, right_index=True, how='outer')

    local_sss_results=pd.DataFrame([("Accuracy",accuracy_score(y_test, y_pred)), ("params",str(grid_result.best_params_)), ("TRAIN",str(train_index)), ("TEST",str(test_index)), ("CM", local_cm), ("Classification report", local_report)]).T
        
    local_sss_results.columns=local_sss_results.iloc[0]
    local_sss_results = local_sss_results[1:]
    sss_results = sss_results.append(local_sss_results)

    # per class accuracy
    local_support = precision_recall_fscore_support(y_test, y_pred)[3]
    local_acc = np.diag(local_cm)/local_support
    sss_per_class_results.append(local_acc)

elapsed = time() - start
print("Time elapsed: {0:.2f} minutes ({1:.1f} sec)".format(
    elapsed / 60, elapsed))


#%%
# Results
sss_results.to_csv("lgr_sssCV_record.csv", index=False)
sss_results = pd.read_csv("lgr_sssCV_record.csv")

# Accuracy distribution
lgr_acc_distrib = sss_results["Accuracy"]
lgr_acc_distrib.columns=["Accuracy"]
lgr_acc_distrib.to_csv("lgr_acc_distrib.csv", header=True, index=False)
lgr_acc_distrib = pd.read_csv("lgr_acc_distrib.csv")
lgr_acc_distrib = np.round(lgr_acc_distrib, 2)
print(lgr_acc_distrib)

#%%
# summarizing coefficients
sss_coef.dropna(axis=1, inplace=True)
sss_coef["coef mean"] = sss_coef.mean(axis=1)
sss_coef["coef sem"] = sss_coef.sem(axis=1)
sss_coef.to_csv("coef_repeatedCV_coef.csv")
sss_coef = pd.read_csv("coef_repeatedCV_coef.csv")


#%% plotting coefficients
n_features = 10
sss_coef = pd.read_csv("coef_repeatedCV_coef.csv")
sss_coef.sort_values(by="coef mean", ascending=False, inplace=True)
coef_plot_data = sss_coef.drop(["coef sem", "coef mean"], axis=1).T
coef_plot_data = coef_plot_data.iloc[1:,:].drop(coef_plot_data.columns[n_features:-n_features], axis=1)


sns.set(context="paper",
    style="white",
    font_scale=2.0,
    rc={"font.family": "Dejavu Sans"})

plt.figure(figsize=(6,8))
sns.barplot(data=coef_plot_data, orient="h", palette="RdYlBu", capsize=.2)
plt.ylabel("Wavenumbers", weight = "bold")
plt.xlabel("Coeffients", weight = "bold")
plt.savefig("D:\OneDrive\Documents\MANNU - Spectra data\DBS for measuring malaria prevalence\Round 2 PCR\Atmos smoothed\PCR vs MIRS + ML - 80 - 20\ML analysis\lgr_coeffients-2_responsetoreview.png", dpi = 300, bbox_inches="tight")


#%%
# plotting accuracy distribution
plt.figure(figsize=(2.25,3))
sns.distplot(lgr_acc_distrib, kde=False, bins=12)
# plt.savefig("lgr_acc_distrib.png", bbox_inches="tight")

#%%
# class distribution 
class_names = y.sort_values().unique()
lgr_per_class_acc_distrib = pd.DataFrame(sss_per_class_results, columns=class_names)
lgr_per_class_acc_distrib.dropna().to_csv("lgr_per_class_acc_distrib.csv")
lgr_per_class_acc_distrib = pd.read_csv("lgr_per_class_acc_distrib.csv", index_col=0)
lgr_per_class_acc_distrib = np.round(lgr_per_class_acc_distrib, 2)
lgr_per_class_acc_distrib_describe = lgr_per_class_acc_distrib.describe()
lgr_per_class_acc_distrib_describe.to_csv("lgr_per_class_acc_distrib.csv")

#%%
# plotting class distribution
lgr_per_class_acc_distrib = pd.melt(lgr_per_class_acc_distrib, var_name="status new")
sns.set(context="paper",
    style="whitegrid",
    font_scale=2.0,
    rc={"font.family": "Dejavu Sans"})

plt.figure(figsize=(8,6))
sns.violinplot(x="status new", y="value", cut = 0, data=lgr_per_class_acc_distrib)
sns.despine(left=True)
plt.xticks(rotation=0, ha="right")
plt.xticks()
plt.yticks((0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.00))
plt.xlabel(" ")
plt.ylabel("Accuracy", weight = "bold")
plt.savefig("lgr_per_class_acc_distrib-2_.png", dpi = 300, bbox_inches="tight")


#%%
# plot the prediction accuracy as a confusion matrix
sns.set(context="paper",
    style="whitegrid",
    font_scale=2.0,
    rc={"font.family": "Dejavu Sans"})

plt.rcParams["figure.figsize"] = [8,6]
# cm = confusion_matrix(Y_test, Y_pred)
class_names = np.unique(np.sort(y))
plot_confusion_matrix(local_cm, text=True, normalise=True, classes=class_names)
plt.savefig("prediction DBS-2.png", dpi = 300, bbox_inches="tight")


#%%
# Summarising precision, f_score, and recall for the training set
cr = local_report
print(cr)


########################################################
########################################################

#%%
# TESTING THE FINAL MODEL IN THE VALIDATION SET (UNSEEN DATA)

# dump/save/serialize the final model to the disk for future prediction. 

with open('best_DBS_classifier.pkl', 'wb') as fid:
     pickle.dump(best_DBS_classifier, fid)


#%%
# Importing the validation set (initially split from the whole data set) 

validation= pd.read_csv("DBS_validation.csv")
print (validation.head(5))

# age_validation = age_validation.drop(['Species', 'Status', 'Country', 
#                        'RearCnd', 'StoTime'], axis=1)
# age_validation.head(5)

#%%
# checking class distribution in the validation set

class_counts2 = validation.groupby('Status').size()
class_counts2

#%%
# Defining X and Y from the validation set
X = validation.iloc[:,5:]
Y = validation["Status"]
X


#%%
# Standardizing X matrix before prediction
X_val = StandardScaler().fit_transform(X)
X_val


#%%
# Deserializing the final model from the disk to predict new samples

with open('best_DBS_classifier.pkl', 'rb') as fid:
     final_model_loaded = pickle.load(fid)


#%%
# Evaluating the final model predicting new samples (validation set)
Y_val_pred = final_model_loaded.predict(X_val)
# predictions = [round(value) for value in Y_val_pred]
# evaluative prediction
accuracy = accuracy_score(Y, Y_val_pred)
print("Accuracy:%.2f%%" %(accuracy * 100.0))


#%%
# Plot the prediction accuracy as a confusion matrix
sns.set(context="paper",
    style="whitegrid",
    font_scale = 2.0,
    rc={"font.family": "Dejavu Sans"})
    
plt.rcParams["figure.figsize"] = [8,6]
cm = confusion_matrix(Y, Y_val_pred)
class_names = np.unique(np.sort(Y))
plot_confusion_matrix(cm, text=True, normalise=True, classes=class_names)
plt.savefig("validation prediction-2.png", dpi = 300, bbox_inches="tight")


#%%
# summarizing classification report
Cr2 = classification_report(Y_val, Y_val_pred)
print(Cr2)

# template for statistics
# hist(top_wavenumber) ---> decide distribution family for glm
# glm(top_wavenumber ~ age * sex, family=???)
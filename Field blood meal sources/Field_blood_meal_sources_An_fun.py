
#%%
import os
import io
import json
import ast
import itertools
import collections
from time import time
import sklearn
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

from sklearn.model_selection import (

    ShuffleSplit, 
    train_test_split, 
    StratifiedKFold, 
    StratifiedShuffleSplit, 
    KFold
    
    ) 
from sklearn.model_selection import (

    RandomizedSearchCV, 
    GridSearchCV, 
    cross_val_score
    
    )
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import (

    accuracy_score, 
    confusion_matrix, 
    classification_report, 
    max_error, 
    precision_recall_fscore_support, 
    roc_curve, 
    auc, 
    precision_score, 
    recall_score, 
    f1_score
    
    )

from imblearn.under_sampling import RandomUnderSampler

from sklearn import decomposition
from sklearn.pipeline import Pipeline

from sklearn.linear_model import (

    LogisticRegression, 
    LogisticRegressionCV
    
    )

from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier

import matplotlib.pyplot as plt # for making plots
import seaborn as sns
sns.set(

    context = "paper",
    style = "white",
    palette = "deep",
    font_scale = 2.0,
    color_codes = True,
    rc = ({"font.family": "Dejavu Sans"})
    
    )
# %matplotlib inline

plt.rcParams["figure.figsize"] = [6,4]

#%%

# This normalizes the confusion matrix and ensures neat plotting for all outputs.
# Function for plotting confusion matrcies

def plot_confusion_matrix(

    cm, 
    classes,
    normalize = True,
    title = 'Confusion matrix',
    xrotation = 0,
    yrotation = 0,
    cmap = plt.cm.Blues,
    printout = False
    
    ):

    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        if printout:
            print("Normalized confusion matrix")
    else:
        if printout:
            print('Confusion matrix')

    if printout:
        print(cm)
    
    plt.figure(figsize=(6,4))

    plt.imshow(
        
        cm, 
        interpolation = 'nearest', 
        vmin = .2, 
        vmax = 1.0,  
        cmap = cmap
        
        )
        
    # plt.title(title)
    plt.colorbar()
    classes = classes
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=xrotation)
    plt.yticks(tick_marks, classes, rotation=yrotation)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(
            
            j, 
            i, 
            format(cm[i, j], fmt), 
            horizontalalignment="center",
            color="white" if cm[i, j] > thresh else "black"
            
            )

    plt.tight_layout()
    plt.ylabel('True Host', weight = 'bold')
    plt.xlabel('Predicted Host', weight = 'bold')
    plt.savefig(
        (
            "C:\Mannu\Projects\Mannu Phd\Final analysis\Results\Logistic regression\Confusion_Matrix_" 
            + figure_name 
            + "_" 
            + ".png"
        ), 
        dpi = 500, 
        bbox_inches = "tight"
        
    )

#%%

def visualize(figure_name, classes, predicted, true):
    # Sort out predictions and true labels
    # for label_predictions_arr, label_true_arr, classes, outputs in zip(predicted, true, classes, outputs):
    # print('visualize predicted classes', predicted)
    # print('visualize true classes', true)
    classes_pred = np.asarray(predicted)
    classes_true = np.asarray(true)
    print(classes_pred.shape)
    print(classes_true.shape)
    # classes = ['1-9', '10-17']
    cnf_matrix = confusion_matrix(
        
        classes_true, 
        classes_pred, 
        labels = classes
        
        )
    plot_confusion_matrix(cnf_matrix, classes)


#%%

full_data_df = pd.read_csv(

    "C:\Mannu\Projects\Sporozoite Spectra for An funestus s.s\Phd data\Analysis data\Biological_attr.dat", 
    delimiter= '\t'
    
    )

full_data_df.head()

#%%
# data shape
print(full_data_df.shape)

# Checking class distribution
Counter(full_data_df["Cat7"])

#%%

# Select abdomen data == funestus

blood_field_df = full_data_df.query(

    "Cat6 == 'BF' and Cat7 == 'AB' and Cat2 == 'AF'"
    
    )

print('The shape of field blood meal source data : {}'.format(
    blood_field_df.shape)
    )

# Observe first few observations
blood_field_df.head()
Counter(blood_field_df["Cat2"])

# %%

# Import PCR results which contains the ID's of blood fed mosquitoes 
pcr_data_df = pd.read_csv(
    "C:\Mannu\Projects\Mannu Phd\MIRS_blood_meal_PCR.csv"
    )
print(pcr_data_df.head(5))

# Now select only human fed samples 

human_df = pcr_data_df.query("PCR_RESULTS == 'Human'")
# human_df = pcr_data_df.query('PCR_RESULTS in ["Human", "Human_cow", "Human_dog"]')
print(Counter(human_df["PCR_RESULTS"])) # Checking class distribution

# Now select only bovine fed samples 

bovine_df = pcr_data_df.query("PCR_RESULTS == 'Bovine'")
# bovine_df = pcr_data_df.query('PCR_RESULTS in ["Bovine", "Dog", "Cow_dog"]')
print(Counter(bovine_df["PCR_RESULTS"])) # Checking class distribution

# %%

# Select a vector of sample ID from PCR data and use it to index all the human blood samples
# from the blood meal field data

human_b_samples = human_df['SAMPLE_ID']
human_b_samples_df = blood_field_df.query("ID in @human_b_samples")

# create a new column in positive samples dataframe and name the samples as positives
human_b_samples_df['blood_meal'] = 'Human'
human_b_samples_df

#%%

# Select a vector of sample ID from PCR data and use it to index all the bovine blood samples
# from the blood meal field data

bovine_b_samples = bovine_df['SAMPLE_ID']
bovine_b_samples_df = blood_field_df.query("ID in @bovine_b_samples")

# create a new column in positive samples dataframe and name the samples as positives
bovine_b_samples_df['blood_meal'] = 'Bovine'
bovine_b_samples_df

# %%

# Concatinating human and bovine blood-fed dataframes together

human_bov_bldfed_df = pd.concat(

    [human_b_samples_df, bovine_b_samples_df], 
    axis = 0, 
    join = 'outer'

    )
# human_bov_bldfed_df.to_csv("E:\Sporozoite\human_bov_bldfed_df.csv")

# Drop unused columns
human_bov_bldfed_df = human_bov_bldfed_df.drop(
    ['ID', 'Cat2', 'Cat3', 'Cat4', 'Cat5', 'Cat6', 'Cat7', 'StoTime'], 
    axis=1
    )

human_bov_bldfed_df

# %%

# define X (matrix of features) and y (list of labels)

X = human_bov_bldfed_df.iloc[:,:-1] # select all columns except the last one
features = X 
y = human_bov_bldfed_df["blood_meal"]

print('shape of X : {}'.format(X.shape))
print('shape of y : {}'.format(y.shape))

# rescalling the data (undersampling the over respresented class - Bovine fed class)

rus = RandomUnderSampler(random_state = 42)
X_res, y_res = rus.fit_resample(X, y)
y_res_count = collections.Counter(y_res)
print(y_res_count)

# Standardise inputs using standard scaler

X_train, X_test, y_train, y_test = train_test_split(

    X_res, 
    y_res, 
    test_size= .1, 
    random_state = 42,
    shuffle = True
    
    )
print('The shape of X train index : {}'.format(X_train.shape))
print('The shape of y train index : {}'.format(y_train.shape))
print('The shape of X test index : {}'.format(X_test.shape))
print('The shape of y test index : {}'.format(y_test.shape))

# save test set for calculating Blood index
# HBI_testset = pd.concat([y_test, X_test], axis = 1, join = 'outer')
# HBI_testset.to_csv("C:\Mannu\Projects\Mannu Phd\Final analysis\HBI_testset.csv")

X = np.asarray(X_train)
y = np.asarray(y_train)
print('y labels : {}'.format(np.unique(y)))

# Data standardisation 

scaler = StandardScaler().fit(X = X)
X_transformed  = scaler.transform(X = X)

# %%

# Data splitting and defining models
num_folds = 5 # Spliting the training set into 6 parts
validation_size = 0.1 # defining the size of the validation set
seed = 42 # you can choose any integer, this ensures reproducibility of the tests
scoring = 'accuracy' # score model accuracy

random_seed  = np.random.randint(0, 81478)
kf = KFold(

    n_splits = num_folds, 
    shuffle = True, 
    random_state = random_seed
    
    )

models = [] # telling python to create sub names models
models.append(("KNN", KNeighborsClassifier()))
models.append(("LR", LogisticRegression(multi_class = 'ovr', random_state = seed, max_iter = 3500)))
models.append(("SVM", SVC(random_state=seed, kernel = 'linear', gamma = 'auto')))
models.append(("XGB", XGBClassifier(random_state = seed, nthread = 1, n_estimators = 1000)))
models.append(("RF", RandomForestClassifier(random_state = seed, n_estimators = 1000)))
models.append(("MLP", MLPClassifier(random_state=seed, max_iter = 3500,
                                    solver = 'sgd',
                                    activation = 'logistic', alpha = 0.001)))


#%%
# comparative evaluation of different classifiers
results = []
names = []

for name, model in models:
    cv_results = cross_val_score(

        model, 
        X_transformed, 
        y, 
        cv = kf, 
        scoring = scoring
        
        )
    results.append(cv_results)
    names.append(name)
    msg = "Cross val score for {0}: {1:.2%} ± {2:.2%}".format(

        name, 
        cv_results.mean(), 
        cv_results.std()
        
        )   
    print(msg)


#%%
# plotting the results of the classifier

# create a dataframe for plotting using seaborn

results_df = pd.DataFrame(results, columns = (0, 1, 2, 3, 4)).T # columns should correspond to the number of folds, k = 5

# rename columns to have number of components

results_df.columns = ['KNN', 'LR', 'SVM', 'XGB', 'RF', 'MLP']
results_df = pd.melt(results_df) # melt data frame into a long format. 
results_df.rename(columns = {'variable':'Model', 'value':'Accuracy'}, inplace = True)
results_df

sns.boxplot(

    x = results_df['Model'], 
    y = results_df['Accuracy']
    
    )
sns.despine(offset = 10, trim = True)
# plt.title("Algorithm comparison", weight="bold")
plt.xticks(rotation = 90)
plt.yticks(np.arange(0.2, 1.0 + .05, step = 0.2))
plt.ylabel('Accuracy', weight = 'bold')
plt.xlabel(" ");
plt.savefig(

    ("C:\Mannu\Projects\Mannu Phd\Final analysis\Results\Logistic regression\_algo_selection.png"), 
    dpi = 500, 
    bbox_inches = "tight"
    
    )

#%%

# big LOOP
# TUNNING THE SELECTED MODEL

num_rounds = 1 # increase this to 5 or 10 once code is bug-free
scoring = 'accuracy' # score model accuracy

# prepare matrices of results
kf_results = pd.DataFrame() # model parameters and global accuracy score
kf_per_class_results = [] # per class accuracy scores
save_predicted, save_true = [], [] # save predicted and true values for each loop

start = time()

# Specify model

classifier = LogisticRegression(
    
    multi_class = 'ovr', 
    random_state = random_seed, 
    max_iter = 3500
    
    )

# Optimizing hyper-parameters for logistic regression

solvers = ['newton-cg', 'lbfgs', 'liblinear']
c_values = [0.01, 0.1, 1, 10]     # 100, 50, 20,  10, 5]
    
# Create the random grid

random_grid = {
                'solver': solvers,
                'C': c_values
                }

for round in range (num_rounds):
    SEED = np.random.randint(0, 8147)

    # cross validation and splitting of the validation set
    
    for train_index, test_index in kf.split(X_transformed, y):
        X_train_set, X_val = X_transformed[train_index], X_transformed[test_index]
        y_train_set, y_val = y[train_index], y[test_index]

        print('The shape of X train set : {}'.format(X_train_set.shape))
        print('The shape of y train set  : {}'.format(y_train_set.shape))
        print('The shape of X val set : {}'.format(X_val.shape))
        print('The shape of y val set : {}'.format(y_val.shape))

        # generate models using all combinations of settings

        # RANDOMSED GRID SEARCH
        # Random search of parameters, using 5 fold cross validation, 
        # search across 100 different combinations, and use all available cores

        n_iter_search = 100
        gridCV = GridSearchCV(
            
            verbose = 1,
            estimator = classifier, 
            param_grid = random_grid, 
            scoring = scoring, 
            cv = kf, 
            refit = True, 
            n_jobs = -1
            
            )
            
        gridCV_result = gridCV.fit(X_train_set, y_train_set)

        # print out results and give hyperparameter settings for best one
        means = gridCV_result.cv_results_['mean_test_score']
        stds = gridCV_result.cv_results_['std_test_score']
        params = gridCV_result.cv_results_['params']
        for mean, stdev, param in zip(means, stds, params):
            print("%.2f (%.2f) with: %r" % (mean, stdev, param))

        # print best parameter settings
        print("Best: %.2f using %s" % (
            
            gridCV_result.best_score_,
            gridCV_result.best_params_
            
            )
        )

        # Insert the best parameters identified by randomized grid search into the base classifier
        best_classifier = classifier.set_params(

            **gridCV_result.best_params_, 
            n_jobs = -1
            
            )
        
        best_classifier.fit(X_train_set, y_train_set)

        # predict test instances 

        y_pred = best_classifier.predict(X_val)
        # y_test = np.delete(y_res, train_index, axis=0)
        local_cm = confusion_matrix(y_val, y_pred)
        local_report = classification_report(y_val, y_pred)

        # zip predictions for all rounds for plotting averaged confusion matrix
            
        for predicted, true in zip(y_pred, y_val):
            save_predicted.append(predicted)
            save_true.append(true)

        # # append feauture importances
        # local_feat_impces = pd.DataFrame(best_classifier.feature_importances_,
        #                                     index = features.columns).sort_values(by = 0, ascending = False)
        
        # summarizing results
        local_kf_results = pd.DataFrame(
            [
                ("Accuracy", accuracy_score(y_val, y_pred)), 
                ("TRAIN",str(train_index)),
                ("TEST",str(test_index)),
                ("CM", local_cm), 
                ("Classification report", local_report), 
                ("y_test", y_test)
                
                ]
        ).T
            
        local_kf_results.columns = local_kf_results.iloc[0]
        local_kf_results = local_kf_results[1:]
        kf_results = kf_results.append(local_kf_results)

        # per class accuracy
        local_support = precision_recall_fscore_support(y_val, y_pred)[3]
        local_acc = np.diag(local_cm)/local_support
        kf_per_class_results.append(local_acc)

elapsed = time() - start
print("Time elapsed: {0:.2f} minutes ({1:.1f} sec)".format(
    elapsed / 60, elapsed))

#%%
# plot confusion averaged for the validation set

classes = np.unique(np.sort(y))
figure_name = 'baseline_model'
visualize(figure_name, classes, save_predicted, save_true)

#%%
# save the trained model to disk for future use

with open(
    'C:\Mannu\Projects\Mannu Phd\Final analysis\Results\Logistic regression\classifier.pkl', 
    'wb'
    ) as fid:
     pickle.dump(best_classifier, fid)

# %%

# Results
kf_results.to_csv(

    "C:\Mannu\Projects\Mannu Phd\Final analysis\Results\Logistic regression\crf_kfCV_record.csv", 
    index = False
    
    )
kf_results = pd.read_csv("C:\Mannu\Projects\Mannu Phd\Final analysis\Results\Logistic regression\crf_kfCV_record.csv")

# Accuracy distribution
crf_acc_distrib = kf_results["Accuracy"]
crf_acc_distrib.columns=["Accuracy"]

crf_acc_distrib.to_csv(
    "C:\Mannu\Projects\Mannu Phd\Final analysis\Results\Logistic regression\crf_acc_distrib.csv", 
    header = True, 
    index = False
    )

crf_acc_distrib = pd.read_csv("C:\Mannu\Projects\Mannu Phd\Final analysis\Results\Logistic regression\crf_acc_distrib.csv")
crf_acc_distrib = np.round(crf_acc_distrib, 2)
print(crf_acc_distrib)


#%%
# plotting accuracy distribution
plt.figure(figsize=(2.25,3))
sns.distplot(crf_acc_distrib, kde=False, bins=12)
# plt.savefig("lgr_acc_distrib.png", bbox_inches="tight")

#%%
# class distribution 
class_names = np.unique(np.sort(y))
crf_per_class_acc_distrib = pd.DataFrame(kf_per_class_results, columns=class_names)
crf_per_class_acc_distrib.dropna().to_csv("C:\Mannu\Projects\Mannu Phd\Final analysis\Results\Logistic regression\crf_per_class_acc_distrib.csv")

crf_per_class_acc_distrib = pd.read_csv(
    "C:\Mannu\Projects\Mannu Phd\Final analysis\Results\Logistic regression\crf_per_class_acc_distrib.csv", 
    index_col = 0
    )

crf_per_class_acc_distrib = np.round(crf_per_class_acc_distrib, 2)
crf_per_class_acc_distrib_describe = crf_per_class_acc_distrib.describe()
crf_per_class_acc_distrib_describe.to_csv("C:\Mannu\Projects\Mannu Phd\Final analysis\Results\Logistic regression\crf_per_class_acc_distrib.csv")


#%%
# plotting class distribution
lgr_per_class_acc_distrib = pd.melt(crf_per_class_acc_distrib, var_name = "status new")


plt.figure(figsize=(6,4))
sns.violinplot(

    x = "status new", 
    y = "value", 
    cut = 1, 
    data = lgr_per_class_acc_distrib
    
    )
sns.despine(left=True)
plt.xticks(rotation=0, ha="right")
plt.xticks()
plt.yticks(np.arange(0.2, 1.0 + .05, step = 0.2))
plt.xlabel(" ")
plt.ylabel("Prediction accuracy", weight = "bold")
plt.savefig(

    ("C:\Mannu\Projects\Mannu Phd\Final analysis\Results\Logistic regression\per_class_acc_distrib_full_wn.png"), 
    dpi = 500, 
    bbox_inches = "tight"
    
    )


#%%
# Predict test data

# Transform data 

trans_X_test = scaler.transform(X = np.asarray(X_test))

# Predict the age of mosquitos preserved in ethanol 

y_test_pred = best_classifier.predict(trans_X_test)

accuracy_val = accuracy_score(y_test, y_test_pred)
print("Accuracy: %.2f%%" % (accuracy_val * 100.0))

#%%
# Plotting confusion matrix for X_val 

classes = np.unique(np.sort(y_test))
figure_name = 'test_prediction'
visualize(figure_name, classes,  y_test_pred, y_test)

#%%
# Summarising precision, f_score, and recall for the test set

cr_full_wn_val = classification_report(y_test, y_test_pred)
print('Classification report : {}'.format(cr_full_wn_val))

# save classification report to disk as a csv

cr_full_wn_val = pd.read_fwf(

    io.StringIO(cr_full_wn_val), 
    header = 0
    
    )
cr_full_wn_val = cr_full_wn_val.iloc[0:]
cr_full_wn_val.to_csv("C:\Mannu\Projects\Mannu Phd\Final analysis\Results\Logistic regression\classification_report.csv")

#%%
def my_logistic_report(X_test_,y_test_, threshold=0.5): # you could make it even more general!
    
    y_hat = best_classifier.predict(X_test_)#[:,1]
    print(y_hat)
    # y_hat = np.where(probs>=threshold,1,0)
    
    cm = confusion_matrix(y_test_, y_hat)
    accuracy = accuracy_score(y_test_, y_hat)
    precision = precision_score(y_test_, y_hat, pos_label = 'Bovine')
    recall = recall_score(y_test_, y_hat, pos_label = 'Bovine')
    f1score = f1_score(y_test_, y_hat, pos_label = 'Bovine')
    cm_labeld = pd.DataFrame(cm, index=['Actual : Bovine ','Actual : Human'], columns=['Predict : Bovine','Predict :Human'])
    
    print("-----------------------------------------")
    print('Accuracy  = {}'.format(accuracy))
    print('Precision = {}'.format(precision))
    print('Recall    = {}'.format(recall))
    print('f1_score  = {}'.format(f1score))
    print("-----------------------------------------")
    return cm_labeld

cm_labeld_ = my_logistic_report(trans_X_test, y_test)
# cm_labeld_.to_csv("C:\Mannu\Projects\Mannu Phd\Final analysis\Results\Logistic regression\cm_labeld_.csv")


# %%

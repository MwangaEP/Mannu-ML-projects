#%%
# Importing all modules

import this
import os
import io
import ast
import itertools
import collections
from time import time
from tqdm import tqdm 

import numpy as np # for mathematical computation
import pandas as pd # for mathematical computation

import scipy.stats as stats
# import statsmodels.api as sm
# import statsmodels.formula.api as smf

from random import randint
from collections import Counter 

import pickle

from sklearn.model_selection import KFold
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV

import sklearn.metrics as metrics
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix, precision_recall_fscore_support, mean_squared_error, r2_score, roc_auc_score, roc_curve

from sklearn.preprocessing import StandardScaler

from imblearn.under_sampling import NearMiss, CondensedNearestNeighbour, OneSidedSelection
from imblearn.over_sampling import ADASYN, SMOTE, KMeansSMOTE

from sklearn.linear_model import LogisticRegressionCV
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

from sklearn import decomposition
from sklearn.feature_selection import SelectKBest

from sklearn.pipeline import Pipeline
from sklearn.pipeline import FeatureUnion

from sklearn.inspection import permutation_importance

from xgboost import XGBClassifier

import matplotlib.pyplot as plt # for making plots
import seaborn as sns
sns.set(context="paper",
        style="whitegrid",
        palette="deep",
        font_scale=2.0,
        color_codes=True,
        rc=None)
%matplotlib inline
plt.rcParams["figure.figsize"] = [6,4]

#%%

# define a convenient plotting function (confusion matrix)
def plot_confusion_matrix(cm, classes,
                          normalise = True,
                          text = False,
                          title = 'Confusion matrix',
                          xrotation = 0,
                          yrotation = 0,
                          cmap = plt.cm.Blues,
                          printout = False):
    """
    This function prints and plots the confusion matrix.
    Normalisation can be applied by setting 'normalise=True'.
    """

    if normalise:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        if printout:
            print("Normalized confusion matrix")
        
    else:
        if printout:
            print('Confusion matrix')

    if printout:
        print(cm)

    plt.figure(figsize=(6, 4))
    plt.imshow(cm, interpolation='nearest', cmap=cmap, vmin = 0.2, vmax = 1.0 )
    # plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    # plt.set_ylim(len(classes)-0.5, -0.5)
    plt.xticks(tick_marks, classes, rotation=xrotation)
    plt.yticks(tick_marks, classes, rotation=yrotation)

    fmt = '.2f' if normalise else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]),
                                  range(cm.shape[1])):
            plt.text(j, i, format(cm[i, j], fmt), 
            horizontalalignment="center",
            color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label', weight = 'bold')
    plt.xlabel('Predicted label', weight = 'bold')


#%%
df = pd.read_csv("C:\Mannu\Projects\Sporozoite Spectra for An funestus s.s\ML analysis\Full_sporozoite_gambiae.csv")
df.head()

#%%

# data shape
print(df.shape)
# Checking class distribution abd correlation in the data
Counter(df["Infection"])

#%%

# Select matrix of features
training_features = df.iloc[:,11:] 
training_features = training_features.rename(columns = lambda x :str(x)[14:])
X = training_features.rename(columns = lambda x :str(x)[:-3])
X = np.asarray(X)

#%%
# Select vector of labels 

updated_labels = []

for i in df["Infection"]:
    if i == 0:
        updated_labels.append('Negative')
    else:
        updated_labels.append('Positive')

y = np.asarray(updated_labels)


#%%

# rescalling the data (oversampling the underrepresented class - negative class)

sm = SMOTE(random_state = 42)
X_res, y_res = sm.fit_resample(X, y)
y_res_count = collections.Counter(y_res)
print(y_res_count)

# Standardise inputs using standard scaler

X_train, X_val, y_train, y_val = train_test_split(X_res, y_res, test_size= .05, random_state = 42, shuffle = True)
print('The shape of X train index : {}'.format(X_train.shape))
print('The shape of y train index : {}'.format(y_train.shape))
print('The shape of X val index : {}'.format(X_val.shape))
print('The shape of y val index : {}'.format(y_val.shape))

X = np.asarray(X_train)
y = np.asarray(y_train)
print('y labels : {}'.format(np.unique(y)))

# standardisation 
scl = StandardScaler().fit(X = X)
X_new  = scl.transform(X = X)

#%%

# Data splitting and defining models
num_folds = 5 # Spliting the training set into 6 parts
validation_size = 0.1 # defining the size of the validation set
seed = 42 # you can choose any integer, this ensures reproducibility of the tests
scoring = 'accuracy' # score model accuracy

kf = KFold(n_splits=num_folds, shuffle=True, random_state=seed)

# kf = StratifiedShuffleSplit(
#         n_splits=num_folds, test_size=validation_size, random_state=seed)

RF = RandomForestClassifier(n_estimators = 100, random_state=seed)
XGB = XGBClassifier(random_state=seed, nthread=1)
num_trees = 200

models = [] # telling python to create sub names models
models.append(("KNN", KNeighborsClassifier()))
models.append(("LR", LogisticRegressionCV(multi_class = 'ovr', cv=kf, random_state=seed, max_iter=4000)))
models.append(("SVM", SVC(random_state=seed, kernel = 'linear', gamma = 'auto')))
models.append(("NB", GaussianNB()))
models.append(("SDG", SGDClassifier(random_state = seed, max_iter=1000, early_stopping=True)))
models.append(("XGB", XGBClassifier(random_state=seed, nthread=1, scale_pos_weight = 118/56 )))
models.append(("RF", RandomForestClassifier(random_state=seed, n_estimators=300)))
models.append(("MLP", MLPClassifier(random_state=seed, max_iter=2000,
                                    solver = 'adam',
                                    activation = 'logistic', alpha = 0.01 )))
# models.append(("AdaBoost", AdaBoostClassifier(base_estimator = XGB, n_estimators=num_trees, random_state=seed)))
# models.append(("Bagging", BaggingClassifier(base_estimator=RF, n_estimators=num_trees, random_state=seed)))
 

#%%
# comparative evaluation of different classifiers
results = []
names = []

for name, model in models:
    cv_results = cross_val_score(
        model, X_new, y, cv = kf, scoring = scoring)
    results.append(cv_results)
    names.append(name)
    msg = "Cross val score for {0}: {1:.2%} ± {2:.2%}".format(
        name, cv_results.mean(), cv_results.std())
    print(msg)


#%%
# plotting the results of the classifiers

plt.rcParams["figure.figsize"] = [6,4]
sns.set(context="paper",
        style="whitegrid",
        palette="deep",
        font_scale=2.0,
        color_codes=True,
        rc=({"font.family": "Dejavu Sans"}))

sns.boxplot(x=names, y=results)
sns.despine(offset=10, trim=True)
# plt.title("Algorithm comparison", weight="bold")
plt.xticks(rotation=90)
plt.yticks(np.arange(0.5, 1.0 + .05, step = 0.1))
plt.ylabel('Accuracy', weight = 'bold');
# plt.savefig(("D:\Projects\Sporozoite Spectra for An funestus s.s\ML analysis\Algorithm selection.png"), dpi = 500, bbox_inches="tight")


#%%

# big LOOP
# TUNNING THE SELECTED MODEL

## Set validation procedure
num_folds = 5 # split training set into 5 parts for validation
num_rounds = 5 # increase this to 5 or 10 once code is bug-free
# seed = 4 # pick any integer. This ensures reproducibility of the tests
scoring = 'accuracy' # score model accuracy

# prepare matrices of results
kf_results = pd.DataFrame() # model parameters and global accuracy score
kf_per_class_results = [] # per class accuracy scores
start = time()

# under-sample over-represented classes (Negative class)

for round in range (num_rounds):
    SEED = np.random.randint(0, 814789)

    
    # cross validation and splitting of the validation set
    for train_index, test_index in kf.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        
        # standardise features using standard scaler

        X_train  = scl.transform(X = X_train)
        X_test = scl.transform(X = X_test)

        print('The shape of X train index : {}'.format(X_train.shape))
        print('The shape of y train index : {}'.format(y_train.shape))
        print('The shape of X test index : {}'.format(X_test.shape))
        print('The shape of y test index : {}'.format(y_test.shape))

        # Specify model

        classifier = RandomForestClassifier()

        # Optimizing hyper-parameters for random forest

        # Number of trees in random forest
        n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
        # Number of features to consider at every split
        max_features = ['auto', 'sqrt']
        # Maximum number of levels in tree
        max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
        max_depth.append(None)
        # Minimum number of samples required to split a node
        min_samples_split = [2, 5, 10]
        # Minimum number of samples required at each leaf node
        min_samples_leaf = [1, 2, 4]
        # Method of selecting samples for training each tree
        bootstrap = [True, False]
        
        # Create the random grid
        random_grid = {'n_estimators': n_estimators,
                    'max_features': max_features,
                    'max_depth': max_depth,
                    'min_samples_split': min_samples_split,
                    'min_samples_leaf': min_samples_leaf,
                    'bootstrap': bootstrap}
        print(random_grid)

        # generate models using all combinations of settings

        # RANDOMSED GRID SEARCH
        # Random search of parameters, using 5 fold cross validation, 
        # search across 100 different combinations, and use all available cores

        n_iter_search = 100
        rsCV = RandomizedSearchCV(verbose = 1,
                    estimator = classifier, param_distributions = random_grid, n_iter = n_iter_search, 
                                scoring = scoring, cv = kf, refit = True, n_jobs = -1)
        
        rsCV_result = rsCV.fit(X_train, y_train)

        # print out results and give hyperparameter settings for best one
        means = rsCV_result.cv_results_['mean_test_score']
        stds = rsCV_result.cv_results_['std_test_score']
        params = rsCV_result.cv_results_['params']
        for mean, stdev, param in zip(means, stds, params):
            print("%.2f (%.2f) with: %r" % (mean, stdev, param))

        # print best parameter settings
        print("Best: %.2f using %s" % (rsCV_result.best_score_,
                                    rsCV_result.best_params_))

        # Insert the best parameters identified by randomized grid search into the base classifier
        best_classifier = RandomForestClassifier(**rsCV_result.best_params_)
       
       
        best_classifier.fit(X_train, y_train)

        # predict test instances 

        y_pred = best_classifier.predict(X_test)
        # y_test = np.delete(y_res, train_index, axis=0)
        local_cm = confusion_matrix(y_test, y_pred)
        local_report = classification_report(y_test, y_pred)

        # zip predictions for all rounds for plotting averaged confusion matrix
        
        save_predicted, save_true = [], [] # save predicted and true values for each loop
        for predicted, true in zip(y_pred, y_test):
            save_predicted.append(predicted)
            save_true.append(true)

       # append feauture importances
        local_feat_impces = pd.DataFrame(best_classifier.feature_importances_,
                                         index = training_features.columns).sort_values(by = 0, ascending = False)
    
        # summarizing results
        local_kf_results = pd.DataFrame([("Accuracy", accuracy_score(y_test, y_pred)), 
                                          ("TRAIN",str(train_index)), 
                                          ("TEST",str(test_index)), 
                                          ("CM", local_cm), 
                                          ("Classification report", local_report), 
                                          ("y_test", y_test),
                                          ("Feature importances", local_feat_impces.to_dict())]).T
        
        local_kf_results.columns = local_kf_results.iloc[0]
        local_kf_results = local_kf_results[1:]
        kf_results = kf_results.append(local_kf_results)

        # per class accuracy
        local_support = precision_recall_fscore_support(y_test, y_pred)[3]
        local_acc = np.diag(local_cm)/local_support
        kf_per_class_results.append(local_acc)

elapsed = time() - start
print("Time elapsed: {0:.2f} minutes ({1:.1f} sec)".format(
    elapsed / 60, elapsed))


#%%

# plot averaged confusion matrix for training
averaged_CM = confusion_matrix(save_true, save_predicted)

classes = np.unique(np.sort(y))

plt.rcParams["figure.figsize"] = [8,6]
sns.set(context="paper",
        style="whitegrid",
        palette="deep",
        font_scale=2.0,
        color_codes=True,
        rc=({"font.family": "Dejavu Sans"}))
plot_confusion_matrix(averaged_CM, classes)
plt.savefig(("C:\Mannu\Projects\Sporozoite Spectra for An funestus s.s\ML analysis\Results\_averaged_CM_full_wn.png"), dpi = 500, bbox_inches="tight")

#%%
# Results
kf_results.to_csv("C:\Mannu\Projects\Sporozoite Spectra for An funestus s.s\ML analysis\Results\crf_kfCV_record.csv", index=False)
kf_results = pd.read_csv("C:\Mannu\Projects\Sporozoite Spectra for An funestus s.s\ML analysis\Results\crf_kfCV_record.csv")

# Accuracy distribution
crf_acc_distrib = kf_results["Accuracy"]
crf_acc_distrib.columns=["Accuracy"]
crf_acc_distrib.to_csv("C:\Mannu\Projects\Sporozoite Spectra for An funestus s.s\ML analysis\Results\crf_acc_distrib.csv", header=True, index=False)
crf_acc_distrib = pd.read_csv("C:\Mannu\Projects\Sporozoite Spectra for An funestus s.s\ML analysis\Results\crf_acc_distrib.csv")
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
crf_per_class_acc_distrib.dropna().to_csv("C:\Mannu\Projects\Sporozoite Spectra for An funestus s.s\ML analysis\Results\crf_per_class_acc_distrib.csv")
crf_per_class_acc_distrib = pd.read_csv("C:\Mannu\Projects\Sporozoite Spectra for An funestus s.s\ML analysis\Results\crf_per_class_acc_distrib.csv", index_col=0)
crf_per_class_acc_distrib = np.round(crf_per_class_acc_distrib, 2)
crf_per_class_acc_distrib_describe = crf_per_class_acc_distrib.describe()
crf_per_class_acc_distrib_describe.to_csv("C:\Mannu\Projects\Sporozoite Spectra for An funestus s.s\ML analysis\Results\crf_per_class_acc_distrib.csv")


#%%
# plotting class distribution
lgr_per_class_acc_distrib = pd.melt(crf_per_class_acc_distrib, var_name="status new")
sns.set(context="paper",
    style="whitegrid",
    font_scale=2.0,
    rc={"font.family": "Dejavu Sans"})

plt.figure(figsize=(6,4))
sns.violinplot(x="status new", y="value", cut = 1, data=lgr_per_class_acc_distrib)
sns.despine(left=True)
plt.xticks(rotation=0, ha="right")
plt.xticks()
plt.yticks(np.arange(0.5, 1.0 + .05, step = 0.1))
plt.xlabel(" ")
plt.ylabel("Prediction accuracy", weight = "bold")
plt.savefig(("C:\Mannu\Projects\Sporozoite Spectra for An funestus s.s\ML analysis\Results\per_class_acc_distrib_full_wn.png"), dpi = 500, bbox_inches="tight")


#%% Feature Importances

# make this into bar with error bars across all best models

rskf_results = pd.read_csv("C:\Mannu\Projects\Sporozoite Spectra for An funestus s.s\ML analysis\Results\crf_kfCV_record.csv")

# All feat imp
all_featimp = pd.DataFrame(ast.literal_eval(rskf_results["Feature importances"][0]))

for featimp in rskf_results["Feature importances"][1:]:
    featimp = pd.DataFrame(ast.literal_eval(featimp))
    all_featimp = all_featimp.merge(featimp, left_index=True, right_index=True)

all_featimp["mean"] = all_featimp.mean(axis=1)
all_featimp["sem"] = all_featimp.sem(axis=1)
all_featimp.sort_values(by="mean", inplace=True)

featimp_global_mean = all_featimp["mean"].mean()
featimp_global_sem = all_featimp["mean"].sem()

sns.set(context="paper",
    style="whitegrid",
    font_scale=2.0,
    rc={"font.family": "Dejavu Sans"})


fig = all_featimp["mean"][-40:].plot(figsize=(5, 11),
                                    kind="barh",
                                    # orientation = 'vertical',
                                    legend=False,
                                    xerr=all_featimp["sem"],
                                    ecolor='k')
plt.xlabel("Feature importance", weight = 'bold')
plt.axvspan(xmin=0, xmax=featimp_global_mean+3*featimp_global_sem,facecolor='r', alpha=0.3)
plt.axvline(x=featimp_global_mean, color="r", ls="--", dash_capstyle="butt")
sns.despine()

# Add mean accuracy of best models to plots
plt.annotate("Average MSE:\n{0:.3f} ± {1:.3f}".format(crf_acc_distrib.mean()[
             0], crf_acc_distrib.sem()[0]), xy=(0.06, 0), color="k")

plt.savefig(("C:\Mannu\Projects\Sporozoite Spectra for An funestus s.s\ML analysis\Results\_feature_impces_full_wn.png"), dpi = 500, bbox_inches="tight")

#%%
# Predict validation data

# Transform data using the mean and standard deviation from the model training data

trans_X_val = scl.transform(X = X_val)

# Predict the age of mosquitos preserved in ethanol 

y_val_pred = best_classifier.predict(trans_X_val)

accuracy_val_eth = accuracy_score(y_val, y_val_pred)
print("Accuracy: %.2f%%" % (accuracy_val_eth * 100.0))

#%%
# Plotting confusion matrix for X_val 

plt.rcParams["figure.figsize"] = [8,6]
sns.set(context="paper",
        style="whitegrid",
        palette="deep",
        font_scale=2.0,
        color_codes=True,
        rc=({"font.family": "Dejavu Sans"}))

cm_val = confusion_matrix(y_val, y_val_pred)

plot_confusion_matrix(cm_val, text=True, normalise=True, classes = class_names)
plt.savefig(("C:\Mannu\Projects\Sporozoite Spectra for An funestus s.s\ML analysis\Results\CM_full_wn_val.png"), dpi = 500, bbox_inches="tight")

#%%
# Summarising precision, f_score, and recall for the validation set
cr_full_wn_val = classification_report(y_val, y_val_pred)
print('Classification report : {}'.format(cr_full_wn_val))

# save classification report to disk as a csv

cr_full_wn_val = pd.read_fwf(io.StringIO(cr_full_wn_val), header=0)
cr_full_wn_val = cr_full_wn_val.iloc[1:]
cr_full_wn_val.to_csv("C:\Mannu\Projects\Sporozoite Spectra for An funestus s.s\ML analysis\Results\classification_report_full_wn_val.csv")


#%%

"""
    Train the data using data from important wavenumbers derived from feature importances of the Random forest classifier.
    
    Select all important wavenumbers, and return a dataframe of selected wavenumbers and their intensity.


    """

#%%

# Collect all the important wavenumbers
important_wavenumb = pd.DataFrame(all_featimp["mean"][-100:])
important_wavenumb = important_wavenumb.reset_index()
important_wavenumb = important_wavenumb['index'].to_list()

# wnumber = [ int(x) for x in important_wavenumb]
# wnumber.sort
# print(wnumber)
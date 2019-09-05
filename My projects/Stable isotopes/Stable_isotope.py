# This programs is aiming to see whether we can detect mosquito changes after feeding on D-earth during the larval stage

#%%
# Importing all modules

import this
import os
import io # saving test as dataframes
import ast
import itertools
import collections
from time import time
from tqdm import tqdm 

import numpy as np # for arrays
import pandas as pd # for dataframes

import scipy.stats as stats
import statsmodels.api as sm
import statsmodels.formula.api as smf

from random import randint
from collections import Counter 

import pickle

from sklearn.model_selection import KFold
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV

import sklearn.metrics as metrics
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix, precision_recall_fscore_support, mean_squared_error, r2_score, roc_auc_score, roc_curve

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import Normalizer

from imblearn.under_sampling import RandomUnderSampler
from imblearn.ensemble import EasyEnsemble

from sklearn.linear_model import LogisticRegressionCV

from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import VotingClassifier

import xgboost as xgb
from xgboost import XGBClassifier

from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest

from sklearn.pipeline import Pipeline
from sklearn.pipeline import FeatureUnion



import matplotlib.pyplot as plt # for making plots
import seaborn as sns
sns.set(context="paper",
        style="whitegrid",
        palette="deep",
        font_scale=2.0,
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


# define a convenient plotting function (confusion matrix)
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
    plt.ylabel('True label', color = 'red', weight = 'bold')
    plt.xlabel('Predicted label', color = 'red', weight = 'bold')


#%%
# import negative delimiter file and change it to csv
df = pd.read_csv("D:\Projects\Isotope Samples\Data\stable_isotope.dat", delimiter='\t')
print(df.head())
df.to_csv("D:\Projects\Isotope Samples\Data\stable_isotope.csv")


#%%
# renaming details in columns of interest
df = pd.read_csv("D:\Projects\Isotope Samples\Data\stable_isotope.csv")
df.head()

df.loc[df.Cat4 == 'CONTROL', 'Cat4'] = 'Control'
df.loc[df.Cat4 == 'REPLICATE1', 'Cat4'] = 'Treatment'
df.loc[df.Cat4 == 'REPLICATE2', 'Cat4'] = 'Treatment'
df.loc[df.Cat4 == 'REPLICATE3', 'Cat4'] = 'Treatment'
df.loc[df.Cat4 == 'REPLICATE4', 'Cat4'] = 'Treatment'

df.to_csv("D:\Projects\Isotope Samples\Data\stable_isotope_all.csv")

#%%
df = pd.read_csv("D:\Projects\Isotope Samples\Data\stable_isotope_all.csv")
print(df.head())
# data shape
print(df.shape)

#%%
# Checking class distribution abd correlation in the data
Counter(df["Cat4"])

#%%
# spliting validation set from the train set
X = df.iloc[:,8:] # matrix of features
y = df["Cat4"] # vector of labels
print(X)

seed = 4
rs = ShuffleSplit(n_splits = 10, test_size = 0.2, random_state = seed)
rs.get_n_splits(X)
print(rs)

for train_index, val_index in rs.split(X):
    print("TRAIN:", train_index, "VALIDATION:", val_index)

print(train_index.shape, val_index.shape)


#%%
# saving training set to the disk
df.iloc[train_index,:]
training_set = df.iloc[train_index,:]
training_set.to_csv("D:\Projects\Isotope Samples\ML-analysis\set_training.csv")

#%%
# saving validation set to disk
df.iloc[val_index,:]
validation_set = df.iloc[val_index,:]
validation_set.to_csv("D:\Projects\Isotope Samples\ML-analysis\set_validation.csv")

#%%
# Since our data has the imbalance structure,
# we will now undersample our data at least to have consistence between classes
df = pd.read_csv("D:\Projects\Isotope Samples\ML-analysis\set_training.csv") 

X = df.iloc[:,9:] # matrix of features
y = df["Cat4"] # vector of labels
print(X)

print(Counter(df["Cat4"]))

#%%
## rescalling the data
seed = 4
rus = RandomUnderSampler(random_state = seed)
X_resampled, y_resampled = rus.fit_sample(X, y)
y_resampled_count = collections.Counter(y_resampled)
print(y_resampled_count)

#%%
# defining our new X and Y matrices
X_new = X_resampled
y_new = y_resampled
X_new


#%%
# Data splitting and defining models
num_folds = 7 # Spliting the training set into 10 parts
validation_size = 0.2 # defining the size of the validation set
seed = 4 # choose any integer, this ensures reproducibility of the tests
scoring = 'accuracy' # score model accuracy

skf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=seed)

sss = StratifiedShuffleSplit(
        n_splits=num_folds, test_size=validation_size, random_state=seed)

# estimators = []
# model1 = KNeighborsClassifier()
# estimators.append(('KNN', model1))
# model2 = LogisticRegressionCV(multi_class = 'auto', cv=sss, random_state=seed, max_iter=2500)
# estimators.append(('Logistic', model2))
# model3 = SVC(random_state=seed, gamma='auto')
# estimators.append(('SVM', model3))
# model4 = XGBClassifier(random_state=seed, nthread=1)
# estimators.append(('XGB', model4))
# model5 = RandomForestClassifier(random_state=seed, n_estimators=200)
# estimators.append(('RF', model5))
# model6 = MLPClassifier(random_state=seed, max_iter=2000)
# estimators.append(('MLP', model6))

# create the ensembe model
# ensemble = VotingClassifier(estimators)

RF = RandomForestClassifier(n_estimators = 100, random_state=seed)
num_trees = 300

models = [] # telling python to create sub names models
models.append(("KNN", KNeighborsClassifier()))
models.append(("LR", LogisticRegressionCV(multi_class = 'auto', cv=sss, random_state=seed, max_iter=2500)))
models.append(("SVM", SVC(random_state=seed, gamma='auto')))
models.append(("NB", GaussianNB()))
models.append(("XGB", XGBClassifier(random_state=seed, nthread=1)))
models.append(("RF", RandomForestClassifier(random_state=seed, n_estimators=200)))
models.append(("MLP", MLPClassifier(random_state=seed, max_iter=2000)))
models.append(("AdaBoost", AdaBoostClassifier(n_estimators=num_trees, random_state=seed)))
models.append(("Bagging", BaggingClassifier(base_estimator=RF, n_estimators=num_trees, random_state=seed)))


#%%
# fit the ensembe model
# cv_results = cross_val_score(
#         ensemble, X_new, y_new, cv=sss, scoring=scoring)

# # msg = "Cross val score for {0}: {1:.2%}".format(cv_results.mean())
# print(cv_results.mean())

#%%
# comparative evaluation of different classifiers
results = []
names = []

for name, model in models:
    sss = StratifiedShuffleSplit(
        n_splits=num_folds, test_size=validation_size, random_state=seed)
    # skf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=seed)
    cv_results = cross_val_score(
        ensemble, X_new, y_new, cv=sss, scoring=scoring)
    results.append(cv_results)
    names.append(name)
    msg = "Cross val score for {0}: {1:.2%} ± {2:.2%}".format(
        name, cv_results.mean(), cv_results.std())
    print(msg)

#%%
# plotting the results of the classifiers
sns.set(context="paper",
        style="whitegrid",
        palette="deep",
        font_scale=2.0,
        color_codes=True,
        rc=({"font.family": "Dejavu Sans"}))

plt.rcParams["figure.figsize"] = [8,6]
sns.boxplot(x=names, y=results)
sns.despine(offset=10, trim=True)
plt.title("Algorithm comparison", weight="bold")
plt.xticks(rotation=90)
plt.yticks()
plt.ylabel('Accuracy');
# plt.savefig("D:\Projects\D-Earth Samples\ML analysis output\comparison_algorithm.png", dpi = 500, bbox_inches="tight")

#%%
# standardizing the data
X_new = StandardScaler().fit_transform(X_new)


#%%
# Comparing classifiers with standardized data
results = []
names = []

for name, model in models:
    # skf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=seed)
    sss = StratifiedShuffleSplit(
        n_splits=num_folds, test_size=validation_size, random_state=seed)
    cv_results = cross_val_score(
        model, X_new, y_new, cv=sss, scoring=scoring)
    results.append(cv_results)
    names.append(name)
    msg = "Cross val score for {0}: {1:.2%} ± {2:.2%}".format(
        name, cv_results.mean(), cv_results.std())
    print(msg)

#%%
# plotting 
sns.set(context="paper",
        style="whitegrid",
        palette="deep",
        font_scale=2.0,
        color_codes=True,
        rc=({"font.family": "Dejavu Sans"}))

sns.boxplot(x=names, y=results)
sns.despine(offset=10, trim=True)
plt.title("Algorithm comparison", weight="bold",)
plt.xticks(rotation=90)
plt.yticks()
plt.ylabel('Accuracy', weight = 'bold');
plt.savefig("D:\Projects\Isotope Samples\ML-analysis\selection_algorithm_2.png", dpi = 500, bbox_inches="tight")


#%%
# preparing training dataset
df = pd.read_csv("D:\Projects\Isotope Samples\ML-analysis\set_training.csv") 

print(df.head(5))

# df = df.drop(['Age', 'Species', 'Status', 
                    #    'RearCnd', 'StoTime'], axis=1)

print(df.head(5))

X = df.iloc[:,9:] # matrix of features
y = df["Cat4"] # vector of labels
print(y)
X

#%%

# BIG LOOP

# TUNNING THE SELECTED MODEL
# Using logistic regression, and we will train it more to predict stable isotopes from mosquito samples

# Set validation procedure
num_folds = 7 # split training set into 10 parts for validation
validation_size = 0.2 # size of test set
num_rounds = 5 # increase this to 5 or 10 once code is bug-free
seed = 4 # pick any integer. This ensures reproducibility of the tests
scoring = 'accuracy' # score model accuracy

# prepare matrices of results
sss_results = pd.DataFrame() # model parameters and global accuracy score
sss_per_class_results = [] # per class accuracy scores
sss_coef = pd.DataFrame() # model coeffients
start = time()

## choose a validation approach
sss = StratifiedShuffleSplit(
        n_splits=num_folds, test_size=validation_size, random_state=seed)

# create a pipeline
lgr_pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('clf', LogisticRegressionCV(Cs = 20,
                                fit_intercept = True, 
                                cv = sss, 
                                dual = False, 
                                penalty = 'l2', 
                                scoring = None, 
                                solver = 'lbfgs', 
                                tol = 1e-4, 
                                max_iter = 2500, 
                                class_weight = 'balanced', 
                                n_jobs = -1, 
                                verbose = 1, 
                                refit = True, 
                                intercept_scaling = 1., 
                                multi_class = 'ovr', 
                                random_state = seed))
])

# Defining hyperparameters
solver = ['lbfgs', 'newton-cg']

param_grid = {
    'clf__solver': solver,
}

# split out validation set
for round in range(num_rounds): # we do this to ensure we cover as much of the larger classes as possible
    seed=np.random.randint(0, 8147)

    # under-sample over-represented classes
    rus = RandomUnderSampler(random_state=seed)
    X_resampled, y_resampled = rus.fit_sample(X, y) # produces numpy arrays

    # splitting validation set
    for train_index, test_index in sss.split(X_resampled, y_resampled):
        # print("TRAIN:", train_index, "TEST:", test_index)
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        # grid search
        grid = GridSearchCV(estimator=lgr_pipe, param_grid=param_grid, scoring=scoring, cv=sss) 
        grid_result = grid.fit(X_train, y_train)

        # print out results and give hyperparameter settings for best one
        means = grid_result.cv_results_['mean_test_score']
        stds = grid_result.cv_results_['std_test_score']
        params = grid_result.cv_results_['params']
        for mean, stdev, param in zip(means, stds, params):
            print("%.2f (%.2f) with: %r" % (mean, stdev, param))
    
        # # print best parameter settings
        print("Best: %.2f using %s" % (grid_result.best_score_, grid_result.best_params_))


        lgr_pipe = lgr_pipe.set_params(**grid_result.best_params_)

        lgr_pipe.fit(X_train,y_train)
      
        # predict test instances when using RUS
        y_pred = lgr_pipe.predict(np.delete(X_resampled, train_index, axis=0))
        y_test = np.delete(y_resampled, train_index, axis=0)
        local_cm = confusion_matrix(y_test, y_pred)
        local_report = classification_report(y_test, y_pred)

        # append coefficients to dataframe
        coef_table = pd.DataFrame(lgr_pipe.named_steps['clf'].coef_, columns=X.columns).T
    
        # combine outputs
        sss_coef = pd.merge(sss_coef, coef_table, left_index=True, right_index=True, how='outer')

        # summarizing results
        local_sss_results = pd.DataFrame([("Accuracy",accuracy_score(y_test, y_pred)), ("params",str(grid_result.best_params_)), ("TRAIN",str(train_index)), ("TEST",str(test_index)), ("CM", local_cm), ("Classification report", local_report), ("y_test", y_test)]).T

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
sss_results.to_csv("D:\Projects\Isotope Samples\ML-analysis\lgr_sssCV_record.csv", index=False)
sss_results = pd.read_csv("D:\Projects\Isotope Samples\ML-analysis\lgr_sssCV_record.csv")

# Accuracy distribution
lgr_acc_distrib = sss_results["Accuracy"]
lgr_acc_distrib.columns=["Accuracy"]
lgr_acc_distrib.to_csv("D:\Projects\Isotope Samples\ML-analysis\lgr_acc_distrib.csv", header=True, index=False)
lgr_acc_distrib = pd.read_csv("D:\Projects\Isotope Samples\ML-analysis\lgr_acc_distrib.csv")
lgr_acc_distrib = np.round(100*lgr_acc_distrib)
print(lgr_acc_distrib)

#%%
# plotting accuracy distribution
plt.figure(figsize=(2.25,3))
sns.distplot(lgr_acc_distrib, kde=False, bins=12)
# plt.savefig("lgr_acc_distrib.png", bbox_inches="tight")

# summarizing coefficients
sss_coef.dropna(axis=1, inplace=True)
sss_coef["coef mean"] = sss_coef.mean(axis=1)
sss_coef["coef sem"] = sss_coef.sem(axis=1)
sss_coef.to_csv("D:\Projects\Isotope Samples\ML-analysis\coef_repeatedCV_.csv")

#%%
sss_coef = pd.read_csv("D:\Projects\Isotope Samples\ML-analysis\coef_repeatedCV_.csv")
sss_coef_1 = sss_coef.rename(columns = {'Unnamed: 0': 'Wavenumbers'})
sss_coef_1.to_csv("D:\Projects\Isotope Samples\ML-analysis\coef_repeatedCV_1.csv")

#%% plotting coefficients
n_features = 10
sss_coef = pd.read_csv("D:\Projects\Isotope Samples\ML-analysis\coef_repeatedCV_1.csv")
sss_coef = sss_coef.reset_index().set_index('Wavenumbers')
sss_coef.sort_values(by="coef mean", ascending=False, inplace=True)
coef_plot_data = sss_coef.drop(["coef sem", "coef mean"], axis=1).T
coef_plot_data = coef_plot_data.iloc[2:,:].drop(coef_plot_data.columns[n_features:-n_features], axis=1)


sns.set(context="paper",
    style="white",
    font_scale=2.0,
    rc={"font.family": "Dejavu Sans"})

plt.figure(figsize=(6,8))
sns.barplot(data=coef_plot_data, orient="h", palette="plasma", capsize=.2)
plt.ylabel("Wavenumbers", weight = "bold")
plt.xlabel("Coeffients", weight = "bold")
plt.savefig("D:\Projects\Isotope Samples\ML-analysis\lgr_coeffients-2.png", dpi = 500, bbox_inches="tight")


#%%
# class distribution 
class_names = y.sort_values().unique()
lgr_per_class_acc_distrib = pd.DataFrame(sss_per_class_results, columns=class_names)
lgr_per_class_acc_distrib.dropna().to_csv("D:\Projects\Isotope Samples\ML-analysis\lgr_per_class_acc_distrib.csv")
lgr_per_class_acc_distrib = pd.read_csv("D:\Projects\Isotope Samples\ML-analysis\lgr_per_class_acc_distrib.csv", index_col=0)
lgr_per_class_acc_distrib = np.round(100*lgr_per_class_acc_distrib)
lgr_per_class_acc_distrib_describe = lgr_per_class_acc_distrib.describe()
lgr_per_class_acc_distrib_describe.to_csv("D:\Projects\Isotope Samples\ML-analysis\lgr_per_class_acc_distrib.csv")

#%%
# plotting class distribution
lgr_per_class_acc_distrib = pd.melt(lgr_per_class_acc_distrib, var_name="Conc new")
sns.set(context="paper",
    style="whitegrid",
    font_scale=2.0,
    rc={"font.family": "Dejavu Sans"})

plt.figure(figsize=(8,6))
sns.violinplot(x="Conc new", y="value", cut = 0, data=lgr_per_class_acc_distrib)
sns.despine(left=True)
plt.xticks(rotation=0, ha="right")
plt.xticks()
plt.ylim(85, 100)
plt.yticks()
plt.xlabel(" ")
plt.ylabel("Prediction accuracy", weight = "bold")
plt.savefig("D:\Projects\Isotope Samples\ML-analysis\lgr_per_class_acc_distrib.png", dpi = 500, bbox_inches="tight")


#%%
# plot the prediction accuracy in a confusion matrix for the training set
sns.set(context="paper",
    style="whitegrid",
    font_scale=2.0,
    rc={"font.family": "Dejavu Sans"})

plt.rcParams["figure.figsize"] = [8,6]
# cm = confusion_matrix(Y_test, Y_pred)
class_names = np.unique(np.sort(y))
plot_confusion_matrix(local_cm, text=True, normalise=True, classes=class_names)
plt.savefig("D:\Projects\Isotope Samples\ML-analysis\prediction all_conc.png", dpi = 500, bbox_inches="tight")


#%%
# Summarising precision, f_score, and recall for the training set
cr = local_report
print(cr)

cr = pd.read_fwf(io.StringIO(cr), header=0)
cr = cr.iloc[1:]
cr.to_csv('classification_report_train.csv')

##############################################################
##############################################################

#%%
# TESTING THE FINAL MODEL IN THE VALIDATION SET (UNSEEN DATA)

# dump/save/serialize the final model to the disk for future prediction. 

with open('D:\Projects\Isotope Samples\ML-analysis\isotope_model.pkl', 'wb') as fid:
    pickle.dump(lgr_pipe, fid)

#%%
validation= pd.read_csv("D:\Projects\Isotope Samples\ML-analysis\set_validation.csv")
print (validation.head(5))


#%%
# checking class distribution in the validation set
class_counts2 = validation.groupby('Cat4').size()
class_counts2

#%%
# Defining X and Y from the validation set
X = validation.iloc[:,9:]
Y = validation["Cat4"]
X

#%%
# Deserializing the final model from the disk to predict new samples

with open('D:\Projects\Isotope Samples\ML-analysis\isotope_model.pkl', 'rb') as fid:
    loaded_model = pickle.load(fid)


#%%
# Evaluating the final model predicting new samples (validation set)
Y_val_pred = loaded_model.predict(X)
# predictions = [round(value) for value in Y_val_pred]
# evaluative prediction
accuracy = accuracy_score(Y, Y_val_pred)
print("Accuracy:%.2f%%" %(accuracy * 100.0))


#%%
# Plot the prediction accuracy in a confusion matrix for the new data
sns.set(context="paper",
   style="whitegrid",
   font_scale = 2.0,
   rc={"font.family": "Dejavu Sans"})
   
plt.rcParams["figure.figsize"] = [8,6]
cm = confusion_matrix(Y, Y_val_pred)
class_names = np.unique(np.sort(Y))
plot_confusion_matrix(cm, text=True, normalise=True, classes=class_names)
plt.savefig("D:\Projects\Isotope Samples\ML-analysis\prediction_val_all.png", dpi = 500, bbox_inches="tight")


#%%
# summarizing classification report for the new data
Cr2 = classification_report(Y, Y_val_pred)
print(Cr2)

cr = pd.read_fwf(io.StringIO(Cr2), header=0)
cr = cr.iloc[1:]
cr.to_csv('classification_report_validation.csv')

#############################################################
#############################################################
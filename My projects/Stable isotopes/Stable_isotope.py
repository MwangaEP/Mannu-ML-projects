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
from scipy import interp

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
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix, precision_recall_fscore_support, mean_squared_error, r2_score, roc_auc_score, roc_curve, auc

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import Normalizer

from imblearn.under_sampling import RandomUnderSampler
# from imblearn.ensemble import EasyEnsemble

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
plt.rcParams["figure.figsize"] = [6,4]

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
                          cmap=plt.cm.Purples):
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
    plt.ylabel('True label', weight = 'bold')
    plt.xlabel('Predicted label', weight = 'bold')


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
print(df.shape) # data shape

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
X_resampled

#%%
# Data splitting and defining models
num_folds = 7 # Spliting the training set into 10 parts
validation_size = 0.25 # defining the size of the validation set
seed = 4 # choose any integer, this ensures reproducibility of the tests
scoring = 'accuracy' # score model accuracy

skf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=seed)

skf = StratifiedShuffleSplit(
        n_splits=num_folds, test_size=validation_size, random_state=seed)

# estimators = []
# model1 = KNeighborsClassifier()
# estimators.append(('KNN', model1))
# model2 = LogisticRegressionCV(multi_class = 'auto', cv=skf, random_state=seed, max_iter=2500)
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

# RF = RandomForestClassifier(n_estimators = 100, random_state=seed)
# num_trees = 300

models = [] # telling python to create sub names models
models.append(("KNN", KNeighborsClassifier()))
models.append(("LR", LogisticRegressionCV(multi_class = 'auto', cv=skf, random_state=seed, max_iter=2500)))
models.append(("SVM", SVC(random_state=seed, kernel='linear', gamma='auto')))
models.append(("NB", GaussianNB()))
models.append(("XGB", XGBClassifier(random_state=seed, nthread=1)))
models.append(("RF", RandomForestClassifier(random_state=seed, n_estimators=200)))
models.append(("MLP", MLPClassifier(random_state=seed, max_iter=2000)))


#%%
# fit the ensembe model
# cv_results = cross_val_score(
#         ensemble, X_new, y_new, cv=skf, scoring=scoring)

# # msg = "Cross val score for {0}: {1:.2%}".format(cv_results.mean())
# print(cv_results.mean())

#%%
# comparative evaluation of different classifiers
X_new = StandardScaler().fit_transform(X_resampled)

results = []
names = []

for name, model in models:
    cv_results = cross_val_score(
        model, X_new, y_resampled, cv=skf, scoring=scoring)
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

plt.figure(figsize=(6,4))
sns.boxplot(x=names, y=results)
sns.despine(offset=10, trim=True)
plt.xticks(rotation=90)
plt.yticks()
plt.ylabel('Accuracy', weight = 'bold');
plt.savefig("D:\Projects\Isotope Samples\ML-analysis\selection_algorithm_2.png", dpi = 500, bbox_inches="tight")

#%%
# preparing training dataset
df = pd.read_csv("D:\Projects\Isotope Samples\ML-analysis\set_training.csv") 
print(df.head(5))

X = df.iloc[:,9:] # matrix of features
y = df["Cat4"] # vector of labels

print(X)
print(y)

# change the matrix of features (which is in pandas dataframe) and 
# vector of labels (which is a list) to numpy arrays to make easy use of loops 
X = np.asarray(X)
y = np.asarray(y)

print(X)
print(y)

#%%

# BIG LOOP
# TUNNING THE SELECTED MODEL
# Using support vector machine classifier, and we will train it more to predict stable isotopes from mosquito samples

# Set validation procedure
num_folds = 7 # split training set into 10 parts for validation
validation_size = 0.25 # size of test set
num_rounds = 5 # increase this to 5 or 10 once code is bug-free
# seed = 4 # pick any integer. This ensures reproducibility of the tests
scoring = 'accuracy' # score model accuracy

# prepare matrices of results
skf_results = pd.DataFrame() # model parameters and global accuracy score
skf_per_class_results = [] # per class accuracy scores
skf_coef = pd.DataFrame() # model coeffients
start = time()

# choose a validation approach
skf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=seed)

# create a pipeline
svc_pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('clf', SVC(C = 1.0, 
                kernel = 'linear', 
                degree = 3, 
                gamma='scale', 
                coef0 = 0.0, 
                shrinking = True, 
                probability = True, 
                tol = 1e-3, 
                cache_size = 200, 
                class_weight = None, 
                verbose = False, 
                max_iter = -1, 
                decision_function_shape = 'ovr', 
                break_ties = False, 
                random_state = seed))

])

# Defining hyperparameters
Cs = [0.1, 1, 10, 100]
Gamma = [1, 0.1, 0.01, 0.001]

param_grid = {
    'clf__C': Cs,
    'clf__gamma': Gamma
}

# split out validation set
for round in range(num_rounds): # we do this to ensure we cover as much of the larger classes as possible
    # under-sample over-represented classes
    rus = RandomUnderSampler(random_state=seed)
    # X_resampled, y_resampled = [None, None]
    X_res, y_res = rus.fit_resample(X, y) # produces numpy arrays

    # splitting validation set
    for train_index, test_index in skf.split(X_res, y_res):
        # print("TRAIN:", train_index, "TEST:", test_index)
        X_train, X_test = X_res[train_index], X_res[test_index]
        y_train, y_test = y_res[train_index], y_res[test_index]

        # grid search
        grid = RandomizedSearchCV(estimator=svc_pipe, param_distributions=param_grid, scoring=scoring, cv=skf) 
        grid_result = grid.fit(X_train, y_train)

        # print out results and give hyperparameter settings for best one
        means = grid_result.cv_results_['mean_test_score']
        stds = grid_result.cv_results_['std_test_score']
        params = grid_result.cv_results_['params']
        for mean, stdev, param in zip(means, stds, params):
            print("%.2f (%.2f) with: %r" % (mean, stdev, param))
    
        # # print best parameter settings
        print("Best: %.2f using %s" % (grid_result.best_score_, grid_result.best_params_))


        svc_pipe = svc_pipe.set_params(**grid_result.best_params_)

        svc_pipe.fit(X_train,y_train)
      
        # predict test instances when using RUS
        y_pred = svc_pipe.predict(np.delete(X_res, train_index, axis=0))
        y_test = np.delete(y_res, train_index, axis=0)
        local_cm = confusion_matrix(y_test, y_pred)
        local_report = classification_report(y_test, y_pred)

        # append coefficients to dataframe
        # coef_table = pd.DataFrame(svc_pipe.named_steps['clf'].coef_).T
        # combine outputs
        skf_coef = pd.merge(skf_coef, coef_table, left_index=True, right_index=True, how='outer')

        # summarizing results
        local_skf_results = pd.DataFrame([("Accuracy",accuracy_score(y_test, y_pred)), ("params",str(grid_result.best_params_)), ("TRAIN",str(train_index)), ("TEST",str(test_index)), ("CM", local_cm), ("Classification report", local_report), ("y_test", y_test)]).T

        local_skf_results.columns=local_skf_results.iloc[0]
        local_skf_results = local_skf_results[1:]
        skf_results = skf_results.append(local_skf_results)

        # per class accuracy
        local_support = precision_recall_fscore_support(y_test, y_pred)[3]
        local_acc = np.diag(local_cm)/local_support
        skf_per_class_results.append(local_acc)

elapsed = time() - start
print("Time elapsed: {0:.2f} minutes ({1:.1f} sec)".format(
    elapsed / 60, elapsed))


#%%
# Plot roc-curve with cross-validation

tprs = []
aucs = []
mean_fpr = np.linspace(0, 1, 100)
plt.figure(figsize=(8, 6))

sns.set(context="paper",
        style="whitegrid",
        palette="deep",
        font_scale=2.0,
        color_codes=True,
        rc=({"font.family": "Dejavu Sans"}))

i = 0
for round in range(num_rounds): # we do this to ensure we cover as much of the larger classes as possible
    # under-sample over-represented classes
    rus = RandomUnderSampler(random_state=seed)
    X_res, y_res = rus.fit_resample(X, y) # produces numpy arrays

    for train_index, test_index in skf.split(X_res, y_res):
        X_train, X_test = X_res[train_index], X_res[test_index]
        y_train, y_test = y_res[train_index], y_res[test_index]

        probas_ = svc_pipe.fit(X_train, y_train).predict_proba(np.delete(X_res, train_index, axis=0))
        # Compute ROC curve and area the curve
        y_test = np.delete(y_res, train_index, axis=0)
        fpr, tpr, thresholds = roc_curve(y_test, probas_[:, 1], pos_label= 'Treatment')
        tprs.append(interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)
        plt.plot(fpr, tpr, lw=1, alpha=0.3)
                # label='ROC fold %d (AUC = %0.2f)' % (i, roc_auc))

        i += 1

plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
        label='Chance', alpha=.8)

mean_tpr = np.mean(tprs, axis=0)
mean_tpr[-1] = 1.0
mean_auc = auc(mean_fpr, mean_tpr)
std_auc = np.std(aucs)

plt.plot(mean_fpr, mean_tpr, color='b',
         label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
         lw=2, alpha=.8)

std_tpr = np.std(tprs, axis=0)
tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                 label=r'$\pm$ 1 std. dev.')


plt.xlim([-0.01, 1.01])
plt.ylim([-0.01, 1.01])
plt.xlabel('False Positive Rate', weight = 'bold')
plt.ylabel('True Positive Rate', weight = 'bold')
# plt.title('Cross-Validation ROC of SVM')
plt.legend(loc="lower right", prop={'size': 15})
plt.savefig("D:\Projects\Isotope Samples\ML-analysis\cross_v_roc_curve.png", dpi = 500, bbox_inches="tight")
plt.show()

#%%
# Results
skf_results.to_csv("D:\Projects\Isotope Samples\ML-analysis\svc_skfCV_record.csv", index=False)
skf_results = pd.read_csv("D:\Projects\Isotope Samples\ML-analysis\svc_skfCV_record.csv")

# Accuracy distribution
svc_acc_distrib = skf_results["Accuracy"]
svc_acc_distrib.columns=["Accuracy"]
svc_acc_distrib.to_csv("D:\Projects\Isotope Samples\ML-analysis\svc_acc_distrib.csv", header=True, index=False)
svc_acc_distrib = pd.read_csv("D:\Projects\Isotope Samples\ML-analysis\svc_acc_distrib.csv")
svc_acc_distrib = np.round(svc_acc_distrib, 2)
print(svc_acc_distrib)

#%%
# plotting accuracy distribution
# plt.figure(figsize=(2.25,3))
# sns.distplot(svc_acc_distrib, kde=False, bins=12)
# plt.savefig("svc_acc_distrib.png", bbox_inches="tight")

# summarizing coefficients
skf_coef.dropna(axis=1, inplace=True)
skf_coef["coef mean"] = skf_coef.mean(axis=1)
skf_coef["coef sem"] = skf_coef.sem(axis=1)
print(skf_coef)

skf_coef.to_csv("D:\Projects\Isotope Samples\ML-analysis\coef_repeatedCV_.csv")

#%%
# coef_1 = pd.read_csv("D:\Projects\Isotope Samples\ML-analysis\coef_repeatedCV_.csv")
# print(coef_1)
# coef_2 = coef_1.rename(columns = {'Unnamed: 0': 'Wavenumbers'})
# coef_2.to_csv("D:\Projects\Isotope Samples\ML-analysis\coef_repeatedCV_1.csv")

#%% plotting coefficients
# n_features = 10
# skf_coef = pd.read_csv("D:\Projects\Isotope Samples\ML-analysis\coef_repeatedCV_1.csv")
# skf_coef = skf_coef.reset_index().set_index('Wavenumbers')
# skf_coef.sort_values(by="coef mean", ascending=False, inplace=True)
# coef_plot_data = skf_coef.drop(["coef sem", "coef mean"], axis=1).T
# coef_plot_data = coef_plot_data.iloc[2:,:].drop(coef_plot_data.columns[n_features:-n_features], axis=1)


# sns.set(context="paper",
#     style="white",
#     font_scale=2.0,
#     rc={"font.family": "Dejavu Sans"})

# plt.figure(figsize=(6,8))
# sns.barplot(data=coef_plot_data, orient="h", palette="plasma", capsize=.2)
# plt.ylabel("Wavenumbers", weight = "bold")
# plt.xlabel("Coeffients", weight = "bold")
# plt.savefig("D:\Projects\Isotope Samples\ML-analysis\svc_coeffients-2.png", dpi = 500, bbox_inches="tight")


#%%
# class distribution 
# class_names = y.sort_values().unique()
class_names = np.unique(np.sort(y))
svc_per_class_acc_distrib = pd.DataFrame(skf_per_class_results, columns=class_names)
svc_per_class_acc_distrib.dropna().to_csv("D:\Projects\Isotope Samples\ML-analysis\svc_per_class_acc_distrib.csv")
svc_per_class_acc_distrib = pd.read_csv("D:\Projects\Isotope Samples\ML-analysis\svc_per_class_acc_distrib.csv", index_col=0)
svc_per_class_acc_distrib = np.round(svc_per_class_acc_distrib, 1)
svc_per_class_acc_distrib_describe = svc_per_class_acc_distrib.describe()
svc_per_class_acc_distrib_describe.to_csv("D:\Projects\Isotope Samples\ML-analysis\svc_per_class_acc_distrib.csv")

#%%
# plotting class distribution
svc_per_class_acc_distrib = pd.melt(svc_per_class_acc_distrib, var_name="Conc new")
sns.set(context="paper",
    style="whitegrid",
    font_scale=2.0,
    rc={"font.family": "Dejavu Sans"})

plt.figure(figsize=(6,4))
sns.pointplot(x="Conc new", y="value", join=False, hue = "Conc new", 
                capsize = .1, scale= 4.5, errwidth = 4,
                data=svc_per_class_acc_distrib)
sns.despine(left=True)
plt.xticks(rotation=0, ha="right")
plt.yticks()
plt.ylim()
plt.xlabel(" ")
plt.legend('')
plt.ylabel("Prediction accuracy", weight = "bold")
plt.savefig("D:\Projects\Isotope Samples\ML-analysis\svc_per_class_acc_distrib.png", dpi = 500, bbox_inches="tight")


#%%
# plot the prediction accuracy in a confusion matrix for the training set
sns.set(context="paper",
    style="whitegrid",
    font_scale=2.0,
    rc={"font.family": "Dejavu Sans"})

plt.figure(figsize=(6,4))
# cm = confusion_matrix(Y_test, Y_pred)
class_names = np.unique(np.sort(y))
plot_confusion_matrix(local_cm, text=True, normalise=True, classes=class_names)
plt.savefig("D:\Projects\Isotope Samples\ML-analysis\conf_matrix_lab_train.png", dpi = 500, bbox_inches="tight")


#%%
# Summarising precision, f_score, and recall for the training set
cr = local_report
print(cr)

cr = pd.read_fwf(io.StringIO(cr), header=0)
cr = cr.iloc[1:]
cr.to_csv('D:\Projects\Isotope Samples\ML-analysis\classification_report_train.csv')

# dump/save/serialize the final model to the disk for future prediction. 

with open('D:\Projects\Isotope Samples\ML-analysis\isotope_model_lab.pkl', 'wb') as fid:
    pickle.dump(svc_pipe, fid)

##############################################################
##############################################################

# TESTING THE MODEL ON THE VALIDATION SET

#%%
# Import validation set
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

# seed = 4
# rus = RandomUnderSampler(random_state = seed)
# X_res, y_res = rus.fit_sample(X, Y)
# y_res_count = collections.Counter(y_res)
# print(y_res_count)

#%%
# Deserializing the final model from the disk to predict new samples

with open('D:\Projects\Isotope Samples\ML-analysis\isotope_model_lab.pkl', 'rb') as fid:
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
   
cm = confusion_matrix(Y, Y_val_pred)
class_names = np.unique(np.sort(y))
plot_confusion_matrix(cm, text=True, normalise=True, classes=class_names)
plt.savefig("D:\Projects\Isotope Samples\ML-analysis\conf_matrix_lab_valid.png", dpi = 500, bbox_inches="tight")


#%%
# summarizing classification report for the new data
Cr2 = classification_report(Y, Y_val_pred)
print(Cr2)

cr = pd.read_fwf(io.StringIO(Cr2), header=0)
cr = cr.iloc[1:]
cr.to_csv('D:\Projects\Isotope Samples\ML-analysis\classification_report_validation.csv')

#%%
# plot number of samples predicted
# Prepare datasets for no. of samples in validation set 

no_val = pd.read_csv('D:\Projects\Isotope Samples\ML-analysis\classification_report_validation.csv')
print(no_val.head(5))

no_val = no_val.drop('Unnamed: 0' , axis='columns')
no_val = no_val.rename(columns = {'Unnamed: 0.1': 'Labels'})
no_val = no_val[:2]
no_val['Data_set'] = ['validation', 'validation']

# Prepare datasets for no. of samples in test set 
no_train = pd.read_csv('D:\Projects\Isotope Samples\ML-analysis\classification_report_train.csv')
print(no_train.head(5))

no_train = no_train.drop('Unnamed: 0' , axis='columns')
no_train = no_train.rename(columns = {'Unnamed: 0.1': 'Labels'})
no_train = no_train[:2]
no_train['Data_set'] = ['test', 'test']

# Stack the DataFrames on top of each other
new_dat = pd.concat([no_train, no_val], axis=0)
print(new_dat)

# plotting
sns.set(context="paper",
   style="darkgrid",
   font_scale = 2.0,
   rc={"font.family": "Dejavu Sans"})

plt.figure(figsize=(6,4))
ax = sns.barplot(x="Labels", y="support", data=new_dat, hue= "Data_set")
plt.xlabel(" ")
ax.set_ylim(0, 180)
plt.ylabel("Total no. of samples used for evaluation", weight = 'bold')

#annotate axis = seaborn axis
for p in ax.patches:
             ax.annotate("%.2f" % p.get_height(), (p.get_x() + p.get_width() / 2., p.get_height()),
                 ha='center', va='center', fontsize=15, color='gray', xytext=(0, 20),
                 textcoords='offset points')

plt.savefig("D:\Projects\Isotope Samples\ML-analysis\samples_total used for evaluation_balanced.png", dpi = 500, bbox_inches="tight")

#############################################################
#############################################################

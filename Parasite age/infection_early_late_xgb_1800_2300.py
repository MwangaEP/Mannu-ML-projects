
#%%
# This program uses standard machine learning to learn parasite age

# import all libraries

import os
import io
import ast
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

from random import randint
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
                                f1_score, 
                                recall_score, 
                                precision_score, 
                                precision_recall_fscore_support
                            )

# from mlxtend.feature_selection import SequentialFeatureSelector as SFS
# from mlxtend.plotting import plot_sequential_feature_selection as plot_sfs

from imblearn.under_sampling import RandomUnderSampler

# from sklearn.linear_model import LogisticRegressionCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
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
            rc = ({'font.family': 'Dejavu Sans'})
        )

%matplotlib inline
plt.rcParams["figure.figsize"] = [6, 4]

#%%

# This normalizes the confusion matrix and ensures neat plotting for all outputs.
# Function for plotting confusion matrcies

def plot_confusion_matrix(cm, classes,
                          normalize = True,
                          title = 'Confusion matrix',
                          xrotation=0,
                          yrotation=0,
                          cmap=plt.cm.Purples,
                          printout = False):
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
    plt.xticks(
                tick_marks, 
                classes, 
                rotation = xrotation
            )

    plt.yticks(
                tick_marks, 
                classes, 
                rotation = yrotation
            )

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(
                    j, 
                    i, 
                    format(cm[i, j], fmt),
                    horizontalalignment = "center",
                    color = "white" if cm[i, j] > thresh else "black"
                )

    plt.tight_layout()
    plt.ylabel('True label', weight = 'bold')
    plt.xlabel('Predicted label', weight = 'bold')
    plt.savefig(
                    (
                        r"C:\Mannu\Projects\Mwanga-DBS work\Parasite age\ML analysis\XGB\Confusion_Matrix_" 
                        + figure_name 
                        + "_" 
                        + ".png"
                    ), 
                    dpi = 500, 
                    bbox_inches = "tight"
                )

#%%
#  Define the base directory
base_directory = r"C:\Mannu\Projects\Mwanga-DBS work\Parasite age\ML analysis\XGB"

# Create a function to generate paths within the base directory
def generate_path(*args):
    return os.path.join(base_directory, *args)

#%%
# Visualizing outputs
# for visualizing confusion matrix once the model is trained

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
# importing dataframe 

# Loading dataset  
# Upload infection data

par_df = pd.read_csv(
                        "C:\Mannu\Projects\Mwanga-DBS work\Parasite age\Transformed\Age_spectra_raw_clean.dat", 
                        delimiter = '\t'
                    )

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

# Drop control from the data and classify only early eings
 
par_df = par_df.loc[par_df['Treatment'] != 'Late control']
par_df = par_df.loc[par_df['Treatment'] != 'Late rings']

# count the number of levels in the column and their size
print(Counter(par_df["Treatment"])) # count the number of levels in the column and their size


#%%
# Split training and test data

train_set, test_set = train_test_split(
                                        par_df, 
                                        stratify = par_df[["Treatment", "Cat2"]], 
                                        test_size = 0.2, 
                                        shuffle = True, 
                                        random_state = 42
                                    )

# Count the number of samples in treatment group

print('number of samples per infection status : {}'.format(Counter(train_set['Treatment'])))

#%%

# drop the column with age as a string and keep the age in intergers

train_set_2 = train_set.drop(
                        [
                            'Cat1', 
                            'Cat2', 
                            'Cat3',
                            'Cat5' 
                        ], axis = 1
                    )

train_set_2.head(5)

# Checking class distribution in the data
print(Counter(train_set_2["Treatment"]))


#%%

# Since the resolution of the spectra data is 2cm, 2300 may not be available in the colummn names, 
# Check whats the closest number

# drop treatment column
temp = train_set_2.drop(['Treatment'], axis = 1)

# make a list containing all column names
column_names = temp.columns.tolist()

column_names = [int(x) for x in column_names]

# get the closest wavenumbers
print(list(map(lambda y:min(column_names, key=lambda x:abs(x - y)), [2300])))
print(list(map(lambda y:min(column_names, key=lambda x:abs(x - y)), [1800])))

#%%

# set the start and end column names as integers
start_col = 2301
end_col = 1801

# get the column names between the start and end column
cols_to_drop = [str(i) for i in range(start_col, end_col - 1, -2)]

train_set_3 = train_set_2.drop(cols_to_drop, axis = 1)

train_set_3.head()

#%%

# define X (matrix of features) and y (list of labels)

X = np.asarray(train_set_3.iloc[:,:-1]) # select all columns except the first one 
y = np.asarray(train_set_3["Treatment"])

# Balance the training data
rus = RandomUnderSampler(random_state = 42)
X_res, y_res = rus.fit_resample(X, y)

# scale data
scaler = StandardScaler().fit(X = X_res)
scl_features = scaler.transform(X = X_res)

#%%

# define parameters
num_folds = 5 # split data into five folds
seed = np.random.randint(0, 81470) # random seed value
scoring = 'accuracy' # metric for model evaluation

# specify cross-validation strategy
kf = KFold(
            n_splits = num_folds, 
            shuffle = True, 
            random_state = seed
        )

# make a list of models to test
models = []
models.append(
                (
                    'KNN', KNeighborsClassifier()
                )
            )

models.append(
                (
                    'LR', LogisticRegression(
                                                multi_class = 'ovr', 
                                                max_iter = 2000, 
                                                random_state = seed
                                            )
                )
            )

models.append(
                (
                    'SVM', SVC(
                                kernel = 'linear', 
                                gamma = 'auto', 
                                random_state = seed
                            )
                )
            )

models.append(
                (
                    'RF', RandomForestClassifier(
                                                    n_estimators = 500, 
                                                    random_state = seed
                                                )
                )
            )

models.append(
                ('XGB', XGBClassifier(
                                        random_state = seed, 
                                        n_estimators = 500
                                    )
                )
            )

#%%

# Evaluate models to get the best perfoming model

results = []
names = []

for name, model in models:
    cv_results = cross_val_score(
                                    model, 
                                    scl_features, 
                                    y_res, 
                                    cv = kf, 
                                    scoring = scoring
                                )
    results.append(cv_results)
    names.append(name)
    msg = 'Cross validation score for {0}: {1:.2%}'.format(
                                                            name, 
                                                            cv_results.mean(), 
                                                            cv_results.std()
                                                        )
    print(msg)

#%%

results_df = pd.DataFrame(
                            results, 
                            columns = (0, 1, 2, 3, 4)
                        ).T # columns should correspond to the number of folds, k = 5

# rename columns to have number of components

# results_df.columns = ['KNN', 'LR', 'SVM', 'RF', 'XGB']
results_df.columns = names
results_df = pd.melt(results_df) # melt data frame into a long format. 
results_df.rename(
                    columns = {'variable':'Model', 'value':'Accuracy'}, 
                    inplace = True
                )
results_df


# Plotting the algorithm selection 

plt.figure(figsize = (6, 4))

sns.boxplot(
                x = results_df['Model'], 
                y = results_df['Accuracy']
            )
# sns.boxplot(x = names, y = results, width = .4)
sns.despine(offset = 10, trim = True)
plt.xticks(rotation = 90)
plt.yticks(np.arange(0.2, 1.0 + .1, step = 0.2))
plt.ylabel('Accuracy', weight = 'bold')
plt.xlabel(" ")
plt.tight_layout()

# create path for plot
algorithm_sel_path = generate_path('algorithm_sel.png')
plt.savefig(
                algorithm_sel_path, 
                bbox_inches = 'tight', 
                dpi = 500
            )


#%%
# train XGB classifier and tune its hyper-parameters with randomized grid search 

# define parameters
scoring = 'accuracy' # metric for model evaluation

classifier = XGBClassifier(random_state = seed)

# set hyparameter

estimators = [300, 500, 800, 1000, 1200]
rate = [0.05, 0.10, 0.15, 0.20, 0.30]
depth = [2, 3, 4, 5, 6, 8, 10, 12, 15]
child_weight = [1, 3, 5, 7]
gamma = [0.0, 0.1, 0.2, 0.3, 0.4]
bytree = [0.1, 0.2, 0.3, 0.4, 0.5, 0.7]

random_grid = {'n_estimators': estimators, 
              'learning_rate': rate, 
              'max_depth': depth,
              'min_child_weight': child_weight, 
              'gamma': gamma, 
              'colsample_bytree': bytree}

# prepare matrices of results
kf_results = pd.DataFrame() # model parameters and global accuracy score
kf_per_class_results = [] # per class accuracy scores
sss_coef = pd.DataFrame() 

save_predicted = [] # save predicted values for plotting averaged confusion matrix
save_true = [] # save true values for plotting averaged confusion matrix
num_rounds = 5

start = time()

for round in range(num_rounds):
    SEED = np.random.randint(0, 81470)

    for train_index, test_index in kf.split(scl_features, y_res):

        # Split data into test and train

        X_train_set, X_val = scl_features[train_index], scl_features[test_index]
        y_train_set, y_val = y_res[train_index], y_res[test_index]

        # check the shape the splits
        print(X_train_set.shape)
        print(y_train_set.shape)
        print(X_val.shape)
        print(y_val.shape)

        # RANDOMSED GRID SEARCH
        n_iter_search = 10
        rsCV = RandomizedSearchCV(
                                    verbose = 1, 
                                    estimator = classifier, 
                                    param_distributions = random_grid, 
                                    n_iter = n_iter_search, 
                                    scoring = scoring, 
                                    cv = kf
                                )
        
        rsCV_result = rsCV.fit(X_train_set, y_train_set)

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
        classifier = classifier.set_params(**rsCV_result.best_params_)

        # Fitting the best classifier
        classifier.fit(X_train_set, y_train_set)

        # Predict X_test
        y_pred = classifier.predict(X_val)

        # Summarize outputs for plotting averaged confusion matrix

        for predicted, true in zip(y_pred, y_val):
            save_predicted.append(predicted)
            save_true.append(true)

        # summarize for plotting per class distribution

        local_cm = confusion_matrix(y_val, y_pred)
        local_report = classification_report(y_val, y_pred)

        # append feauture importances
        local_feat_impces = pd.DataFrame(classifier.feature_importances_,
                                            index = train_set_3.iloc[:,:-1].columns).sort_values(by = 0, ascending = False)
        
        # summarizing results
        local_kf_results = pd.DataFrame([("Accuracy", accuracy_score(y_val, y_pred)), 
                                            ("TRAIN",str(train_index)), 
                                            ("TEST",str(test_index)), 
                                            ("CM", local_cm), 
                                            ("Classification report", local_report), 
                                            ("y_test", y_val),
                                            ("Feature importances", local_feat_impces.to_dict())]).T   

        local_kf_results.columns = local_kf_results.iloc[0]
        local_kf_results = local_kf_results[1:]
        kf_results = pd.concat(
                    [
                        kf_results, 
                        local_kf_results
                    ], 
                    axis = 0, 
                    join = 'outer'
                ).reset_index(drop = True)

        # per class accuracy
        local_support = precision_recall_fscore_support(y_val, y_pred)[3]
        local_acc = np.diag(local_cm)/local_support
        kf_per_class_results.append(local_acc)

# save the trained model to disk for future use

# classifier.save_model('C:\Mannu\QMBCE\Thesis\Fold\Standard_ml\_ml_2\classifier')

elapsed = time() - start
print("Time elapsed: {0:.2f} minutes ({1:.1f} sec)".format(
    elapsed / 60, elapsed))

# %%

# plot confusion averaged for the validation set
figure_name = 'baseline_model'
classes = np.unique(np.sort(y_val))
# plotting class distribution

plt.figure(figsize = (8, 6))
visualize(
            figure_name, 
            classes, 
            save_true, 
            save_predicted
        )

# %%
# preparing dataframe for plotting per class accuracy

classes =  np.unique(np.sort(y_val))
rf_per_class_acc_distrib = pd.DataFrame(kf_per_class_results, columns = classes)
rf_per_class_acc_distrib.dropna().to_csv(generate_path('_rf_per_class_acc_distrib.csv'))
rf_per_class_acc_distrib = pd.read_csv(generate_path("_rf_per_class_acc_distrib.csv"), index_col=0)
rf_per_class_acc_distrib = np.round(rf_per_class_acc_distrib, 1)
rf_per_class_acc_distrib_describe = rf_per_class_acc_distrib.describe()
rf_per_class_acc_distrib_describe.to_csv(generate_path("_rf_per_class_acc_distrib.csv"))

#%%
# plotting class distribution


plt.figure(figsize = (6, 4))

rf_per_class_acc_distrib = pd.melt(rf_per_class_acc_distrib, var_name="Label new")
# g = sns.pointplot(x="Label new", y="value", join = False, hue = "Label new",
#                 capsize = .1, scale= 4.5, errwidth = 4,
                # data = rf_per_class_acc_distrib)
g = sns.violinplot(x="Label new", y="value", hue = "Label new",
                data = rf_per_class_acc_distrib)

sns.despine(left=True)
plt.xticks(ha="right")
plt.yticks()
plt.ylim(ymin=0.5, ymax=1.0)
plt.xlabel(" ")
g.legend().set_visible(False)
# plt.legend(' ', frameon = False)
plt.ylabel("Prediction accuracy", weight = 'bold')
plt.grid(False)
plt.tight_layout()
plt.savefig(
                generate_path('_rf_per_class_acc_distrib.png'), 
                dpi = 500, 
                bbox_inches = "tight"
            )


#%%

test_set_2 = test_set.drop(
                        [
                            'Cat1', 
                            'Cat2', 
                            'Cat3',
                            'Cat5' 
                        ], axis = 1
                    )

test_set_2.head(5)

# Checking class distribution in the test data
print(Counter(test_set_2["Treatment"]))

# drop wavenumbers that are just noise 
test_set_3 = test_set_2.drop(
                                cols_to_drop, 
                                axis = 1
                            )

test_set_3.head()

#%%

# define X (matrix of features) and y (list of labels)

X_test = np.asarray(test_set_3.iloc[:,:-1]) # select all columns except the last
y_test = np.asarray(test_set_3["Treatment"])

# generates output predictions based X_test passed

# Scale features
X_test_scl = scaler.transform(X = X_test)

# Prediction
predictions = classifier.predict(X_test_scl)

# Examine the accuracy of the model in predicting glasgow data 

accuracy = accuracy_score(y_test, predictions)
print("Accuracy:%.2f%%" %(accuracy * 100.0))

# compute precision, recall and f-score metrics

cr_pca = classification_report(y_test, predictions, labels = classes)
print(cr_pca)

#%%

# save classification report to disk as a csv

cr = pd.read_fwf(io.StringIO(cr_pca), header=0)
cr = cr.iloc[0:]
cr.to_csv(generate_path('classification_report.csv'))

#%%

# plot the confusion matrix for the test data 

figure_name = 'test_set'
classes = np.unique(np.sort(y_test))
visualize(figure_name, classes, predictions, y_test)



# %%
# make this into bar with error bars across all best models
# Results

kf_results.to_csv(generate_path('crf_kfCV_record.csv'), index = False)
rskf_results = pd.read_csv(generate_path('crf_kfCV_record.csv'))

crf_acc_distrib = kf_results["Accuracy"]
crf_acc_distrib.columns=["Accuracy"]

crf_acc_distrib.to_csv(
                        generate_path("crf_acc_distrib.csv"), 
                        header = True, 
                        index = False
                    )

crf_acc_distrib = pd.read_csv(generate_path("crf_acc_distrib.csv"))
crf_acc_distrib = np.round(crf_acc_distrib, 2)
print(crf_acc_distrib)

# All feat imp
all_featimp = pd.DataFrame(ast.literal_eval(rskf_results["Feature importances"][0]))

for featimp in rskf_results["Feature importances"][1:]:
    featimp = pd.DataFrame(ast.literal_eval(featimp))
    all_featimp = pd.concat(
                                [
                                    all_featimp, 
                                    featimp
                                ], 
                                axis = 1, 
                                ignore_index = True
                            )

all_featimp["mean"] = all_featimp.mean(axis=1)
all_featimp["sem"] = all_featimp.sem(axis=1)
all_featimp.sort_values(by="mean", inplace=True)

featimp_global_mean = all_featimp["mean"].mean()
featimp_global_sem = all_featimp["mean"].sem()

sns.set(context="paper",
    style="white",
    font_scale=2.0,
    rc={"font.family": "Dejavu Sans"})


fig = all_featimp["mean"][-50:].plot(figsize=(3, 14),
                                    kind="barh",
                                    # orientation = 'vertical',
                                    legend=False,
                                    xerr=all_featimp["sem"],
                                    ecolor='k')
plt.xlabel("Feature importance", weight = 'bold')
# plt.axvspan(xmin=0, xmax=featimp_global_mean+3*featimp_global_sem,facecolor='r', alpha=0.3)
# plt.axvline(x=featimp_global_mean, color="r", ls="--", dash_capstyle="butt")
sns.despine()

# Add mean accuracy of best models to plots
plt.annotate(
                "Average Accuracy:\n{0:.3f} ± {1:.3f}".format(
                                                                crf_acc_distrib.mean()[0], 
                                                                crf_acc_distrib.sem()[0]
                                                            ), 
                                                            xy = (0.06, 0), 
                                                            color = "k"
            )

plt.savefig(
                generate_path("_feature_impces_full_wn.png"), 
                dpi = 500, 
                bbox_inches = "tight"
            )

# %%

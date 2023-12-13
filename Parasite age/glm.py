
#%%

import json

import numpy as np 
import pandas as pd

import statsmodels.api as sm
import statsmodels.formula.api as smf
import scipy.stats as stats
import pymer4.models as pmm 
from statsmodels.graphics.regressionplots import plot_leverage_resid2

from collections import Counter 

import matplotlib.pyplot as plt # for making plots
import seaborn as sns
sns.set(context = "paper",
        style = "white",
        palette = "deep",
        font_scale = 2.0,
        color_codes = True,
        rc = None)
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

# rename factors in variable (Cat1) to make define age

Age, Infection = [], []

for row in par_df['Cat1']:

    if row == 'EC':
        Age.append('Early control')
    
    elif row == 'LC':
        Age.append('Late control')
    
    elif row == 'ET':
        Age.append('Early rings')

    else:
        Age.append('Late rings')


# Append age column to the dataframe
par_df['Age'] = Age

# rename factors in variable (Cat1) to make define infection status

for row in par_df['Cat1']:

    if row == 'EC':
        Infection.append('Negative')
    
    elif row == 'LC':
        Infection.append('Negative')
    
    elif row == 'ET':
        Infection.append('Positive')

    else:
        Infection.append('Positive')


par_df['Infection'] = Infection

# drop unused columns

par_df_2 = par_df.drop(['Cat1', 'Cat2', 'Cat3', 'Cat5'], axis = 1) 
par_df_2.head(5)

# Checking class distribution in the data
print(Counter(par_df_2["Age"]))
print(Counter(par_df_2["Infection"]))

# par_df_2.to_csv("C:\Mannu\Projects\Mwanga-DBS work\Parasite age\Background_wn_removed\late control vs late rings\glm_df.csv", index=False)


#%%
# # Drop control from the data and classify only
 
# par_df_2 = par_df_2.loc[par_df_2['Age'] != 'Early control']
# par_df_2 = par_df_2.loc[par_df_2['Age'] != 'Early rings']

# print(Counter(par_df_2["Treatment"])) # count the number of levels in the column and their size


# Since the resolution of the spectra data is 2cm, 2300 may not be available in the colummn names, 
# Check whats the closest number

# drop treatment column temporarly
temp_df = par_df_2.drop(['Age', 'Infection'], axis = 1)

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

# Early rings and Early control datasets

# par_df_early = par_df_2.loc[par_df_2['Age'] != 'Early control']
# par_df_early = par_df_early.loc[par_df_early['Age'] != 'Early rings']

# count the number of levels in the column and their size
print(Counter(par_df_2 ["Age"]))

# loading important wavenumbers for only early control vs early rings

with open('C:\Mannu\Projects\Mwanga-DBS work\Parasite age\Background_wn_removed\late control vs late rings lgr\important_wavenumbers.txt') as json_file:
    important_wavenumb_early = json.load(json_file)

important_wavenumb_early = important_wavenumb_early[:10]
important_wavenumb_early.append("Age")
important_wavenumb_early.append("Infection")

print(important_wavenumb_early)

# select the columns
par_early_selected_df = par_df_2.loc[:, important_wavenumb_early]

# par_early_selected_df.loc[par_early_selected_df.Age == 'Early control', 'Age'] = 'Early'
# par_early_selected_df.loc[par_early_selected_df.Age == 'Early rings', 'Age'] = 'Early'
# par_early_selected_df.loc[par_early_selected_df.Age == 'Late control', 'Age'] = 'Late'
# par_early_selected_df.loc[par_early_selected_df.Age == 'Late rings', 'Age'] = 'Late'

par_early_selected_df['Age'].replace('Early control', 'Early', inplace=True)
par_early_selected_df['Age'].replace('Early rings', 'Early', inplace=True)
par_early_selected_df['Age'].replace('Late control', 'Late', inplace=True)
par_early_selected_df['Age'].replace('Late rings', 'Late', inplace=True)

# par_early_selected_df.to_csv("C:\Mannu\Projects\Mwanga-DBS work\Parasite age\Background_wn_removed\late control vs late rings\glm_earlyrings_df.csv", index=False)
par_early_selected_df

#%%

# Subset the data to only include the variables of interest
early_data_1 = par_early_selected_df[['3639', 'Age', 'Infection' ]]

# Convert 'age' and 'Infection' columns to category data type
early_data_1['Age'] = early_data_1['Age'].astype('category')
early_data_1['Infection'] = early_data_1['Infection'].astype('category')

###################

# Balance categories in the data
# Group the data by the two columns and get the size of each group
group_sizes = early_data_1.groupby(['Infection', 'Age']).size()

# Find the minimum group size
min_size = group_sizes.min()

# Sample min_size number of rows from each group to create a new dataframe with balanced classes
balanced_data = early_data_1.groupby(['Infection', 'Age'], group_keys=False).apply(lambda x: x.sample(min_size))
print(balanced_data.groupby(['Infection', 'Age']).size())

# We will use the likelihood functions from both models to test whether the alternative 
# model fits the data better, compared to the null model. Mathematically, the comparison is made 
# easier by using the log of the likelihoods. The statistic calculated from a likelihood ratio 
# test follows a chi-square distribution with degrees of freedom equal to the 
# difference in degrees of freedom of the two models

# stats.chisqprob = lambda chisq, df: stats.chi2.sf(chisq, df)

# # def lrtest(llmax, llmin):
# #     lr = 2 * (llmax - llmin) 
# #     p = stats.chisqprob(lr, 1) # llmax has 1 degrees of freedom more than llmin
# #     return lr, p

full_model = sm.GLM.from_formula("Q('3639') ~ Age + Infection + Age:Infection", data=balanced_data, family=sm.families.Gaussian())
full_model_results = full_model.fit()
print(full_model_results.summary())

# The llf attribute is generated for each model—this is the log likelihood statistic. 
# The likelihood ratio test then compares the log likelihood values and tests whether the 
# alternative model is significantly different to the null model. 

# Get the log-likelihood statistic for the full model
# llf_full_model = np.round(full_model_results.llf, 2) 

reduced_model = sm.GLM.from_formula("Q('3639') ~ Age + Infection", data=balanced_data, family=sm.families.Gaussian())
reduced_model_results = reduced_model.fit()
print(reduced_model_results.summary())

# Get the log-likelihood statistic for the reduced model
# llf_reduced_model = np.round(reduced_model_results.llf, 2)

# Perform likelihood ratio test
# lr, p = lrtest(llf_full_model , llf_reduced_model)

# # Print the test results
# print('LR test, p value: {:.2f}, {:.4f}'.format(lr, p))

# Calculate the likelihood ratio test statistic and p-value

# calculate the likelihood ratio test statistic as lr, and the p-value as p. The stats.chi2.cdf function 
# calculates the cumulative distribution function of the chi-square distribution with degrees of freedom 
# equal to the difference in the residual degrees of freedom between the two models. The p-value is then 
# calculated as 1 - cdf.

lr = full_model_results.deviance - reduced_model_results.deviance
p = 1 - stats.chi2.cdf(lr, reduced_model_results.df_resid - full_model_results.df_resid)

# Print the results
print('Likelihood Ratio Test:')
print('LR Statistic:', lr)
print('p-value:', p)

#%%

# compare the coefficients for Infection in M2 and M3, as this will indicate the size of the effect of Infection 
# that is actually due to Age.

reduced_model = sm.GLM.from_formula("Q('3639') ~ Age + Infection", data=balanced_data, family=sm.families.Gaussian())
reduced_model_results = reduced_model.fit()
print(reduced_model_results.summary())

reduced_model_age = sm.GLM.from_formula("Q('3639') ~ Age", data=balanced_data, family=sm.families.Gaussian())
reduced_model_age_results = reduced_model_age.fit()
print(reduced_model_age_results.summary())

# %%


# # specify model formula
# formula = "Q('3645') + Q('2437') ~ Age + Infection"

# # fit binomial GLM
# model = sm.formula.glm(formula = formula, data = balanced_data, family=sm.families.Binomial()).fit()

# # print model summary
# print(model.summary())

# %%

test_df = pd.read_csv(r"C:\Mannu\Projects\Training ML\NgowoDataFinal23.csv")

test_df['Village'] = test_df['Village'].astype('category')
# test_model = smf.glm('Angambiae ~ Position + Temperature + Humidity + (1|Village)', data=test_df, family = sm.families.NegativeBinomial(alpha=1))
# test_model_results = test_model.fit()
# print(test_model_results.summary())

# # Fit Poisson MLM
# model = sm.GLM(test_df["Angambiae"], test_df[["Position", "Temperature", "Humidity"]], family=sm.families.Poisson())

# # Add random effects
# model = model.fit(formula = "Angambiae ~ Position + Temperature + Humidity + (1 | Village)", groups = test_df["Village"])

# # Print summary
# print(model.summary())

# %%

# Fit the model with negative binomial distribution
model = pmm.Lmer("Angambiae ~ Position + Temperature + Humidity + (1 | Village)", data = test_df, family='negbin').fit()

print(model.summary())


# %%

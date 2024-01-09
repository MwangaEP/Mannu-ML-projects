
#%%

import json

import numpy as np 
import pandas as pd

import statsmodels.api as sm
import statsmodels.formula.api as smf
import scipy.stats as stats
# import pymer4.models as pmm 
from statsmodels import graphics
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

# view dataset, first ten raws
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

# Since the resolution of the spectra data is 2cm, 2300 may not be available in the colummn names, 
# Check whats the closest number

# drop treatment column temporarly
temp_df = par_df_2.drop(['Age', 'Infection'], axis = 1)

# make a list containing all column names
col_names = temp_df.columns.tolist()

col_names = [int(x) for x in col_names]

# get the closest wavenumbers
start_col_1 = list(map(lambda y:min(col_names, key=lambda x:abs(x - y)), [3855]))
end_col_1 = list(map(lambda y:min(col_names, key=lambda x:abs(x - y)), [3650]))

start_col_2 = list(map(lambda y:min(col_names, key=lambda x:abs(x - y)), [2500]))
end_col_2 = list(map(lambda y:min(col_names, key=lambda x:abs(x - y)), [1700]))

print(start_col_1)
print(end_col_1)
print(start_col_2)
print(end_col_2)

# get the column names between the start and end column
cols_to_drop_1 = [str(i) for i in range(int(start_col_1[0]), int(end_col_1[0]) - 1, -2)]
cols_to_drop_2 = [str(i) for i in range(int(start_col_2[0]), int(end_col_2[0]) - 1, -2)]


par_df_2 = par_df_2.drop(cols_to_drop_1, axis=1)
par_df_2 = par_df_2.drop(cols_to_drop_2, axis=1)

par_df_2.head()

#%%

# count the number of levels in the column and their size
print(Counter(par_df_2["Age"]))

# loading important wavenumbers for only early rings vs late rings
important_wavenumb = pd.read_json(
                                    'C:\Mannu\Projects\Mwanga-DBS work\Parasite age\ML analysis\Early_Late\wavenumbers_coef_11.txt',
                                    orient = 'records', 
                                    lines = True 
                                )

# get the column names
important_wavenumb_2 = [str(item) for item in list(important_wavenumb.columns)]

# select the columns
par_selected_wn_df = par_df_2.loc[:, important_wavenumb_2]

# Append age and infection in the dataset
par_selected_wn_df['Age'] = Age
par_selected_wn_df['Infection'] = Infection

# Rename age columns to have only two categories for the red blood cells, either early or late
par_selected_wn_df['Age'].replace('Early control', 'Early', inplace = True)
par_selected_wn_df['Age'].replace('Early rings', 'Early', inplace = True)
par_selected_wn_df['Age'].replace('Late control', 'Late', inplace = True)
par_selected_wn_df['Age'].replace('Late rings', 'Late', inplace = True)

# par_selected_wn_df .to_csv("C:\Mannu\Projects\Mwanga-DBS work\Parasite age\Background_wn_removed\late control vs late rings\glm_earlyrings_df.csv", index=False)
par_selected_wn_df 

#%%

# Subset the data to only include the variables of interest
# glm_df = par_selected_wn_df[['2553', 'Age', 'Infection']]
glm_df = par_selected_wn_df 

# Convert 'age' and 'Infection' columns to category data type
glm_df['Age'] = glm_df['Age'].astype('category')
glm_df['Infection'] = glm_df['Infection'].astype('category')

###################

# Balance categories in the data
# Group the data by the two columns and get the size of each group
group_sizes = glm_df.groupby(['Infection', 'Age']).size()

# Find the minimum group size
min_size = group_sizes.min()

# Sample min_size number of rows from each group to create a new dataframe with balanced classes
balanced_data = glm_df.groupby(
                                ['Infection', 'Age'],
                                group_keys = False
                            ).apply(lambda x: x.sample(min_size))

print(balanced_data.groupby(['Infection', 'Age']).size())

#%%
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

full_model = sm.GLM.from_formula(
                                    "Q('3613') ~ Age + Infection + Age:Infection", 
                                    data = balanced_data, 
                                    family = sm.families.Gaussian()
                                )

full_model_results = full_model.fit()
print(full_model_results.summary())

# The llf attribute is generated for each model—this is the log likelihood statistic. 
# The likelihood ratio test then compares the log likelihood values and tests whether the 
# alternative model is significantly different to the null model. 

# Get the log-likelihood statistic for the full model
# llf_full_model = np.round(full_model_results.llf, 2) 

reduced_model = sm.GLM.from_formula(
                                        "Q('3613') ~ Age + Infection", 
                                        data = balanced_data, 
                                        family = sm.families.Gaussian()
                                    )
                                
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

reduced_model = sm.GLM.from_formula(
                                        "Q('3613') ~ Age + Infection", 
                                        data = balanced_data, 
                                        family = sm.families.Gaussian()
                                    )

reduced_model_results = reduced_model.fit()
print(reduced_model_results.summary())

#%%

reduced_model_age = sm.GLM.from_formula(
                                            "Q('3613') ~ Age", 
                                            data = balanced_data, 
                                            family = sm.families.Gaussian())
reduced_model_age_results = reduced_model_age.fit()
print(reduced_model_age_results.summary())

# %%

reduced_model_inf = sm.GLM.from_formula(
                                            "Q('3613') ~ Infection", 
                                            data = balanced_data, 
                                            family = sm.families.Gaussian())
reduced_model_inf_results = reduced_model_inf.fit()

# Q-Q Plot
graphics.gofplots.qqplot(reduced_model_inf_results.resid_deviance, line = 'r')

print(reduced_model_inf_results.summary())

# %%

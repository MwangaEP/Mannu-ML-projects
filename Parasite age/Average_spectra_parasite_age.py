#%% 
# Import modules
import numpy as np 
import pandas as pd
from collections import Counter 

import matplotlib.pyplot as plt # for making plots
import seaborn as sns

sns.set(context = "paper",
        style = "whitegrid",
        palette = "deep",
        font_scale = 2.0,
        color_codes = True,
        rc=None)
# %matplotlib inline
plt.rcParams["figure.figsize"] = [6,4]

#%%

# Loading dataset  
# Upload parasite age data 

par_age_df = pd.read_csv("C:\Mannu\Projects\Mwanga-DBS work\Parasite age\Transformed\Age_spectra_raw_clean.dat", delimiter = '\t')
print(par_age_df.head())

print(par_age_df.shape)

# Checking class distribution in the data
print(Counter(par_age_df["Cat1"]))

# drops columns of no interest
# train_data = par_age_df.drop(['Unnamed: 0'], axis = 1)
par_age_df.head(10)


#%%

# rename factors in age variable (Cat1) to make it more informative

Age = []

for row in par_age_df['Cat1']:

    if row == 'EC':
        Age.append('Control')
    
    elif row == 'LC':
        Age.append('Control')
    
    elif row == 'ET':
        Age.append('Early rings')

    else:
        Age.append('Late rings')

# print(Age)

par_age_df['Age'] = Age

# drop the column with age as a string and keep the age in intergers

par_age_df_2 = par_age_df.drop(['Cat1', 'Cat2', 'Cat3', 'Cat5'], axis = 1) 
par_age_df_2.head(5)

# Checking class distribution in the data
print(Counter(par_age_df_2["Age"]))

#%%

dbs_df = pd.read_csv("C:\Mannu\Projects\Mwanga-DBS work\Parasite age\Transformed\DBS_spectra.dat", delimiter = '\t')
print(dbs_df.head())

print(dbs_df.shape)

# Checking class distribution in the data
print(Counter(dbs_df["Cat1"]))

# drops columns of no interest
dbs_df = dbs_df.drop(['Cat1', 'Cat3'], axis = 1)
dbs_df.head(10)

#%%
# calculate the mean of each class to plot the average spectra 

control = par_age_df_2.loc[par_age_df_2['Age'] == 'Control']
control = pd.DataFrame(control.iloc[:,:-1].mean().T).reset_index()
# control = hum.reset_index()
control.rename(columns = {'index':'wavenumber', 0:'absorbance'}, inplace = True)
# control.to_csv("C:\Mannu\Projects\Anophles Funestus Age Grading (WILD)\young_age_average.csv", index = False)

early = par_age_df_2.loc[par_age_df_2['Age'] == 'Early rings']
early = pd.DataFrame(early.iloc[:,:-1].mean().T).reset_index()
early.rename(columns = {'index':'wavenumber', 0:'absorbance'}, inplace = True)
# early.to_csv("C:\Mannu\Projects\Anophles Funestus Age Grading (WILD)\old_age_average.csv", index = False)

late = par_age_df_2.loc[par_age_df_2['Age'] == 'Late rings']
late = pd.DataFrame(late.iloc[:,:-1].mean().T).reset_index()
late.rename(columns = {'index':'wavenumber', 0:'absorbance'}, inplace = True)
# late.to_csv("C:\Mannu\Projects\Anophles Funestus Age Grading (WILD)\old_age_average.csv", index = False)

dbs = pd.DataFrame(dbs_df.iloc[:,1:].mean().T).reset_index()
dbs.rename(columns = {'index':'wavenumber', 0:'absorbance'}, inplace = True)

#%%
sns.set(context = 'paper',
        style = 'white',
        palette = 'deep',
        font_scale = 2.0,
        color_codes = True,
        rc = ({'font.family': 'Dejavu Sans'}))
plt.figure(figsize = (8, 4))

plt.plot(pd.to_numeric(dbs['wavenumber']).sort_values(ascending = False), dbs['absorbance'],  color = "green", linewidth=0.7)
# plt.plot(pd.to_numeric(control['wavenumber']).sort_values(ascending = False), control['absorbance'], color = "black", linewidth=0.7)
plt.plot(pd.to_numeric(early['wavenumber']).sort_values(ascending = False), early['absorbance'],  color = "blue", linewidth=0.7)
plt.plot(pd.to_numeric(late['wavenumber']).sort_values(ascending = False), late['absorbance'],  color = "red", linewidth=0.7)
plt.legend(['Protein saver cards', 'Early rings', 'Late rings'])
plt.xlabel("Wavenumbers / cm-1", weight = 'bold')
plt.ylabel("Absorbance", weight = 'bold')
plt.xlim(4000, 500)
plt.savefig("C:\Mannu\Projects\Mwanga-DBS work\Parasite age\ML analysis\Averaged_sepctra_graph", dpi = 500, bbox_inches="tight")
# %%
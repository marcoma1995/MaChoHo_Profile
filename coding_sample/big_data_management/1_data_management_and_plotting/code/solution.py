# Import packages
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from replace_values import (neg100_to_missing,
                            neg_to_missing,
                            invert_bool)
from create_path_name import create_path_name




""" Step 3: Load the chs_data. Create a list of unique child ids
and discard observations not measured in 1986 to 2010.
"""

# 3.1 Load the chs_data.
chs_data_path= create_path_name('original_data', 'chs_data.dta')
chs_data = pd.read_stata(chs_data_path)

# 3.2 Create a list of unique child ids.
# Drop duplicates in childid.
childids_drop = chs_data.drop_duplicates('childid')

# Create childids as the list of child ids.
childids = np.array(childids_drop.childid).tolist()

# 3.3 Discard observations not measured in 1986 to 2010.
years = list(range(1986, 2011, 2))
chs_data_drop = chs_data[chs_data.year.isin(years)]


# Step 4: Clean and transform the bpi data set.

# 4.1 Load the bpi data set.
bpi_data_path= create_path_name('original_data', 'BEHAVIOR_PROBLEMS_INDEX.dta')
bpi_data = pd.read_stata(bpi_data_path)

# 4.2 Drop observations which don't belong to children in the chs_data.
bpi_clean = bpi_data[bpi_data.C0000100.isin(childids)]

# 4.3 Replace negative numbers by pd.np.nan.

# Replace all negative numbers in dataset by pd.np.nan.
bpi = bpi_clean.applymap(lambda bpi_clean: neg_to_missing(bpi_clean) \
                               if isinstance(bpi_clean, (int, float))
                               else bpi_clean
                        )

# 4.4 Create dictionary.
# Create path and use it to load bpi_var_info.csv.
bpi_var_info_path = create_path_name('original_data','bpi_variable_info.csv')
bpi_var_info = pd.read_csv(bpi_var_info_path)    
                                    
""" Set variable names that do not change over time (invariant) to 0. 
Then change the variable survey year to an integer.
""" 
bpi_var_info.loc[bpi_var_info['survey_year'] == 'invariant', ['survey_year']] = 0
bpi_var_info['survey_year'] = bpi_var_info['survey_year'].astype(int)

# Initialize the dictionary and create the keys for it.
bpi_vars = {}
bpi_dict = {}
keys = range(1986,2010,2)

""" Create a dictionary where for each key (year), we get all associated 
variables.
Add the invariant variables to all years.
"""
for k in keys:
    list_k = [
              'C0000100',
              'C0000200',
              'C0005800'
              ]
    for i in range(0, len(bpi_var_info['nlsy_name'])):
        if (bpi_var_info.iloc[i]['survey_year'] == k):
            list_k.append(bpi_var_info.iloc[i]["nlsy_name"])
            bpi_vars[k] = list_k

""" Create a dictionary where for each key (year) we have a dataframe of all 
the data (i.e. obs) associated with that year.
These dataframes already exclude any variables not present in bpi_var_info. 
Also rename columns in dataframe to readable_names and add year variable to 
each dataframe in the dictionary.
"""
for k in keys:
    bpi_dict[k] = bpi[bpi_vars[k]]
    bpi_dict[k] = bpi_dict[k].rename(columns=dict(zip(bpi_var_info['nlsy_name'],
                                            bpi_var_info['readable_name'])))
    year = pd.DataFrame({'year':[k]*11521}) 
    bpi_dict[k] = bpi_dict[k].join(year)
    
# Step 5: Transform the dataset from wide to long and save it as a csv.
    
# Transform dataset from wide to long.
bpi_long = pd.DataFrame()
for k in keys:
    bpi_long = bpi_long.append(bpi_dict[k].pivot(index = 'childid', 
                               columns = 'year'
                               ))
    
bpi_long = bpi_long.stack(1)

# Sort long dataset so each id's years of responses are put together.
bpi_long = bpi_long.sort_index() 

#Save data set in bld/bpi_long.csv. 
bpi_long_path= create_path_name('bld', 'bpi_long.csv')
bpi_long.to_csv(bpi_long_path)

# Step 6: Merge the long bpi dataset with the chs_dataset.

""" Only keep observations that are present in the chs_dataset.
Use the columns childid and year as merge keys and give all
variables that appear in both datasets the suffixes _chs
respectively  _bpi. 
"""
bpi_merged = chs_data.merge(bpi_long, left_on=['childid','year'],
                            suffixes=('_chs','_bpi'), right_index=True
                            )

# Save bpi_merged as a comma separated file in bld/bpi_merged.csv.
bpi_merged_path= create_path_name('bld', 'bpi_merged.csv')
bpi_merged.to_csv(bpi_merged_path)

""" Step 7: Calculate scores for each subscale of the bpi. Standardize
the scores to mean 0 and variance 1 for each age group.
Save the result as a csv.
"""
# Find all variables relevant for the subscales and store them in a list.
subscales = [
            'antisocial',
            'anxiety',
            'dependent',
            'headstrong',
            'hyperactive',
            'peer'
            ]
variable_list = bpi_long.columns.tolist()
relevant_variables = []
for i in range(len(subscales)):
    for j in range(len(variable_list)):
        if variable_list[j].startswith(subscales[i]):
            relevant_variables.append(variable_list[j])

""" Create the scores for the subscales.  
Create two variables for all measures of the subscales. One variable which 
takes the value 1 if the answer is some sort of Somtime True and Often True, 
and one variable which takes value 0 if the answer is not true.
The variables take the value -100 if the answer was something else or is
missing.
"""
for variable in relevant_variables:
    bpi_merged["{}_value_0".format(variable)]= np.where(
            bpi_merged["{}".format(variable)].str.contains(
                    "not",case=False, na=False),0,-100)
    bpi_merged["{}_value_1".format(variable)]= np.where(
            bpi_merged["{}".format(variable)].str.contains(
                    "often|some",case=False, na=False),1,-100)

# Set all values of -100 to np.nan. This also prepares the chs-data for step 8.
bpi_merged = bpi_merged.applymap(lambda bpi_merged: neg100_to_missing(bpi_merged) \
                               if isinstance(bpi_merged, (int, float)) 
                               else bpi_merged)  
""" Combine the two variables for each measure to one variable which is named
by the measure and the suffix_values
"""
for variable in relevant_variables:
    bpi_merged["{}_values".format(variable)]= bpi_merged[
                                                "{}_value_0".format(variable)]
    new_input = bpi_merged["{}_value_1".format(variable)]
    new_column = pd.Series(new_input, name="{}_values".format(variable))
    bpi_merged.update(new_column)

# Create the subscale scores.
# Create a list containing lists of all variables for each subscale.
subscale_variables=[]    
for i in range(len(subscales)):
    list_of_variables = []
    for j in range(len(variable_list)):
        if variable_list[j].startswith(subscales[i]):
            list_of_variables.append(variable_list[j])
    subscale_variables.append(list_of_variables)

# Create variables containing the scores named by the subscale and _score.    
for i, subscale in enumerate(subscales):
    score =bpi_merged["{}_values".format(subscale_variables[i][0])]
    for j in range(len(subscale_variables[i])-1):
        score= score + bpi_merged["{}_values".format(
                                            subscale_variables[i][j+1])]
        bpi_merged["{}_score".format(subscale)]= (score 
                                           / len(subscale_variables[i]))
    
# Slice the data set according to age groups below 6, 6 to 11 and above 11.
bpi_merged_below_6=bpi_merged.loc[bpi_merged["age"]<6]
bpi_merged_6_to_11=bpi_merged.loc[(bpi_merged["age"]>5) 
                                & (bpi_merged["age"]<12)]
bpi_merged_above_11=bpi_merged.loc[bpi_merged["age"]>11]

# Standardize subscales in the data set slices.
for subscale in subscales:
    bpi_merged_below_6["{}_standardized".format(subscale)]=(bpi_merged_below_6[
                                                "{}_score".format(subscale)] 
            - bpi_merged_below_6["{}_score".format(subscale)].mean())/(bpi_merged_below_6["{}_score".format(subscale)].std())
    bpi_merged_6_to_11["{}_standardized".format(subscale)]=(bpi_merged_6_to_11[
            "{}_score".format(subscale)] 
            - bpi_merged_6_to_11["{}_score".format(subscale)].mean())/(bpi_merged_6_to_11["{}_score".format(subscale)].std())
    bpi_merged_above_11["{}_standardized".format(subscale)]=(bpi_merged_above_11[
            "{}_score".format(subscale)] 
            - bpi_merged_above_11["{}_score".format(subscale)].mean())/(bpi_merged_above_11["{}_score".format(subscale)].std())

    # Add standardized slices as variables to the whole data frame. 
    bpi_merged["{}_standardized_below_6".format(subscale)]=bpi_merged_below_6[
                                        "{}_standardized".format(subscale)]
    bpi_merged["{}_standardized_6_to_11".format(subscale)]=bpi_merged_6_to_11[
                                        "{}_standardized".format(subscale)]
    bpi_merged["{}_standardized_above_11".format(subscale)]=bpi_merged_above_11[
                                        "{}_standardized".format(subscale)]
    
    # Combine age groups to one variable named subscale_standardized where 
    # subscale takes the name of the subscales.
    bpi_merged["{}_standardized".format(subscale)]=bpi_merged["{}_standardized_below_6".format(subscale)]
    new_input = bpi_merged["{}_standardized_6_to_11".format(subscale)]
    new_column = pd.Series(new_input, name="{}_standardized".format(subscale))
    bpi_merged.update(new_column)
    new_input = bpi_merged["{}_standardized_above_11".format(subscale)]
    new_column = pd.Series(new_input, name="{}_standardized".format(subscale))
    bpi_merged.update(new_column)

bpi_final = bpi_merged    
bpi_final_path = create_path_name('bld', 'bpi_final.csv')
bpi_final.to_csv(bpi_final_path)    

# Step 8: Make regression plots for each subscale that show how the score in 
# the bpi_data relates to the same score in the chs_data.

bpi_subscales = [
                 'antisocial_standardized', 
                 'anxiety_standardized', 
                 'headstrong_standardized', 
                 'hyperactive_standardized', 
                 'peer_standardized'
                 ]
chs_subscales = [
                 'bpiA',
                 'bpiB',
                 'bpiC',
                 'bpiD',
                 'bpiE'
                 ]
title = [
        'regplot_antisocial',
        'regplot_anxiety',
        'regplot_hyperactive',
        'regplot_peer',
        'regplot_headstrong'
        ]

for x, y, title in zip( bpi_subscales,chs_subscales, title):
    sns.regplot(x= x,
                y= y, data = bpi_final, x_jitter = 0.02
                ).set_title(title)
    save_path = create_path_name('bld', '{}.png'.format(title))
    plt.savefig(save_path)
    
    plt.show()

# Step 9: Create a heat-map of the correlation matrix of the items from the
# subscales.

# Create heatmap dataset as boolean values of observations from relevant vars 
# of bpi_long
bpi_heatmap = bpi_long[relevant_variables].astype(str)
for v in relevant_variables:
    bpi_heatmap[v] = bpi_long[v].str.contains("Not True", case = False)

# Since Booleans are coded in reverse (not true = true), invert the dataset
# to obtain proper coding.    
bpi_heatmap = bpi_heatmap.applymap(lambda bpi_heatmap: invert_bool(bpi_heatmap))

#Obtain correlations and create heatmap
bpi_heatmap.corr()
sns.heatmap(bpi_heatmap.corr(), vmin=-1, vmax=1, cmap="RdYlGn", center=0)
heatmap_path = create_path_name('bld','heatmap.png' )
plt.savefig(heatmap_path)
plt.show()




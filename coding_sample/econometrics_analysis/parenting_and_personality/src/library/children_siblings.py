import numpy as np

import pandas as pd

from bld.project_paths import project_paths_join as ppj

# Read the dataset.
youth_long = pd.read_stata(ppj("IN_DATA", "jugendl.dta"))

# Locate the ids variable.
ids = youth_long.iloc[:, :4]

# Create a list of variable we need.
variable = ['jl0275', 'jl0276',
            'jl1404', 'jl0446']

# Locate control variable we need from the list.
number_of_siblings_raw = youth_long.loc[:, variable]


# Rename  columns to readable name.
number_of_siblings_raw.columns = ["number_of_brothers",
                                  "number_of_sisters",
                                  'number_of_siblings_1000',
                                  "number_of_siblings_500"]

# Create a dataframe merge all the data.
data = pd.concat([ids, number_of_siblings_raw], axis=1)

# Replace all negative number into pd.np.nan.
dict_n = {-8: pd.np.nan, -1: pd.np.nan, -3: pd.np.nan, -2: 0}
data_dict = data.replace(dict_n)

# Do this sum because there is no value loss.
data_dict['number_of_siblings_sum'] = data_dict[
    "number_of_sisters"] + data_dict["number_of_brothers"]

# Create a frame with only sum, number of sisters and brothers
a = data_dict.loc[:, 'number_of_siblings_1000': 'number_of_siblings_sum']

# Creates the final measure by taking the max value of the thre columns.
data_dict['number_of_siblings_final'] = np.nanmax(a, axis=1)

number_of_siblings = pd.concat([ids,
                                data_dict.number_of_siblings_final], axis=1)

# Output the data as a whole.


def save_data(siblings):
    siblings.to_csv(ppj("OUT_DATA", "children_siblings.csv"),
                    sep=",")


if __name__ == "__main__":
    siblings = number_of_siblings
    save_data(siblings)

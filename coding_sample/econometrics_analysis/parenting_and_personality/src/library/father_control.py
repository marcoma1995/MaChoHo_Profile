import numpy as np

import pandas as pd

from bld.project_paths import project_paths_join as ppj

# Read the dataset.
pgen = pd.read_stata(ppj("IN_DATA", "pgen.dta"))
biol = pd.read_stata(ppj("IN_DATA", "biol.dta"))
pequiv = pd.read_stata(ppj("IN_DATA", "pequiv.dta"))
jungendl = pd.read_stata(ppj("IN_DATA", "jugendl.dta"))

# Extract years of education and nationality of parents
pgen_relevant = pgen.loc[:, ['cid', 'pid', 'syear', 'pgbilzeit']]
pgen_relevant.columns = ['cid', 'pid', 'syear', 'years_of_education']

# Extract gender of parents
pequiv_relevant = pequiv.loc[:, ['cid', 'pid', 'syear', 'd11102ll']]
pequiv_relevant.columns = ['cid', 'pid', 'syear', 'sex_of_parents']

# Extract migration control of parents,
biol_relevant = biol.loc[:, ['cid', 'pid',
                             'syear', 'lb0011', 'lb0013', 'lb0014']]
biol_relevant.columns = ['cid', 'pid', 'syear', 'year_of_birth',
                         'born_in_germany', 'german_nationality']

# Merge it together.
parents = pequiv_relevant.merge(pgen_relevant, how='left',
                                on=['cid', 'pid', 'syear'])
parents_controls = parents.merge(biol_relevant, how='left',
                                 on=['cid', 'pid', 'syear'])

# Keep observation only in jungendl.
childids = np.array(jungendl.cid).tolist()
parents_controls_drop = parents_controls[
    parents_controls.cid.isin(childids)]

# Replace all negative number into pd.np.nan.
dict_n = {'[-1] keine Angabe': pd.np.nan,
          '[-8] Frage in diesem Jahr nicht Teil des Frageprograms': pd.np.nan,
          -2: pd.np.nan, -8: pd.np.nan, -1: pd.np.nan,
          '[-5] In Fragebogenversion nicht enthalten': pd.np.nan,
          '[-2] trifft nicht zu': pd.np.nan}
parents_control = parents_controls_drop.replace(dict_n)

# Change the answer into number for creating measure.
dict_v = {'[1] Deutschland': 1, '[1] Male           1': 0,
          '[2] Female         2': 1, '[2] ausserhalb Deutschlands': 0,
          '[1] Ja': 1, '[2] Nein': 2}
parents_control_replace = parents_control.replace(dict_v)

# Create the DataFrame for paternal control.
father_controls = parents_control_replace.loc[:, ['pid', 'syear',
                                                  'sex_of_parents',
                                                  'years_of_education',
                                                  'year_of_birth',
                                                  'born_in_germany',
                                                  'german_nationality']]
father_controls.columns = ['pid_father', 'syear',
                           'sex_of_father', 'years_of_education_father',
                           'year_of_birth_father',
                           'father_born_in_germany',
                           'father_german_nationality']

# Create the DataFrame for maternal control.
mother_controls = parents_control_replace.loc[:, ['pid', 'syear',
                                                  'sex_of_parents',
                                                  'years_of_education',
                                                  'year_of_birth',
                                                  'born_in_germany',
                                                  'german_nationality']]
mother_controls.columns = ['pid_mother', 'syear',
                           'sex_of_mother', 'years_of_education_mother',
                           'year_of_birth_mother', 'mother_born_in_germany',
                           'mother_german_nationality']


def save_data(father_control):
    father_control.to_csv(ppj("OUT_DATA", "father_control.csv"),
                          sep=",")


if __name__ == "__main__":
    father_control = father_controls
    save_data(father_control)

import numpy as np

import pandas as pd

from bld.project_paths import project_paths_join as ppj

# Read the dataset.
trait = pd.read_csv(ppj("OUT_DATA", "measure_personality_children.csv"),
                    delimiter=",")
involvement = pd.read_csv(ppj("OUT_DATA", "measure_parental_involvment.csv"),
                          delimiter=",")
child_control = pd.read_csv(ppj("OUT_DATA", "children_control.csv"),
                            delimiter=",")
mother_control = pd.read_csv(ppj("OUT_DATA", "mother_control.csv"),
                             delimiter=",")
father_control = pd.read_csv(ppj("OUT_DATA", "father_control.csv"),
                             delimiter=",")
hh_control = pd.read_csv(ppj("OUT_DATA", "hh_control.csv"),
                         delimiter=",")
child_basic = pd.read_csv(ppj("OUT_DATA", "child_basic.csv"),
                          delimiter=",")
mother_trait = pd.read_csv(ppj("OUT_DATA", "measure_personality_mother.csv"),
                           delimiter=",")
father_trait = pd.read_csv(ppj("OUT_DATA", "measure_personality_father.csv"),
                           delimiter=",")
child_siblings = pd.read_csv(ppj("OUT_DATA", "children_siblings.csv"),
                             delimiter=",")

# Drop unnames:0 column.
df = [trait, involvement, child_control, hh_control, father_control,
      mother_control, child_basic, father_trait, mother_trait, child_siblings]

for a in df:
    a = a.drop(['Unnamed: 0'], axis=1)


# Merge all DataFrame into one.
data = child_basic.merge(trait, on=['cid', 'pid', 'syear'], how='left')
data = data.merge(involvement, on=['cid', 'pid', 'syear'], how='right')
data = data.merge(mother_trait, on=['pid_mother'], how='left')
data = data.merge(father_trait, on=['pid_father'], how='left')
data = data.merge(father_control, on=['pid_father', 'syear'], how='left')
data = data.merge(mother_control, on=['pid_mother', 'syear'], how='left')
data = data.merge(hh_control, on=['cid', 'syear', 'hid'], how='left')
data = data.merge(child_control, on=['cid', 'syear', 'pid', 'hid'], how='left')
data = data.merge(
    child_siblings, on=[
        'cid', 'syear', 'pid', 'hid'], how='left')
data = data.drop_duplicates()


# Create age difference variables.
data.birth_year_mother = data.birth_year_mother.replace({-5: pd.np.nan})
data.birth_year_father = data.birth_year_father.replace({-5: pd.np.nan})
data['age_difference_mother'] = data.birth_year - data.birth_year_mother
data['age_difference_father'] = data.birth_year - data.birth_year_father

# Create age difference variables.
data['age_difference_mother'] = data.birth_year - data.birth_year_mother
data['age_difference_father'] = data.birth_year - data.birth_year_father

# Create migration variable
data[['born_in_germany', 'german_nationality_father', 'german_nationality_mother']]
data['one'] = 1
data['zero'] = 0
data['migration_background'] = np.where(
    data.born_in_germany == data.zero, data.one, data.zero)
data['migration_background'] = np.where(
    data.german_nationality_father == data.zero,
    data.one, data.migration_background)
data['migration_background'] = np.where(
    data.german_nationality_mother == data.zero,
    data.one, data.migration_background)
data['migration_background_strict'] = data.apply(
    lambda row:
    row.german_nationality_mother + row.german_nationality_father,
    axis=1)

dict_mb = {2: 0, 1: 1, 0: 1}

data.migration_background_strict = data.migration_background_strict.replace(
    dict_mb)

# Create updates siblings variable.
data['number_of_siblings'] = data.number_of_siblings.fillna(
    data.number_of_siblings_final)

# Create log income.
data['log_monthly_hh_net_income'] = np.log(
    data["monthly_hh_net_income"].loc[data['monthly_hh_net_income'] != 0])

# Create per capita hh_income.
data['monthly_per_capita_net_income'] = data[
    "monthly_hh_net_income"] / data['number_of_persons_in_hh']
data['log_monthly_per_capita_net_income'] = np.log(
    data["monthly_per_capita_net_income"].loc[
        data['monthly_per_capita_net_income'] != 0])


data_male = data.loc[data['sex'] == 0]
data_female = data.loc[data['sex'] == 1]
data_final = data.drop_duplicates()

# Output the dataset.


def save_data(merge):
    merge.to_csv(ppj("OUT_DATA", "merge.csv"),
                 sep=",")


if __name__ == "__main__":
    merge = data_final
    save_data(merge)

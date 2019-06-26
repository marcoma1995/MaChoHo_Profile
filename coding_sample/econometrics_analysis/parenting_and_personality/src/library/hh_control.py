import pandas as pd

import numpy as np

from bld.project_paths import project_paths_join as ppj

# Read the dataset.
hbrutto = pd.read_stata(ppj("IN_DATA", "hbrutto.dta"))
hlong = pd.read_stata(ppj("IN_DATA", "hl.dta"))
kidlong = pd.read_stata(ppj("IN_DATA", "kidl.dta"))

# Select relevant data from hbrutto.
hbrutto_relevant = hbrutto.loc[:, ['cid', 'hid', 'syear', 'hhgr']]
hbrutto_relevant.columns = ['cid', 'hid', 'syear', 'number_of_persons_in_hh']

hlong_relevant = hlong.loc[:, ['cid', 'hid', 'syear', 'hlc0005', 'hlc0002',
                               'hlc0043']]
hlong_relevant.columns = ['cid', 'hid', 'syear', 'hh_monthly_net_income',
                          'hh_monthly_net_income_gen', 'number_of_children']

# Select relevant data from kidlong.
kidlong_relevant = kidlong.loc[:, ['cid', 'hid', 'syear', 'k_nrkid']]
kidlong_relevant.columns = ['cid', 'hid', 'syear', 'children_below16_in_hh']

# Merge the DataFrame together.
hh_controls = hbrutto_relevant.merge(hlong_relevant, how='left',
                                     on=['cid', 'hid', 'syear'])
hh_controls = hh_controls.merge(kidlong_relevant, how='left',
                                on=['cid', 'hid', 'syear'])

# Replace negative number into pd.np.nan.
dic_hh = {-1: pd.np.nan, -2: pd.np.nan, -3: pd.np.nan, -8: pd.np.nan}
hh_controls_clean = hh_controls.replace(dic_hh)

# Create number of siblings.
hh_controls_clean['number_children'] = np.where(
    hh_controls_clean.number_of_children
    >= hh_controls_clean.children_below16_in_hh,
    hh_controls_clean.number_of_children,
    hh_controls_clean.children_below16_in_hh)

hh_controls_clean['number_of_siblings'] = hh_controls_clean.number_children - 1

# Create income variable.
hh_controls_clean["monthly_hh_net_income"
                  ] = hh_controls_clean.hh_monthly_net_income.fillna(
    hh_controls_clean.hh_monthly_net_income_gen)


hh_controls = hh_controls_clean.drop(['number_of_children',
                                      'number_children'], axis=1)


# Output the data as a whole.
def save_data(hh_control):
    hh_control.to_csv(ppj("OUT_DATA", "hh_control.csv"),
                      sep=",")


if __name__ == "__main__":
    hh_control = hh_controls
    save_data(hh_control)

import pandas as pd

from bld.project_paths import project_paths_join as ppj

# Load dataset which containes tracker for parents.
bioage17_raw = pd.read_stata(ppj("IN_DATA", "bioage17.dta"))

# Extract relevant variables contained in bioage17 but not in jugendl +
# identifiers.
bioage17_relevant = bioage17_raw.loc[:, [
    'hhnr', 'persnr', 'syear', 'sex', 'bygebjah', 'bymnr', 'byvnr']]
bioage17_relevant.columns = [
    'cid',
    'pid',
    'syear',
    'sex',
    'birth_year',
    'pid_mother',
    'pid_father']
bioage17_relevant

# Replace number.
dic_replace = {-1: pd.np.nan, '[1] maennlich': 0, '[2] weiblich': 1}
bioage17_relevant_nan = bioage17_relevant.replace(dic_replace)

# Output the DataFrame.


def save_data(child_basic):
    child_basic.to_csv(ppj("OUT_DATA", "child_basic.csv"),
                       sep=",")


if __name__ == "__main__":
    child_basic = bioage17_relevant_nan
    save_data(child_basic)

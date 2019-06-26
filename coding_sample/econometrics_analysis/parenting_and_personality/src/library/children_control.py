import pandas as pd

from bld.project_paths import project_paths_join as ppj

# Read the dataset.
youth_long_raw = pd.read_stata(ppj("IN_DATA", "jugendl.dta"))

# Select the survey year between 2000 and 2016.
youth_long = youth_long_raw.loc[(youth_long_raw['syear'] != 2000) &
                                (youth_long_raw['syear'] != 2016)]

# Create DataFrames of id data.
ids = youth_long.iloc[:, :4]

# Create a list of variable we need for controlling.
variable = ['jl0235', "jl0244", "jl0242", 'jl0288',
            'jl0126']

# Locate the control variable we need from the list.
control_variable = youth_long.loc[:, variable]

# Rename  columns to readable name.
control_variable.columns = ["born_in_germany",
                            "german_nationality_since_birth",
                            "second_nationality",
                            "father_mother_live_in_hh",
                            "year_of_leaving_school"]

# Create a dataframe merge all the data.
data = pd.concat([ids, control_variable], axis=1)

# Replace all negative number into pd.np.nan.
dict_n = {'[-1] keine Angabe': pd.np.nan,
          '[-8] Frage in diesem Jahr nicht Teil des Frageprograms': pd.np.nan,
          '[-3] nicht valide': pd.np.nan,
          '[-4] unzulaessige Mehrfachantwort': pd.np.nan,
          '[-5] nicht im Fragebogen enthalten': pd.np.nan,
          -8: pd.np.nan, -2: pd.np.nan,
          '[-2] trifft nicht zu': pd.np.nan,
          '[-5] In Fragebogenversion nicht enthalten': pd.np.nan,
          -1: pd.np.nan, -3: pd.np.nan}

data_nan = data.replace(dict_n)

# Seperate father_mother_live_in_hh into two columns.
data_nan["father_lives_in_hh"] = data.father_mother_live_in_hh.replace(
    {'[3] Nur Mutter': 0})

data_nan["mother_lives_in_hh"] = data.father_mother_live_in_hh.replace(
    {'[2] Nur Vater': 0})

# Change the answer into number for creating measure.
dict_number = {'[1] Ja': 1, '[1] Ja, beide': 1, '[2] Nein': 0,
               '[1] Schulnote 1 (1-6)': 1, '[2] Schulnote 2 (1-6)': 2,
               '[3] Schulnote 3 (1-6)': 3, '[4] Schulnote 4 (1-6)': 4,
               '[5] Schulnote 5 (1-6)': 5, '[6] Schulnote 6 (1-6)': 6,
               '[1] Seit Geburt': 1, '[2] Spaeter Erworben': 0,
               '[3] Ja, in Westdeutschland': 1, '[4] Ja, in Ostdeutschland': 1,
               '[3] Nur Mutter': 1, '[2] Nur Vater': 1,
               '[4] Nein, beide nicht': 0, '[3] Nein': 0,
               '[1] Ja, vielleicht': 1, '[2] Ja, sicher': 1}

df = data_nan.replace(dict_number)

# Output the data as a whole.


def save_data(child_control):
    child_control.to_csv(ppj("OUT_DATA", "children_control.csv"),
                         sep=",")


if __name__ == "__main__":
    child_control = df
    save_data(child_control)

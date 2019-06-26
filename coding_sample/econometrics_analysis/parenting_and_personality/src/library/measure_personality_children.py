import pandas as pd

from sklearn.preprocessing import StandardScaler

from sklearn.decomposition import PCA as sklearnPCA

from bld.project_paths import project_paths_join as ppj

# Read the dataset.
youth_long_raw = pd.read_stata(ppj("IN_DATA", "jugendl.dta"))

# Select the survey year between 2000 and 2016.
youth_long = youth_long_raw.loc[(youth_long_raw['syear'] != 2000) &
                                (youth_long_raw['syear'] != 2016)]

# Create DataFrames of id data.
ids = youth_long.iloc[:, :4]

# Locate big 5 variable.
big_five = youth_long.loc[:, 'jl0365':'jl0379']

# Rename big 5 columns to readable name.
big_five.columns = ['work_carefully', 'communicative', 'abrasive', 'new_idea',
                    'often_worry', 'forgiving_nature', 'lazy', 'outgoing',
                    'esthetics', 'often_nervous', 'work_efficiently',
                    'reserved', 'considerate', 'lively_imagination',
                    'be_relaxed']

# Locate Locus of Control variable.
loc = youth_long.loc[:, 'jl0350':'jl0359']

# Rename big 5 columns to readable name.
loc.columns = ['life_depends_on_self', 'not_achieve_derserved',
               'achieve_luck', 'others_determine', 'workhard_to_sucess',
               'doubt_ability', 'background_determine',
               'born_determine', 'little_control', 'change_through_activites']

# Create a dataframe merge all the data.
data = pd.concat([ids, big_five, loc], axis=1)

# Replace all negative number into pd.np.nan.
dict_n = {'[-1] keine Angabe': pd.np.nan,
          '[-8] Frage in diesem Jahr nicht Teil des Frageprograms': pd.np.nan}
data_nan = data.replace(dict_n)

# Replace all string variable we use into number.
dict_f = {'[7] Trifft voll zu': 7, '[1] Trifft ueberhaupt nicht zu': 1,
          '[7] 7 stimme voll zu, (Skala 1-7)': 7,
          '[6] 6 auf Skala 1-7': 6, '[5] 5 auf Skala 1-7': 5,
          '[4] 4 auf Skala 1-7': 4, '[3] 3 auf Skala 1-7': 3,
          '[2] 2 auf Skala 1-7': 2,
          '[1] 1 stimme ueberhaupt nicht zu, (Skala 1-7': 1,
          '[1] 1 Trifft ueberhaupt nicht zu, (Skala 1-7)': 1,
          '[7] 7 Trifft voll zu, (Skala 1-7)': 7}

data_replace = data_nan.replace(dict_f)

# Reserves the scale for 'Negative' items.
# Create a dict of number I want to replace.
dict_r = {1: 7, 7: 1, 2: 6, 6: 2, 3: 5, 5: 3}

# Create list of 'Negavie' items and dictionary for things I want to replace.
negative = [
    'lazy',
    'abrasive',
    'reserved',
    'be_relaxed',
    'not_achieve_derserved',
    'achieve_luck',
    'others_determine',
    'doubt_ability',
    'background_determine',
    'born_determine',
    'little_control']

# Create new DataFrame and update the data.
reverse = data_replace.loc[:, negative].replace(dict_r)
data_replace.update(reverse)

# Replace missing value with pd.np.nan.
data_clean = data_replace.replace(r'\s+', pd.np.nan, regex=True)

# Create Locus of Control DataFrame for principal component analysis.
measure_loc = data_replace.loc[:, ['pid', 'life_depends_on_self',
                                   'not_achieve_derserved',
                                   'achieve_luck', 'others_determine',
                                   'workhard_to_sucess', 'doubt_ability',
                                   'background_determine', 'born_determine',
                                   'little_control',
                                   'change_through_activites']]

# Drop the nan data.
measure_loc_dropna = measure_loc.dropna()
measure_loc_droppid = measure_loc_dropna.drop(['pid'], axis=1)

# Create Locus of Control measure by principal component analysis.
measures_loc_clean_std = StandardScaler(
).fit_transform(measure_loc_droppid.dropna())
sklearn_pca = sklearnPCA(n_components=1)
locus_of_control_youth = sklearn_pca.fit_transform(
    measures_loc_clean_std) * (-1)

# Print the factor loadings.
print("Factor loadings of LoC of Children from PCA are {}".format(
    sklearn_pca.components_ * (-1)))

# Standardise the LoC measure.
locus_of_control_youth_std = StandardScaler().fit_transform(
    locus_of_control_youth)

# Save LoC scores in the exsisting DataFrame.
data_loc_clean = measure_loc.dropna()
data_loc_clean['loc_score'] = locus_of_control_youth
data_loc_clean['loc_score_std'] = locus_of_control_youth_std

# Create DataFrame with only the LoC PCA score.
measure_loc = pd.concat([measure_loc_dropna.pid,
                         data_loc_clean.loc_score,
                         data_loc_clean.loc_score_std], axis=1)

# Create list of variable corresond to Big Five Personality.
openness_ls = ['lively_imagination', 'new_idea', 'esthetics']
conscientiousness_ls = ['lazy', 'work_efficiently', 'work_carefully']
extraversion_ls = ['reserved', 'work_efficiently', 'work_carefully']
agreeableness_ls = ['forgiving_nature', 'considerate', 'abrasive']
neuroticism_ls = ['often_worry', 'often_nervous', 'be_relaxed']

# Create list of coloumns we want to create.
trait_ls = [neuroticism_ls, agreeableness_ls, extraversion_ls,
            conscientiousness_ls, openness_ls]
trait = ['neuroticism', 'agreeableness', 'extraversion',
         'conscientiousness', 'openness']

# Define a loop for creating the measure.
for x, y in zip(trait, trait_ls):
    data_clean[x] = data_clean[y].mean(1)

# Replace missing value in measure with pd.np.nan.
trait_replace = data_clean.replace(r'\s+', pd.np.nan, regex=True)

# Standardise all the measure and create as dataframe.


def standardise(x): return (x - x.mean()) / x.std()


trait = data_clean[trait].pipe(standardise)

# Merge the measure with id.
measure_big_five = pd.concat([ids, trait], axis=1).dropna()
df = pd.merge(measure_big_five, measure_loc, on='pid', how='inner')

# Output the data as a whole.


def save_data(personality):
    personality.to_csv(ppj("OUT_DATA", "measure_personality_children.csv"),
                       sep=",")


if __name__ == "__main__":
    personality = df
    save_data(personality)

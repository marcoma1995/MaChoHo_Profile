import pandas as pd

from bld.project_paths import project_paths_join as ppj

# Read the dataset.
adults2005 = pd.read_stata(ppj("IN_DATA", "vp.dta"))
adults2009 = pd.read_stata(ppj("IN_DATA", "zp.dta"))
adults2013 = pd.read_stata(ppj("IN_DATA", "bdp.dta"))


# Extract Column of Big 5 Variables we need for the research.
big_adults_2005 = adults2005.loc[:, 'vp12501':'vp12515']
big_adults_2009 = adults2009.loc[:, 'zp12001':'zp12015']
big_adults_2013 = adults2013.loc[:, 'bdp15101':'bdp15115']


# Rename to meaningful names. (Big Five)
for big_five in [big_adults_2005, big_adults_2009, big_adults_2013]:

    big_five.columns = ['work_carefully', 'communicative', 'abrasive',
                        'new_idea', 'often_worry',
                        'forgiving_nature', 'lazy', 'outgoing',
                        'esthetics', 'often_nervous', 'work_efficiently',
                        'reserved', 'considerate', 'lively_imagination',
                        'be_relaxed']


# Extract Column of basic variables we need for the research.
ids2005 = adults2005.loc[:, ['hhnr', 'persnr', 'welle', 'vp14701',
                             'vp14702', 'vp135']]
ids2009 = adults2009.loc[:, ['hhnr', 'persnr', 'welle', 'zp12901',
                             'zp12902', 'zp137']]

ids2013 = adults2013.loc[:, ['hhnr', 'persnr', 'welle', 'bdp13401',
                             'bdp13403', 'bdp143']]

# Rename identifiers to match the other data sets.
ids2005.columns = ['cid', 'pid_parents', 'syear', 'sex_parent_2005',
                   'birth_year_parent_2005', 'german_nationality_2005']
ids2009.columns = ['cid', 'pid_parents', 'syear', 'sex_parent_2009',
                   'birth_year_parent_2009', 'german_nationality_2009']
ids2013.columns = ['cid', 'pid_parents', 'syear', 'sex_parent_2013',
                   'birth_year_parent_2013', 'german_nationality_2013']

# Merge the ids with big five or LoC variable.
data_adults_2005 = pd.concat([ids2005, big_adults_2005],
                             axis=1)
data_adults_2009 = pd.concat([ids2009, big_adults_2009], axis=1)
data_adults_2013 = pd.concat([ids2013, big_adults_2013], axis=1)

# Create a dataframe merge all the data.
data_adults_whole = pd.concat([data_adults_2005, data_adults_2009,
                               data_adults_2013,
                               ], sort=False)
data_adults = data_adults_whole.reset_index(drop=True)

# Replace all negative number into pd.np.nan.
dict_n = {'[-1] keine Angabe': pd.np.nan}
data_adults_nan = data_adults.replace(dict_n)

# Replace all string variable we use into number.
dict_adults_f = {
    '[7] Trifft voll zu': 7,
    '[1] Trifft ueberhaupt nicht zu': 1,
    '[7] 7 stimme voll zu, (Skala 1-7)': 7,
    '[6] 6 auf Skala 1-7': 6,
    '[5] 5 auf Skala 1-7': 5,
    '[4] 4 auf Skala 1-7': 4,
    '[3] 3 auf Skala 1-7': 3,
    '[2] 2 auf Skala 1-7': 2,
    '[1] 1 stimme ueberhaupt nicht zu, (Skala 1-7': 1,
    '[1] Ja': 1,
    '[2] Nein': 0,
    '[-5] In Fragebogenversion nicht enthalten': pd.np.nan,
    '[7] 7 Stimme voll zu, (Skala 1-7)': 7,
    '[1] 1 Stimme ueberhaupt nicht zu, (Skala 1-7)': 1}

data_adults_replace = data_adults_nan.replace(dict_adults_f)

# Reserves the scale for 'Negative' items.
# Create list of 'Negavie' items and dictionary for things I want to replace.
# Create a dict of number I want to replace.
# Replace the number by creating new DataFrame and update.

dict_adults_r = {1: 7, 7: 1, 2: 6, 6: 2, 3: 5, 5: 3}
negative = ['little_control', 'lazy', 'abrasive', 'reserved', 'be_relaxed']

reverse = data_adults_replace.loc[:, negative].replace(dict_adults_r)

data_adults_replace.update(reverse)

data_adults_clean = data_adults_replace.replace(r'\s+', pd.np.nan, regex=True)


# Create list of variable corresond to Big Five.
openness_ls = ['lively_imagination', 'new_idea', 'esthetics']
conscientiousness_ls = ['lazy', 'work_efficiently', 'work_carefully']
extraversion_ls = ['reserved', 'work_efficiently', 'work_carefully']
agreeableness_ls = ['forgiving_nature', 'considerate', 'abrasive']
neuroticism_ls = ['often_worry', 'often_nervous', 'be_relaxed']

# Create list of coloumns we want to create.
trait_ls = [neuroticism_ls, agreeableness_ls,
            extraversion_ls, conscientiousness_ls, openness_ls]
trait = ['neuroticism', 'agreeableness',
         'extraversion', 'conscientiousness', 'openness']

# Define a loop for creating the measure.
for x, y in zip(trait, trait_ls):
    data_adults_clean[x] = data_adults_clean[y].mean(1)

# Create function for standardising the data.


def standardise(x): return (x - x.mean()) / x.std()


# Standardise all the measure and create as dataframe.
trait = data_adults_clean[trait].pipe(standardise)

# Merge the measure with id.
measures_adults = pd.concat([data_adults_replace, trait], axis=1)
measures_adults = measures_adults.rename(columns={'pid_parents': 'pid'})

# Create dataset which can be easily merged on the youth data.
# Create data frame which contains all pid available in the data set, once.
pid_parents = pd.DataFrame(measures_adults.pid.unique(), columns=[
    'pid_parents'])


# Create datasets for each year with year specific names.
# Create 2005.
adults2005_fit = measures_adults.loc[:,
                                     ['pid',
                                      'cid',
                                      'syear',
                                      'sex_parent_2005',
                                      'birth_year_parent_2005',
                                      'german_nationality_2005',
                                      'neuroticism',
                                      'agreeableness',
                                      'extraversion',
                                      'conscientiousness',
                                      'openness']]

adults2005_fit_clean = adults2005_fit.loc[adults2005_fit['syear'] == 2005]

adults2005_fit_clean.columns = ['pid_parents', 'cid_parents_2005',
                                'syear', 'sex_parents_2005',
                                'birth_year_parent_2005',
                                'german_nationality_2005',
                                'neuroticism_parents_2005',
                                'agreeableness_parents_2005',
                                'extraversion_parents_2005',
                                'conscientiousness_parents_2005',
                                'openness_parents_2005']

adults2005_final = adults2005_fit_clean.drop('syear', axis=1)

# Create 2009.
adults2009_fit = measures_adults.loc[:,
                                     ['pid',
                                      'cid',
                                      'syear',
                                      'sex_parent_2009',
                                      'birth_year_parent_2009',
                                      'german_nationality_2009',
                                      'neuroticism',
                                      'agreeableness',
                                      'extraversion',
                                      'conscientiousness',
                                      'openness']]

adults2009_fit_clean = adults2009_fit.loc[adults2009_fit['syear'] == 2009]

adults2009_fit_clean.columns = ['pid_parents', 'cid_parents_2009',
                                'syear', 'sex_parents_2009',
                                'birth_year_parent_2009',
                                'german_nationality_2009',
                                'neuroticism_parents_2009',
                                'agreeableness_parents_2009',
                                'extraversion_parents_2009',
                                'conscientiousness_parents_2009',
                                'openness_parents_2009']

adults2009_final = adults2009_fit_clean.drop('syear', axis=1)


# Create 2013.
adults2013_fit = measures_adults.loc[:,
                                     ['pid',
                                      'cid',
                                      'syear',
                                      'sex_parent_2013',
                                      'birth_year_parent_2013',
                                      'german_nationality_2013',
                                      'neuroticism',
                                      'agreeableness',
                                      'extraversion',
                                      'conscientiousness',
                                      'openness']]

adults2013_fit_clean = adults2013_fit.loc[adults2013_fit['syear'] == 2013]

adults2013_fit_clean.columns = ['pid_parents', 'cid_parents_2013',
                                'syear', 'sex_parents_2013',
                                'birth_year_parent_2013',
                                'german_nationality_2013',
                                'neuroticism_parents_2013',
                                'agreeableness_parents_2013',
                                'extraversion_parents_2013',
                                'conscientiousness_parents_2013',
                                'openness_parents_2013']

adults2013_final = adults2013_fit_clean.drop('syear', axis=1)


# Merge the data in a large dataframe.
parents_merge = pid_parents.merge(adults2005_final,
                                  on='pid_parents', how='left')
parents_merge = parents_merge.merge(adults2009_final,
                                    on='pid_parents', how='left')

parents_merge = parents_merge.merge(adults2013_final,
                                    on='pid_parents', how='left')


# Create the final score by taking the first available data.
a = ['openness_parents', 'conscientiousness_parents', 'extraversion_parents',
     'agreeableness_parents', 'neuroticism_parents']

b = [parents_merge.openness_parents_2005,
     parents_merge.conscientiousness_parents_2005,
     parents_merge.extraversion_parents_2005,
     parents_merge.agreeableness_parents_2005,
     parents_merge.neuroticism_parents_2005]

c = [parents_merge.openness_parents_2009,
     parents_merge.conscientiousness_parents_2009,
     parents_merge.extraversion_parents_2009,
     parents_merge.agreeableness_parents_2009,
     parents_merge.neuroticism_parents_2009]

d = [parents_merge.openness_parents_2013,
     parents_merge.conscientiousness_parents_2013,
     parents_merge.extraversion_parents_2013,
     parents_merge.agreeableness_parents_2013,
     parents_merge.neuroticism_parents_2013]

for x, y, z, s in zip(a, b, c, d):
    parents_merge[x] = y.fillna(z)

    parents_merge[x] = parents_merge[x].fillna(s)

# Create sex of parents.
parents_merge['mother_sex'] = 1
parents_merge['father_sex'] = 0

# Creat birth-year.
parents_merge['birth_year_parents'] = parents_merge[
    'birth_year_parent_2005'].fillna(parents_merge.birth_year_parent_2009)

parents_merge['birth_year_parents'] = parents_merge[
    'birth_year_parents'].fillna(parents_merge.birth_year_parent_2013)

# Creat german nationality.
parents_merge['german_nationality_parents'] = parents_merge[
    'german_nationality_2005'].fillna(
    parents_merge.german_nationality_2009)

parents_merge['german_nationality_parents'] = parents_merge[
    'german_nationality_parents'].fillna(
    parents_merge.german_nationality_2013)


# Output the data as a whole.


def save_data(personality):
    personality.to_csv(ppj("OUT_DATA", "measure_personality_adult.csv"),
                       sep=",")


if __name__ == "__main__":
    personality = parents_merge
    save_data(personality)

import pandas as pd

from sklearn.preprocessing import StandardScaler

from sklearn.decomposition import PCA as sklearnPCA

from bld.project_paths import project_paths_join as ppj

# Read the dataset.
youth_long_raw = pd.read_stata(ppj("IN_DATA", "jugendl.dta"))
youth2001 = pd.read_stata(ppj("IN_DATA", "rjugend.dta"))
youth2002 = pd.read_stata(ppj("IN_DATA", "sjugend.dta"))
youth2003 = pd.read_stata(ppj("IN_DATA", "tjugend.dta"))
youth2004 = pd.read_stata(ppj("IN_DATA", "ujugend.dta"))
youth2005 = pd.read_stata(ppj("IN_DATA", "vjugend.dta"))
youth2006 = pd.read_stata(ppj("IN_DATA", "wjugend.dta"))
youth2007 = pd.read_stata(ppj("IN_DATA", "xjugend.dta"))
youth2008 = pd.read_stata(ppj("IN_DATA", "yjugend.dta"))
youth2009 = pd.read_stata(ppj("IN_DATA", "zjugend.dta"))
youth2010 = pd.read_stata(ppj("IN_DATA", "bajugend.dta"))
youth2011 = pd.read_stata(ppj("IN_DATA", "bbjugend.dta"))
youth2012 = pd.read_stata(ppj("IN_DATA", "bcjugend.dta"))
youth2013 = pd.read_stata(ppj("IN_DATA", "bdjugend.dta"))
youth2014 = pd.read_stata(ppj("IN_DATA", "bejugend.dta"))
youth2015 = pd.read_stata(ppj("IN_DATA", "bfjugend.dta"))
youth2016 = pd.read_stata(ppj("IN_DATA", "bgjugend.dta"))


# Select the survey year between 2000 and 2016.
youth_long = youth_long_raw.loc[(youth_long_raw['syear'] != 2000) &
                                (youth_long_raw['syear'] != 2016)]

# Create DataFrames of id data.
ids = youth_long.iloc[:, :4]

# Extract variables concerning both parents.
involvement_parents = youth_long.loc[:, "jl0168":"jl0175"]
involvement_parents = involvement_parents.drop(["jl0170", "jl0174"], axis=1)

# Rename columns to readable names.
involvement_parents.columns = ['par_interested_in_performance',
                               'par_help_studying', 'parents_evening',
                               'teachers_officehours', 'outside_officehours',
                               'no_school_activity']

# Transform "Parents help with studying" into separated variables for parents.
m_helps_studying = involvement_parents.par_help_studying
f_helps_studying = involvement_parents.par_help_studying

# Create renaming dictionairy for mother and father.
dict_m = {'[1] Ja, Vater und Mutter': 1, '[2] Ja, nur die Mutter': 1,
          '[3] Ja, nur der Vater': 0, '[4] Nein': 0,
          '[-1] keine Angabe': pd.np.nan}
dict_f = {'[1] Ja, Vater und Mutter': 1, '[2] Ja, nur die Mutter': 0,
          '[3] Ja, nur der Vater': 1, '[4] Nein': 0,
          '[-1] keine Angabe': pd.np.nan}

m_helps_studying = m_helps_studying.replace(dict_m)
f_helps_studying = f_helps_studying.replace(dict_f)

# Extract variables concerning mothers involvement.
involvement_m = youth_long.loc[:, ['jl0040', 'jl0044', 'jl0046',
                                   'jl0048', 'jl0052', 'jl0054']]

involvement_m.columns = [
    'm_talks_about_things',
    'm_asks_decisions',
    'm_expresses_opinion',
    'm_solves_problems',
    'm_asks_opinion',
    'm_gives_reason']

# Extract variables concerning fathers involvement.
involvement_f = youth_long.loc[:, ['jl0041', 'jl0045', 'jl0047',
                                   'jl0049', 'jl0053', 'jl0055']]
involvement_f.columns = ['f_talks_about_things', 'f_asks_decisions',
                         'f_expresses_opinion', 'f_solves_problems',
                         'f_asks_opinion', 'f_gives_reason']

# Create one involvement data set.
involvement_data = pd.concat([ids, involvement_parents,
                              involvement_m, involvement_f,
                              f_helps_studying, m_helps_studying], axis=1)

involvement_data.columns = ['cid', 'syear', 'hid', 'pid',
                            'par_interested_in_performance',
                            'par_help_studying', 'parents_evening',
                            'teachers_officehours', 'outside_officehours',
                            'no_school_activity', 'm_talks_about_things',
                            'm_asks_decisions', 'm_expresses_opinion',
                            'm_solves_problems', 'm_asks_opinion',
                            'm_gives_reason', 'f_talks_about_things',
                            'f_asks_decisions', 'f_expresses_opinion',
                            'f_solves_problems', 'f_asks_opinion',
                            'f_gives_reason', 'f_help_studying',
                            'm_help_studying']

# There is an error in the variabels Mother talks about things that worry you
# jl0043 and Father talks about things
# that worry you, as the second doesn't exist and the first doesn't
# follow the naming pattern.
# We therefore have to create that variable from the core data and
# add to the long data set. I did so in the seperated

# Extract necessary data.
worries2001 = youth2001.loc[:, [
    'hhnr', 'persnr', 'syear', 'rj1402m', 'rj1402v']]
worries2002 = youth2002.loc[:, [
    'hhnr', 'persnr', 'syear', 'sj1402m', 'sj1402v']]
worries2003 = youth2003.loc[:, [
    'hhnr', 'persnr', 'syear', 'tj1402m', 'tj1402v']]
worries2004 = youth2004.loc[:, [
    'hhnr', 'persnr', 'syear', 'uj1402m', 'uj1402v']]
worries2005 = youth2005.loc[:, [
    'hhnr', 'persnr', 'syear', 'vj1402m', 'vj1402v']]
worries2006 = youth2006.loc[:, ['hhnr', 'persnr', 'syear', 'wj1403', 'wj1404']]
worries2007 = youth2007.loc[:, ['hhnr', 'persnr', 'syear', 'xj1403', 'xj1404']]
worries2008 = youth2008.loc[:, ['hhnr', 'persnr', 'syear', 'yj1403', 'yj1404']]
worries2009 = youth2009.loc[:, ['hhnr', 'persnr', 'syear', 'zj1403', 'zj1404']]
worries2010 = youth2010.loc[:, [
    'hhnr', 'persnr', 'syear', 'baj1403', 'baj1404']]
worries2011 = youth2011.loc[:, [
    'hhnr', 'persnr', 'syear', 'bbj1403', 'bbj1404']]
worries2012 = youth2012.loc[:, [
    'hhnr', 'persnr', 'syear', 'bcj1403', 'bcj1404']]
worries2013 = youth2013.loc[:, [
    'hhnr', 'persnr', 'syear', 'bdj1403', 'bdj1404']]
worries2014 = youth2014.loc[:, [
    'hhnr', 'persnr', 'syear', 'bej1403', 'bej1404']]
worries2015 = youth2015.loc[:, [
    'hhnr', 'persnr', 'syear', 'bfj1403', 'bfj1404']]

worries = [worries2001, worries2002, worries2003, worries2004, worries2005,
           worries2006, worries2007, worries2008, worries2009, worries2010,
           worries2011, worries2012, worries2013, worries2014, worries2015]

# Rename columns to match long format.
for worry in worries:
    worry.columns = ('cid', 'pid', 'syear', 'm_talks_about_worries',
                     'f_talks_about_worries')

# Merge the core dataframe into long.
worries_whole = pd.concat(worries, join="inner")
worry = worries_whole.reset_index(drop=True)
involvement_complete = involvement_data.merge(
    worry, how='left', on=['cid', 'pid', 'syear'])


involvement_clean = involvement_complete.drop('par_help_studying', axis=1)

# Invert scales so higher numbers imply higher frequency.
dic_involve = {'[3] eher wenig': 2, '[2] ziemlich stark': 3,
               '[1] sehr stark': 4, '[4] ueberhaupt nicht': 1,
               '[-1] keine Angabe': pd.np.nan, '[1] Ja': 1,
               '[-2] trifft nicht zu': 0, '[2] Haeufig': 3,
               '[3] Manchmal': 2, '[1] Sehr haeufig': 4,
               '[4] Selten': 1, '[5] Nie': 0, '[-2] trifft nicht zu': 0}
parental_involvement_data = involvement_clean.replace(dic_involve)

# Create "Parents involved in at least one school activity"
# by reverting the no school activity variable.
at_least_one_activity = parental_involvement_data.no_school_activity
dic = {0: 1, 1: 0}
at_least_one_activity.replace(dic)
parental_involvement_data[
    'at_least_one_activity'] = at_least_one_activity.replace(dic)
parental_involvement_data = parental_involvement_data.drop(
    'no_school_activity', axis=1)

# Standardize measures related to mothers involvment.
measure_matrix = parental_involvement_data.drop(
    ['cid', 'syear', 'hid', 'pid'], axis=1)
measures_clean = measure_matrix.dropna()
measures_mother = measures_clean.loc[:,
                                     ['par_interested_in_performance',
                                      'parents_evening',
                                      'teachers_officehours',
                                      'outside_officehours',
                                      'm_talks_about_things',
                                      'm_asks_decisions',
                                      'm_expresses_opinion',
                                      'm_solves_problems',
                                      'm_asks_opinion',
                                      'm_gives_reason',
                                      'm_talks_about_worries',
                                      'at_least_one_activity',
                                      'm_help_studying']]
measures_mother_std = StandardScaler().fit_transform(measures_mother)

# Standardize measures related to fathers involvment.
measures_father = measures_clean.loc[:,
                                     ['par_interested_in_performance',
                                      'parents_evening',
                                      'teachers_officehours',
                                      'outside_officehours',
                                      'f_talks_about_things',
                                      'f_asks_decisions',
                                      'f_expresses_opinion',
                                      'f_solves_problems',
                                      'f_asks_opinion',
                                      'f_gives_reason',
                                      'f_talks_about_worries',
                                      'at_least_one_activity',
                                      'f_help_studying']]
measures_father_std = StandardScaler().fit_transform(measures_father)

# Define the Principal Component Analysis
sklearn_pca = sklearnPCA(n_components=1)


# Create involvement measure for mother and check the factor loadings.
involvement_mother = sklearn_pca.fit_transform(measures_mother_std) * -1
print("Factor loadings of maternal involvement from PCA are {}".format(
    sklearn_pca.components_ * (-1)))


# Create involvement measure for father and check the factor loadings.
involvement_father = sklearn_pca.fit_transform(measures_father_std) * -1
print("Factor loadings of paternal involvement from PCA are {}".format(
    sklearn_pca.components_ * (-1)))

# Standardise the dataset of involvement measure.
involvement_mother_std = StandardScaler().fit_transform(involvement_mother)
involvement_father_std = StandardScaler().fit_transform(involvement_father)

# Add measures to exsisting DataFrame.
parental_involvement_clean = parental_involvement_data.dropna()
parental_involvement_clean['involvement_mother_std'] = involvement_mother_std
parental_involvement_clean['involvement_father_std'] = involvement_father_std

# Create dataframe with only measure and ids.
df = parental_involvement_clean.loc[:, ['cid', 'syear',
                                        'pid',
                                        'involvement_father_std',
                                        'involvement_mother_std']]

# Output the data as a whole.


def save_data(involvement):
    involvement.to_csv(ppj("OUT_DATA", "measure_parental_involvment.csv"),
                       sep=",")


if __name__ == "__main__":
    involvement = df
    save_data(involvement)

from measure_personality_adult import parents_merge

from bld.project_paths import project_paths_join as ppj

# Create mother data.
mother_merge = parents_merge.loc[:,
                                 ['pid_parents',
                                  'mother_sex',
                                  'birth_year_parents',
                                  'german_nationality_parents',
                                  'agreeableness_parents',
                                  'extraversion_parents',
                                  'neuroticism_parents',
                                  'conscientiousness_parents',
                                  'openness_parents']]

mother_merge.columns = [['pid_mother', 'sex_mother', 'birth_year_mother',
                         'german_nationality_mother', 'agreeableness_mother',
                         'extraversion_mother', 'neuroticism_mother',
                         'conscientiousness_mother', 'openness_mother']]

# Output the DataFrame.


def save_data(mother):
    mother.to_csv(ppj("OUT_DATA", "measure_personality_mother.csv"),
                  sep=",")


if __name__ == "__main__":
    mother = mother_merge
    save_data(mother)

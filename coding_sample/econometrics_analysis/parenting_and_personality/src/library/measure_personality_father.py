from measure_personality_adult import parents_merge

from bld.project_paths import project_paths_join as ppj

# Create father data.
father_merge = parents_merge.loc[:,
                                 ['pid_parents',
                                  'mother_sex',
                                  'birth_year_parents',
                                  'german_nationality_parents',
                                  'agreeableness_parents',
                                  'extraversion_parents',
                                  'neuroticism_parents',
                                  'conscientiousness_parents',
                                  'openness_parents']]

father_merge.columns = [['pid_father', 'sex_father', 'birth_year_father',
                         'german_nationality_father', 'agreeableness_father',
                         'extraversion_father', 'neuroticism_father',
                         'conscientiousness_father', 'openness_father']]

# Output the DataFrame.


def save_data(father):
    father.to_csv(ppj("OUT_DATA", "measure_personality_father.csv"),
                  sep=",")


if __name__ == "__main__":
    father = father_merge
    save_data(father)

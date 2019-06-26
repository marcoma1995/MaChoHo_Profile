import pandas as pd

import statsmodels.formula.api as sm

from statsmodels.iolib.summary2 import summary_col

from bld.project_paths import project_paths_join as ppj


# Read the dataset.
data_child = pd.read_csv(ppj("OUT_DATA", "merge.csv"), delimiter=",")
data_males = pd.read_csv(ppj("OUT_DATA", "merge_male.csv"), delimiter=",")
data_females = pd.read_csv(ppj("OUT_DATA", "merge_female.csv"), delimiter=",")

# Creates the constant columns for all datasets.
data_child['const'] = 1
data_females['const'] = 1
data_males['const'] = 1

# Create list for simple-regression.
X1 = ['const', 'involvement_mother_std', 'involvement_father_std']


# Create list for regression with control regressor.
X2 = ['const', 'involvement_mother_std', 'involvement_father_std',
      'years_of_education_mother', 'years_of_education_father',
      'age_difference_mother', 'age_difference_father',
      'mother_lives_in_hh', 'father_lives_in_hh',
      'number_of_siblings', 'migration_background_strict',
      'log_monthly_per_capita_net_income']

#  Create list for regression with charater and other control regressor.
X3 = ['const', 'involvement_mother_std', 'involvement_father_std',
      'neuroticism_mother', 'neuroticism_father',
      'agreeableness_mother', 'agreeableness_father',
      'extraversion_mother', 'extraversion_father',
      'openness_mother', 'openness_father',
      'conscientiousness_mother', 'conscientiousness_father',
      'years_of_education_mother', 'years_of_education_father',
      'age_difference_mother', 'age_difference_father',
      'mother_lives_in_hh', 'father_lives_in_hh',
      'number_of_siblings', 'migration_background_strict',
      'log_monthly_per_capita_net_income']


# Robust Regression on LoC.
loc_child = sm.OLS(
    data_child['loc_score_std'],
    data_child[X1],
    missing='drop').fit().get_robustcov_results()
loc_control = sm.OLS(data_child['loc_score_std'],
                     data_child[X2], missing='drop').fit(
).get_robustcov_results()
loc_personality = sm.OLS(data_child['loc_score_std'],
                         data_child[X3], missing='drop').fit(
).get_robustcov_results()
loc_male = sm.OLS(data_males['loc_score_std'],
                  data_males[X3], missing='drop').fit(
).get_robustcov_results()
loc_female = sm.OLS(data_females['loc_score_std'],
                    data_females[X3], missing='drop').fit(
).get_robustcov_results()

# Robust Regression on Extraversion.
extraversion_child = sm.OLS(data_child['extraversion'],
                            data_child[X1], missing='drop').fit(
).get_robustcov_results()
extraversion_control = sm.OLS(data_child['extraversion'],
                              data_child[X2], missing='drop').fit(
).get_robustcov_results()
extraversion_personality = sm.OLS(data_child['extraversion'],
                                  data_child[X3], missing='drop').fit(
).get_robustcov_results()
extraversion_male = sm.OLS(data_males['extraversion'],
                           data_males[X3], missing='drop').fit(
).get_robustcov_results()
extraversion_female = sm.OLS(data_females['extraversion'],
                             data_females[X3], missing='drop').fit(
).get_robustcov_results()

# Robust Regression on Conscientiousness.
conscientiousness_child = sm.OLS(data_child['conscientiousness'],
                                 data_child[X1], missing='drop').fit(
).get_robustcov_results()
conscientiousness_control = sm.OLS(data_child['conscientiousness'],
                                   data_child[X2], missing='drop').fit(
).get_robustcov_results()
conscientiousness_personality = sm.OLS(data_child['conscientiousness'],
                                       data_child[X3], missing='drop').fit(
).get_robustcov_results()
conscientiousness_male = sm.OLS(data_males['conscientiousness'],
                                data_males[X3], missing='drop').fit(
).get_robustcov_results()
conscientiousness_female = sm.OLS(data_females['conscientiousness'],
                                  data_females[X3], missing='drop').fit(
).get_robustcov_results()

# Robust Regression on Openness.
openness_child = sm.OLS(data_child['openness'],
                        data_child[X1], missing='drop').fit(
).get_robustcov_results()
openness_control = sm.OLS(data_child['openness'],
                          data_child[X2], missing='drop').fit(
).get_robustcov_results()
openness_personality = sm.OLS(data_child['openness'],
                              data_child[X3], missing='drop').fit(
).get_robustcov_results()
openness_male = sm.OLS(data_males['openness'],
                       data_males[X3], missing='drop').fit(
).get_robustcov_results()
openness_female = sm.OLS(data_females['openness'],
                         data_females[X3], missing='drop').fit(
).get_robustcov_results()

# Robust Regression on Neuroticism.
neuroticism_child = sm.OLS(data_child['neuroticism'],
                           data_child[X1], missing='drop').fit(
).get_robustcov_results()
neuroticism_control = sm.OLS(data_child['neuroticism'],
                             data_child[X2], missing='drop').fit(
).get_robustcov_results()
neuroticism_personality = sm.OLS(data_child['neuroticism'],
                                 data_child[X3], missing='drop').fit(
).get_robustcov_results()
neuroticism_male = sm.OLS(data_males['neuroticism'],
                          data_males[X3], missing='drop').fit(
).get_robustcov_results()
neuroticism_female = sm.OLS(data_females['neuroticism'],
                            data_females[X3], missing='drop').fit(
).get_robustcov_results()

# Robust Regression on Agreeableness.
agreeableness_child = sm.OLS(data_child['agreeableness'],
                             data_child[X1], missing='drop').fit(
).get_robustcov_results()
agreeableness_control = sm.OLS(data_child['agreeableness'],
                               data_child[X2], missing='drop').fit(
).get_robustcov_results()
agreeableness_personality = sm.OLS(data_child['agreeableness'],
                                   data_child[X3], missing='drop').fit(
).get_robustcov_results()
agreeableness_male = sm.OLS(data_males['agreeableness'],
                            data_males[X3], missing='drop').fit(
).get_robustcov_results()
agreeableness_female = sm.OLS(data_females['agreeableness'],
                              data_females[X3], missing='drop').fit(
).get_robustcov_results()

# Create order of control variable in Summary.
Order = ['const', 'involvement_mother_std', 'involvement_father_std',
         'openness_mother',
         'openness_father', 'conscientiousness_mother',
         'conscientiousness_father', 'extraversion_mother',
         'extraversion_father', 'agreeableness_mother',
         'agreeableness_father', 'neuroticism_mother',
         'neuroticism_father', 'years_of_education_mother',
         'years_of_education_father', 'number_of_siblings',
         'log_monthly_per_capita_net_income',
         'migration_background_strict',
         'mother_lives_in_hh', 'father_lives_in_hh',
         'age_difference_mother', 'age_difference_father'
         ]

# Create info_dict for Summary.
info_dict = {'R-squared': lambda x: f"{x.rsquared:.2f}",
             'No. observations': lambda x: f"{int(x.nobs):d}"}

# Summary of simple regression with children personality and parental
# involvnment.
results_table_1 = summary_col(results=[loc_child, openness_child,
                                     conscientiousness_child,
                                     extraversion_child, agreeableness_child,
                                     neuroticism_child],
                            float_format='%0.2f',
                            stars=True,
                            model_names=['LoC',
                                         'O',
                                         'C',
                                         'E',
                                         'A',
                                         'N'],
                            info_dict=info_dict)

results_table_1.add_title('Children Sample Regressions')

results_table_simple = results_table_1.as_latex()

# Summary of regression with control variable.
results_table_2 = summary_col(results=[loc_control, openness_control,
                                     conscientiousness_control,
                                     extraversion_control,
                                     agreeableness_control,
                                     neuroticism_control],
                            float_format='%0.2f',
                            stars=True,
                            model_names=['LoC',
                                         'O',
                                         'C',
                                         'E',
                                         'A',
                                         'N'],
                            info_dict=info_dict,
                            # order for showing results
                            regressor_order=Order)

results_table_2.add_title('Children Sample Regressions (Control)')

results_table_control = results_table_2.as_latex()

# Summary of regression with control variable.
results_table_3 = summary_col(results=[loc_personality, openness_personality,
                                     conscientiousness_personality,
                                     extraversion_personality,
                                     agreeableness_personality,
                                     neuroticism_personality],
                            float_format='%0.2f',
                            stars=True,
                            model_names=['LoC',
                                         'O',
                                         'C',
                                         'E',
                                         'A',
                                         'N'],
                            info_dict=info_dict,
                            # order for showing results
                            regressor_order=Order)

results_table_3.add_title('Children Sample Regressions (Extended Control)')

results_table_extended = results_table_3.as_latex()


# Summary of regression with control variable. (Female Sample)
results_table_4 = summary_col(results=[loc_female,
                                     openness_female,
                                     conscientiousness_female,
                                     extraversion_female,
                                     agreeableness_female,
                                     neuroticism_female],
                            float_format='%0.2f',
                            stars=True,
                            model_names=['LoC',
                                         'O',
                                         'C',
                                         'E',
                                         'A',
                                         'N'],
                            info_dict=info_dict,
                            regressor_order=Order)

results_table_4.add_title('Females Sample Regressions')

results_table_female = results_table_4.as_latex()


# Summary of regression with control variable. (Male Sample)
results_table_5 = summary_col(results=[loc_male,
                                     openness_male,
                                     conscientiousness_male,
                                     extraversion_male,
                                     agreeableness_male,
                                     neuroticism_male],
                            float_format='%0.2f',
                            stars=True,
                            model_names=['LoC',
                                         'O',
                                         'C',
                                         'E',
                                         'A',
                                         'N'],
                            info_dict=info_dict,
                            regressor_order=Order)

results_table_5.add_title('Males Sample Regressions')

results_table_male = results_table_5.as_latex()

file = open(ppj("OUT_ANALYSIS", "regression_result.tex"), 'w')

file.write(results_table_simple)

file.write(results_table_control)

file.write(results_table_extended)

file.write(results_table_female)

file.write(results_table_male)

file.close()

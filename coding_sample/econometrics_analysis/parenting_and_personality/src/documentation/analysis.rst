.. _analysis:

************************************
Main model estimations / simulations
************************************

Documentation of the code in *src.analysis*. This is the core of the project. 


We do our regression analysis here.
We in total have 5 different robust regression based on the dataset I created in data_management. 

Model 1. Simple Robust Regression, with no control variable

Model 2. Robust Regression with control, which we regress the two variable with other household control variable for example, income, if parent is living at home.

Model 3. Robust Regression controlling with parents personality and all other control variable from Model 2. 

Model 4. Robust Regression (Male Sample) controlling with variable in Model 3. In here, we only regress with male sample.

Model 5. Robust Regression (Female Sample) controlling with variable in Model 3. In here, we only regress with female sample.
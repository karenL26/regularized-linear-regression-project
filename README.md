# Regularized Linear Regression Project
> 4 Geeks Academy - project by @karenL26

In this project it was built a regularized linear regression model to predict the availability of ICU Beds using the information about Active Primary Care Physicians per 100000 Population 2018 (AAMC), Total nurse practitioners (2019), Total Hospitals (2019) and Total Specialist Physicians (2019).

The project start with an EDA[^1] over a large dataset with a lot of features related to socio demographic and health resources data by county in the United States, right before the Covid-19 pandemic started (data from 2018 and 2019). It was taken from the WIDS 2022 competition on Kaggle. 
From the EDA the target variable **ICU Beds_x** and the predictors were selected.

In the model construction[^2] was applied the LASSO model which help also in the feature selection to obtain the most important features that influence in the target variable. Finally using hypertune for the parameters was obtained a increasing the score. 



[^1]: You can access to the EDA clicking [here](src/explore_r_lr.ipynb).

[^2]: You can access to the model construction and results clicking [here](src/explore_r_lr.ipynb) and for the rawest version clicking [here](src/app.py).
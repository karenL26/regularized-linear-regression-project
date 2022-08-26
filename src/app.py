import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt  
import seaborn as sns
import plotly.express as px
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso
from numpy import arange
from sklearn.linear_model import LassoCV
from sklearn.model_selection import RepeatedKFold

######################
# Data Preprocessing #
######################

# Reading the dataset
df_raw = pd.read_csv("https://raw.githubusercontent.com/4GeeksAcademy/regularized-linear-regression-project-tutorial/main/dataset.csv")

# We make a copy of the original dataset
df = df_raw.copy()
# Feature selection based on defined target ICU Beds_x
df = df[['fips', 'Active Physicians per 100000 Population 2018 (AAMC)', 'Total Active Patient Care Physicians per 100000 Population 2018 (AAMC)', 'Active Primary Care Physicians per 100000 Population 2018 (AAMC)', 
            'Total nurse practitioners (2019)', 'Total physician assistants (2019)', 'Total Hospitals (2019)', 'Total Specialist Physicians (2019)', 'ICU Beds_x', 'Total Population']]

# We remove the outliers only in the cases where it doesn't mean a bigger loss of data
df=df.drop(df[df['Active Physicians per 100000 Population 2018 (AAMC)'] > 356].index)
df=df.drop(df[df['Total Active Patient Care Physicians per 100000 Population 2018 (AAMC)'] > 297.4].index)
df=df.drop(df[df['Active Primary Care Physicians per 100000 Population 2018 (AAMC)'] > 120].index)


#####################
# Model and results #
#####################

# Separating the target variable (y) from the predictors(X)
X = df.drop(['ICU Beds_x'], axis=1)
y = df['ICU Beds_x']

# Spliting the dataset into training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Lasso with default parameters
pipeline = make_pipeline(StandardScaler(), Lasso()) 
pipeline.fit(X_train, y_train)
y_pred=pipeline.predict(X_test)
print(pipeline[1].coef_, pipeline[1].intercept_)

# We keep the features that are relevant for the model, that is features with a weight greater than 0
mask=pipeline[1].coef_!=0
X=X.loc[:,mask]

# Last step, we must split the dataset into training and testing with the new feature selection
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

########### Hyperparameters using cross validation ###########

cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
# define model
pipeline_opt = make_pipeline(StandardScaler(), LassoCV(alphas=arange(0, 1, 0.01), cv=cv, n_jobs=-1))
# fit model
pipeline_opt.fit(X, y)

print("Score with alpha in train dataset:", round(pipeline_opt.score(X_train, y_train), 4))
print("Score with alpha in test dataset:", round(pipeline_opt.score(X_test, y_test), 4)) 

# We save the model with joblib
dirname = os.path.dirname(__file__)
filename = os.path.join(dirname, '../data/processed/reg_lr.pkl')

joblib.dump(pipeline_opt, filename)

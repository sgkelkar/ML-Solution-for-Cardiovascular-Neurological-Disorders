#!/usr/bin/env python
# coding: utf-8

# ## Problem 2 Introduction

# 1a. Sneha Kelkar, sgk18001
# 
# 1b. In this notebook, we are comparing two AI models to predict the most likely value of the motor UPDRS for Parkinson's patients. Primarily, we are using AdaBoost and Random forest.
# In the end, we will compare the results displayed by each model and pick the best

# ### 1c. Methods

# 1.  As described, no missing value is recorded, so handling null values is not an issue
# 2. Even though it is unclear whether we should scale the values or not, through trial and error, scaling the variables produced better results
# 3. We will use GridSearch to optimize the models, comparing the validation values. We will optimize the models using GridSearchCV
# 4. We will use cross validation to obtain reliable estimates of the test mean squared error and make sure no patient is simultaneously represented in both split data sets

# In[41]:


# ---------------------------------------------------------------------------------------------
# load the libraries that are required for this project:
#---------------------------------------------------------------------------------------------
import pandas as pd             # Pandas is for data analysis and structure manipulation
import numpy as np              # NumPy is for numerical operations 


# ## Data Cleaning

# In[42]:


# load the dataset and drop 'total_UPDRS'
df = pd.read_csv('dataset_parkinson.csv')
df = df.drop('total_UPDRS', axis = 1)


# In[43]:


#inspecting the data types and checking for missing values
df.info()


# In[44]:


df.describe()


# ### Scaling and splitting the data

# In[45]:


from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
unique_patients = df['name'].unique()
np.random.shuffle(unique_patients)
print(unique_patients)

# Split unique patients into train and test groups
train_patients, test_patients = train_test_split(unique_patients, test_size=0.3, random_state=42)

train_df = df[df['name'].isin(train_patients)]
test_df = df[df['name'].isin(test_patients)]

X_train = train_df.drop(['motor_UPDRS', 'name'], axis = 1)
X_test = test_df.drop(['motor_UPDRS', 'name'], axis = 1)
y_train = train_df['motor_UPDRS']
y_test = test_df['motor_UPDRS']

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# #### after getting inconsistent results for sample values using 'StratifiedShuffleSplit' we can use 'test_train_split' from the same library which randomly splits the data into trianing and testing sets automatically for a balanced data set

# ## Model Training

# In[46]:


from sklearn.model_selection import cross_val_score 
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
#based off of Practice 04


# ## Random Forest

# In[47]:


# Hyper-parameter grid
param_grid_rf = {
    'n_estimators': [10, 50, 100, 200],
    'max_depth': [10, 20, 30, 40],
    'max_features': [None, 'sqrt', 'log2'],
    'bootstrap': [True],
}

rf = RandomForestRegressor(random_state =42)
grid_rf = GridSearchCV(rf, param_grid=param_grid_rf, scoring='neg_mean_squared_error', cv=3, n_jobs=-1)
grid_rf.fit(X_train_scaled, y_train)


best_params_rf = grid_rf.best_params_
best_score_rf = grid_rf.best_score_
# Print the best parameters and their score on the validation set
print("Best parameters for Random Forest:", best_params_rf)
print("Best MSE on the validation set for Random Forest:", best_score_rf)
# Now you can retrain your model on the full training set with the best parameters
rf_best = RandomForestRegressor(**best_params_rf, random_state=42)
rf_best.fit(X_train_scaled, y_train)

y_pred = rf_best.predict(X_test_scaled)
print(f'Mean squared error: {mean_squared_error(y_test, y_pred)}')


# ## Gradient Boosting (AdaBoost)

# In[48]:


param_grid_adaboost = {
    'n_estimators': [50, 100, 200],
    'learning_rate': [0.01, 0.1, 1],
    'estimator__max_depth': [1, 2, 3],
}
base_estimator = DecisionTreeRegressor()
estimator = AdaBoostRegressor(estimator=base_estimator)
grid_search_ada = GridSearchCV(estimator=estimator, param_grid=param_grid_adaboost, scoring='neg_mean_squared_error', cv=3)

grid_search_ada.fit(X_train_scaled, y_train)

best_params_adaboost = grid_search_ada.best_params_
best_score_adaboost = grid_search_ada.best_score_
# Print the best parameters and their score on the validation set
print("Best parameters for AdaBoost:", best_params_adaboost)
print("Best score on the validation set for AdaBoost:", best_score_adaboost)

best_model = DecisionTreeRegressor(max_depth = best_params_adaboost['estimator__max_depth'])
best_model = AdaBoostRegressor(estimator = best_model, n_estimators = best_params_adaboost['n_estimators'], 
                               learning_rate = best_params_adaboost['learning_rate'], random_state=42)
best_model.fit(X_train_scaled, y_train)

y_pred = best_model.predict(X_test_scaled)
test_MSE = mean_squared_error(y_test, y_pred)
print(f" Mean Squared Error on the test set: {test_MSE:.4f}")


# ## 1d. Results
# According to the results displayed, Forest Tree shows better results. Therefore, we will pick it model for our application.
# When allowing patients to enter both the training and the test dataset, the results become much more reasonable.
# As it currently stands, the model cannot be used for prediction due to the lack of consistency in the data, this was. 
# EDIT:
# I noticed that the more I ran the code, the worse the results became.. 

# In[ ]:





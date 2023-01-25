# -*- coding: utf-8 -*-
"""
Created on Wed Oct  6 11:41:48 2021

@author: Ramanzani S. Kalule
email: kramanzani@gmail.com
"""
import pandas as pd
features = pd.read_csv('PoreProps2.csv')
print(features.head(5))
print('The shape of our features is:', features.shape)
# Descriptive statistics for each column
print(features.describe())
import numpy as np
# Labels are the values we want to predict
labels = np.array(features[['Permeability','Porosity']])
features= features.drop(['Permeability','Area','MeanIntensity','FileName','Porosity'], axis = 1)
# Saving feature names 
feature_list = list(features.columns)
features = np.array(features)

from sklearn.model_selection import train_test_split
train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size = 0.25, random_state = 42)
print('Training Features Shape:', train_features.shape)
print('Training Labels Shape:', train_labels.shape)
print('Testing Features Shape:', test_features.shape)
print('Testing Labels Shape:', test_labels.shape)

#############################################################################################
print('Random serach for Random Forest on a multi input permeability')
from sklearn.experimental import enable_hist_gradient_boosting  # noqa
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LassoCV, MultiTaskLassoCV, RidgeCV, LinearRegression
from sklearn.multioutput import MultiOutputRegressor

rf = MultiOutputRegressor(RandomForestRegressor(random_state = 42))

''' Random Hyperparameter Grid '''
# To use RandomizedSearchCV, we first need to create a parameter grid to sample from during fitting:
from sklearn.model_selection import RandomizedSearchCV
import numpy as np
estimator__n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
estimator__max_features = ['auto', 'sqrt']
estimator__max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
estimator__max_depth.append(None)
estimator__min_samples_split = [2, 5, 10]
estimator__min_samples_leaf = [1, 2, 4]
estimator__bootstrap = [True, False]

# Create the random grid
random_grid = {'estimator__n_estimators': estimator__n_estimators,
               'estimator__max_features': estimator__max_features,
               'estimator__max_depth': estimator__max_depth,
               'estimator__min_samples_split': estimator__min_samples_split,
               'estimator__min_samples_leaf': estimator__min_samples_leaf,
               'estimator__bootstrap': estimator__bootstrap}

# Use the random grid to search for best hyperparameters
rf = MultiOutputRegressor(RandomForestRegressor())
rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 100, cv = 3, verbose=2, random_state=42, n_jobs = -1)
rf_random.fit(train_features, train_labels)
rf_random.best_params_
print(rf_random.best_params_)

def evaluate(model, test_features, test_labels):
    predictions = model.predict(test_features)
    errors = abs(predictions - test_labels)
    mape = 100 * np.mean(errors / test_labels)
    mse = np.mean(errors*errors)
    accuracy = 100 - mape
    print('Model Performance')
    print('Average Error: {:0.4f} degrees.'.format(np.mean(errors)))
    print('mape = {:0.2f}%.'.format(mape))
    print('mse = {:0.4f}.'.format(mse))
    print('Accuracy = {:0.2f}%.'.format(accuracy))
       
    sse = ((predictions - test_labels) ** 2).sum(axis=0, dtype=np.float64)
    tse = ((test_labels - np.average(test_labels, axis=0)) ** 2).sum(axis=0, dtype=np.float64)
    r2_score = np.mean(1 - (sse / tse))
    print('r_2 = {:0.4f}.'.format(r2_score))

    return accuracy, r2_score, sse, tse

base_model = MultiOutputRegressor(RandomForestRegressor(n_estimators = 10, random_state = 42))
base_model.fit(train_features, train_labels)
base_accuracy = evaluate(base_model, test_features, test_labels)

best_random = rf_random.best_estimator_
random_accuracy = evaluate(best_random, test_features, test_labels)
print(base_accuracy)
print(random_accuracy)

#################################################################################
print('Random serach for LassoCV on a multi input permeability')
lasso = MultiOutputRegressor(LassoCV())
#RandomizedSearchCV
from sklearn.model_selection import RandomizedSearchCV
import numpy as np

estimator__max_iter = [100, 500, 1000, 1500, 2000]
estimator__n_alphas = [50, 75, 100, 500]
estimator__eps = [1e-1, 1e-2, 1e-3, 1e-4]
estimator__cv = [3,5,7,9]
# Create the random grid
random_grid = {'estimator__max_iter': estimator__max_iter,
               'estimator__n_alphas': estimator__n_alphas,
               'estimator__cv': estimator__cv,
               'estimator__eps': estimator__eps}

# Use the random grid to search for best hyperparameters
lasso = MultiOutputRegressor(LassoCV())

lasso_random = RandomizedSearchCV(estimator = lasso, param_distributions = random_grid, n_iter = 100, cv = 3, verbose=2, random_state=42, n_jobs = -1)
# Fit the random search model
lasso_random.fit(train_features, train_labels)
lasso_random.best_params_
print(lasso_random.best_params_)

base_model = MultiOutputRegressor(LassoCV())
base_model.fit(train_features, train_labels)
base_accuracy = evaluate(base_model, test_features, test_labels)

best_random = lasso_random.best_estimator_
random_accuracy = evaluate(best_random, test_features, test_labels)

print(base_accuracy)
print(random_accuracy)
#################################################################################
print('Random serach for Linear Regression on a multi input permeability')
lr = MultiOutputRegressor(LinearRegression())
from pprint import pprint
print('Parameters currently in use:\n')
pprint(lr.get_params())

''' Random Hyperparameter Grid '''

from sklearn.model_selection import RandomizedSearchCV
import numpy as np
# Important hyperparameters
estimator__fit_intercept = ['True', 'False']
estimator__normalize = ['True', 'False']
# Create the random grid
random_grid = {'estimator__normalize': estimator__normalize,
               'estimator__fit_intercept': estimator__fit_intercept}
pprint(random_grid)

# Use the random grid to search for best hyperparameters
lr = MultiOutputRegressor(LinearRegression())
lr_random = RandomizedSearchCV(estimator = lr, param_distributions = random_grid, 
                               n_iter = 100, cv = 3, verbose=2, random_state=42, n_jobs = -1)
# Fit the random search model
lr_random.fit(train_features, train_labels)
lr_random.best_params_
print(lr_random.best_params_)

base_model = MultiOutputRegressor(LinearRegression())
base_model.fit(train_features, train_labels)
base_accuracy = evaluate(base_model, test_features, test_labels)

best_random = lr_random.best_estimator_
random_accuracy = evaluate(best_random, test_features, test_labels)

print(base_accuracy)
print(random_accuracy)
#################################################################################
print('Random serach for Ridge Regression on a multi input permeability')
rg = MultiOutputRegressor(RidgeCV())
from pprint import pprint
# Look at parameters used by our current forest
print('Parameters currently in use:\n')
pprint(rg.get_params())

from sklearn.model_selection import RandomizedSearchCV
import numpy as np

estimator__fit_intercept = ['True', 'False']
estimator__normalize = ['True', 'False']
estimator__alpha_per_target = ['True', 'False']
estimator__cv = [3,5,7,9]
# Create the random grid
random_grid = {'estimator__fit_intercept': estimator__fit_intercept,
               'estimator__normalize': estimator__normalize,
               'estimator__cv': estimator__cv}
pprint(random_grid)

# Use the random grid to search for best hyperparameters
rg = MultiOutputRegressor(RidgeCV())
rg_random = RandomizedSearchCV(estimator = rg, param_distributions = random_grid, n_iter = 100, cv = 3, verbose=2, random_state=42, n_jobs = -1)
# Fit the random search model
rg_random.fit(train_features, train_labels)
rg_random.best_params_
print(rg_random.best_params_)

base_model = MultiOutputRegressor(RidgeCV())
base_model.fit(train_features, train_labels)
base_accuracy = evaluate(base_model, test_features, test_labels)

best_random = rg_random.best_estimator_
random_accuracy = evaluate(best_random, test_features, test_labels)

print(base_accuracy)
print(random_accuracy)
#################################################################################
print('Random serach for Gradient Boosting Regression on a multi input permeability')
gb = MultiOutputRegressor(GradientBoostingRegressor())
from pprint import pprint
# Look at parameters used by our current forest
print('Parameters currently in use:\n')
pprint(gb.get_params())

from sklearn.model_selection import RandomizedSearchCV
import numpy as np

# Maximum number iterations
estimator__learning_rate = [0.1, 0.01, 0.001,0.0001]
estimator__n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
estimator__criterion = ['friedman_mse', 'squared_error', 'mse', 'mae']
estimator__loss = ['squared_error', 'ls', 'absolute_error', 'lad', 'huber', 'quantile']
estimator__max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
estimator__max_depth.append(None)
estimator__min_samples_split = [2, 5, 10]
estimator__min_samples_leaf = [1, 2, 4]
# Create the random grid
random_grid = {'estimator__learning_rate': estimator__learning_rate,
               'estimator__n_estimators': estimator__n_estimators,
               'estimator__criterion': estimator__criterion,
              'estimator__loss': estimator__loss,
              'estimator__max_depth': estimator__max_depth,
              'estimator__min_samples_split': estimator__min_samples_split,
              'estimator__min_samples_leaf': estimator__min_samples_leaf}
pprint(random_grid)

# Use the random grid to search for best hyperparameters
gb = MultiOutputRegressor(GradientBoostingRegressor())
gb_random = RandomizedSearchCV(estimator = gb, param_distributions = random_grid, n_iter = 100, cv = 3, verbose=2, random_state=42, n_jobs = -1)
# Fit the random search model
gb_random.fit(train_features, train_labels)
gb_random.best_params_
print(gb_random.best_params_)

base_model = MultiOutputRegressor(GradientBoostingRegressor())
base_model.fit(train_features, train_labels)
base_accuracy = evaluate(base_model, test_features, test_labels)
best_random = gb_random.best_estimator_
random_accuracy = evaluate(best_random, test_features, test_labels)
print(base_accuracy)
print(random_accuracy)
#################################################################################
print('.......................................End.........................................................')
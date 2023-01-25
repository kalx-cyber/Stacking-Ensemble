# -*- coding: utf-8 -*-
"""
Created on Wed Oct  6 11:20:04 2021

@author: Ramanzani S. Kalule
email: kramanzani@gmail.com
"""
import os
os.getcwd()
import pandas as pd
features = pd.read_csv('PoreProps2.csv')
features.head()
print('The shape of our features is:', features.shape)

# Descriptive statistics for each column
features.describe()

import numpy as np
# Labels are the values we want to predict
labels = np.array(features['Porosity'])
features= features.drop(['Porosity','Area','MeanIntensity','FileName'], axis = 1)
feature_list = list(features.columns)
features = np.array(features)

# Using Skicit-learn to split data into training and testing sets
from sklearn.model_selection import train_test_split
# Split the data into training and testing sets
train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size = 0.25, random_state = 42)
print('Training Features Shape:', train_features.shape)
print('Training Labels Shape:', train_labels.shape)
print('Testing Features Shape:', test_features.shape)
print('Testing Labels Shape:', test_labels.shape)

#############################################################################################
print('Random serach for Random Forest on a single input permeability')
from sklearn.experimental import enable_hist_gradient_boosting  # noqa
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LassoCV, MultiTaskLassoCV, RidgeCV, LinearRegression
from sklearn.multioutput import MultiOutputRegressor

rf = RandomForestRegressor(random_state = 42)
from pprint import pprint
print('Parameters currently in use:\n')
pprint(rf.get_params())

from sklearn.model_selection import RandomizedSearchCV
import numpy as np
n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
max_features = ['auto', 'sqrt']
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
max_depth.append(None)
min_samples_split = [2, 5, 10]
min_samples_leaf = [1, 2, 4]
bootstrap = [True, False]

random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}
pprint(random_grid)

rf = RandomForestRegressor()
# Random search of parameters
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
    r2_score = 1 - (sse / tse)
    print('r_2 = {:0.4f}.'.format(r2_score))

    return accuracy, r2_score, sse, tse

base_model = RandomForestRegressor(n_estimators = 10, random_state = 42)
base_model.fit(train_features, train_labels)
base_accuracy = evaluate(base_model, test_features, test_labels)

best_random = rf_random.best_estimator_
random_accuracy = evaluate(best_random, test_features, test_labels)
print(base_accuracy)
print(random_accuracy)
#################################################################################
print('Random serach for LassoCV on a single input permeability')

from sklearn.experimental import enable_hist_gradient_boosting  # noqa
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LassoCV, MultiTaskLassoCV, RidgeCV, LinearRegression
from sklearn.multioutput import MultiOutputRegressor

lasso = LassoCV()
from pprint import pprint
# Look at parameters used by our current forest
print('Parameters currently in use:\n')
pprint(lasso.get_params())

from sklearn.model_selection import RandomizedSearchCV
import numpy as np
max_iter = [100, 500, 1000, 1500, 2000]
n_alphas = [50, 75, 100, 500]
eps = [1e-1, 1e-2, 1e-3, 1e-4]
cv = [3,5,7,9]

random_grid = {'max_iter': max_iter,
               'n_alphas': n_alphas,
               'cv': cv,
               'eps': eps}
pprint(random_grid)

lasso = LassoCV()
lasso_random = RandomizedSearchCV(estimator = lasso, param_distributions = random_grid, n_iter = 100, cv = 3, verbose=2, random_state=42, n_jobs = -1)
lasso_random.fit(train_features, train_labels)
lasso_random.best_params_
print(lasso_random.best_params_)

base_model = LassoCV()
base_model.fit(train_features, train_labels)
base_accuracy = evaluate(base_model, test_features, test_labels)

best_random = lasso_random.best_estimator_
random_accuracy = evaluate(best_random, test_features, test_labels)
print(base_accuracy)
print(random_accuracy)

###############################################################################################
print('Random serach for Linear Regression on a single input permeability')

from sklearn.experimental import enable_hist_gradient_boosting  # noqa
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LassoCV, MultiTaskLassoCV, RidgeCV, LinearRegression
from sklearn.multioutput import MultiOutputRegressor

lr = LinearRegression()
from pprint import pprint
print('Parameters currently in use:\n')
pprint(lr.get_params())

from sklearn.model_selection import RandomizedSearchCV
import numpy as np
fit_intercept = ['True', 'False']
normalize = ['True', 'False']

random_grid = {'normalize': normalize,
               'fit_intercept': fit_intercept}
pprint(random_grid)

lr = LinearRegression()
lr_random = RandomizedSearchCV(estimator = lr, param_distributions = random_grid, n_iter = 100, cv = 3, verbose=2, random_state=42, n_jobs = -1)
lr_random.fit(train_features, train_labels)
lr_random.best_params_
print(lr_random.best_params_)

base_model = LinearRegression()
base_model.fit(train_features, train_labels)
base_accuracy = evaluate(base_model, test_features, test_labels)

best_random = lr_random.best_estimator_
random_accuracy = evaluate(best_random, test_features, test_labels)

print(base_accuracy)
print(random_accuracy)
##############################################################################################################
print('Random serach for Ridge Regression on a single input permeability')

from sklearn.experimental import enable_hist_gradient_boosting  # noqa
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LassoCV, MultiTaskLassoCV, RidgeCV, LinearRegression
from sklearn.multioutput import MultiOutputRegressor

rg = RidgeCV()
from pprint import pprint
print('Parameters currently in use:\n')
pprint(rg.get_params())

from sklearn.model_selection import RandomizedSearchCV
import numpy as np

fit_intercept = ['True', 'False']
normalize = ['True', 'False']
alpha_per_target = ['True', 'False']
cv = [3,5,7,9]

# Create the random grid
random_grid = {'fit_intercept': fit_intercept,
               'normalize': normalize,
               'cv': cv}
pprint(random_grid)

rg = RidgeCV()
rg_random = RandomizedSearchCV(estimator = rg, param_distributions = random_grid, n_iter = 100, cv = 3, verbose=2, random_state=42, n_jobs = -1)
# Fit the random search model
rg_random.fit(train_features, train_labels)
rg_random.best_params_
print(rg_random.best_params_)

base_model = RidgeCV()
base_model.fit(train_features, train_labels)
base_accuracy = evaluate(base_model, test_features, test_labels)

best_random = rg_random.best_estimator_
random_accuracy = evaluate(best_random, test_features, test_labels)

print(base_accuracy)
print(random_accuracy)
#################################################################################################
print('Random serach for Gradient Boosting Regression on a single input permeability')

gb = GradientBoostingRegressor()
from pprint import pprint
# Look at parameters used by our current forest
print('Parameters currently in use:\n')
pprint(gb.get_params())

from sklearn.model_selection import RandomizedSearchCV
import numpy as np
learning_rate = [0.1, 0.01, 0.001,0.0001]
n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
criterion = ['friedman_mse', 'squared_error', 'mse', 'mae']
loss = ['squared_error', 'ls', 'absolute_error', 'lad', 'huber', 'quantile']
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
max_depth.append(None)
min_samples_split = [2, 5, 10]
min_samples_leaf = [1, 2, 4]

random_grid = {'learning_rate': learning_rate,
               'n_estimators': n_estimators,
               'criterion': criterion,
              'loss': loss,
              'max_depth': max_depth,
              'min_samples_split': min_samples_split,
              'min_samples_leaf': min_samples_leaf}
pprint(random_grid)

gb = GradientBoostingRegressor()
gb_random = RandomizedSearchCV(estimator = gb, param_distributions = random_grid, n_iter = 100, cv = 3, verbose=2, random_state=42, n_jobs = -1)
gb_random.fit(train_features, train_labels)
gb_random.best_params_
print(gb_random.best_params_)

base_model = GradientBoostingRegressor()
base_model.fit(train_features, train_labels)
base_accuracy = evaluate(base_model, test_features, test_labels)

best_random = gb_random.best_estimator_
random_accuracy = evaluate(best_random, test_features, test_labels)

print(base_accuracy)
print(random_accuracy)
print('.......................................End.........................................................')
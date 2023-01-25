# -*- coding: utf-8 -*-
"""
Created on Wed Oct  6 12:02:16 2021

@author: Ramanzani S. Kalule
email: kramanzani@gmail.com
"""

'''Single-Output (Permeability) Combine predictors using stacking
INSPIRED BY:  
#  Guillaume Lemaitre <g.lemaitre58@gmail.com>
#  Maria Telenczuk    <https://github.com/maikia>
'''

from sklearn import set_config
set_config(display='diagram')


'''We will use Rock Poreprops_ dataset.'''

import os
os.getcwd()
import numpy as np
import pandas as pd
import os
from sklearn.datasets import fetch_openml
from sklearn.utils import shuffle
from sklearn import preprocessing

os.getcwd()

def load_data():
    df = pd.read_csv("PoreProps2.csv")
    df = df.drop("FileName",1)
    print(df.info())
    print(df.head(5))
    
    X = df.drop('Permeability', axis=1)
    X = X.drop('MeanIntensity', axis=1)
    X = X.drop('Area', axis=1)
    
    '''  min-max scaling ''' 
    min_max_scaler = preprocessing.MinMaxScaler()
#     X = min_max_scaler.fit_transform(X)
    X = pd.DataFrame(X)

    '''  standardisation ''' 
    #X = preprocessing.scale(X)

    
    y = df['Permeability']

    features = ['BoundingBoxArea',
                'ConvexArea',
                'Eccentricity',
                'equivalent_diameter',
                'orientation',
                'MajorAxisLength',
                'MinorAxisLength',
                'MaxIntensity',
                'MinIntensity',
                'Perimeter',
                'filled_area',
                'solidity',
                'Porosity',
                'Permeability']

    #X = X[features]
    X, y = shuffle(X, y, random_state=0)

    #X = X[:600]
    #y = y[:600]
    return X, y# np.log(y)


X, y = load_data()

from sklearn.model_selection import train_test_split
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 42)
print('Training Features Shape:', X_train.shape)
print('Training Labels Shape:', y_train.shape)
print('Testing Features Shape:', X_test.shape)
print('Testing Labels Shape:', y_test.shape) 

from sklearn.compose import make_column_selector
#cat_selector = make_column_selector(dtype_include=object)
num_selector = make_column_selector(dtype_include=np.number)
#cat_selector(X)
#print(cat_selector(X))
print(num_selector(X))

from sklearn.compose import make_column_transformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OrdinalEncoder
#cat_tree_processor = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
num_tree_processor = SimpleImputer(strategy="mean", add_indicator=True)
tree_preprocessor = make_column_transformer((num_tree_processor, num_selector))
print(tree_preprocessor)


from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
#cat_linear_processor = OneHotEncoder(handle_unknown="ignore")
num_linear_processor = make_pipeline(StandardScaler(), SimpleImputer(strategy="mean", add_indicator=True))
linear_preprocessor = make_column_transformer((num_linear_processor, num_selector))
print(linear_preprocessor)


from sklearn.linear_model import LassoCV
lasso_pipeline = make_pipeline(linear_preprocessor, LassoCV(n_alphas = 100, max_iter = 1000, eps = 0.0001, cv = 7))
print(lasso_pipeline)

from sklearn.linear_model import RidgeCV 
Ridge_pipeline = make_pipeline(linear_preprocessor, RidgeCV(fit_intercept=True, normalize = True, cv = 3)) 
print(Ridge_pipeline)

from sklearn.linear_model import LinearRegression
lr_pipeline = make_pipeline(linear_preprocessor, LinearRegression(normalize=True, fit_intercept=True, copy_X=True))
print(lr_pipeline)

from sklearn.ensemble import RandomForestRegressor
rf_pipeline = make_pipeline(tree_preprocessor, RandomForestRegressor(random_state=42, n_estimators = 1400, min_samples_split = 2, min_samples_leaf = 1, max_features = 'auto', max_depth = 100, bootstrap = True))
print(rf_pipeline)

from sklearn.ensemble import GradientBoostingRegressor
gb_pipeline = make_pipeline(tree_preprocessor, GradientBoostingRegressor(n_estimators=1000, min_samples_split = 2, min_samples_leaf = 2, max_depth=None, loss = 'huber', learning_rate = 0.1, criterion = 'mae', random_state = 42))
print(gb_pipeline)

from sklearn.ensemble import StackingRegressor
estimators = [('Linear', lr_pipeline),
              ('LassoCV', lasso_pipeline),
              ('RidgeCV', Ridge_pipeline),
              ('Random Forest', rf_pipeline),
              ('Gradient Boosting', gb_pipeline)]

stacking_regressor = StackingRegressor(estimators=estimators, final_estimator=LinearRegression(normalize=True, fit_intercept=True, copy_X=True))
stacking_regressor
print(StackingRegressor)


import time
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_validate, cross_val_predict

def plot_regression_results(ax, y_true, y_pred, title, scores, elapsed_time):
    """Scatter plot of the predicted vs true targets."""
    ax.plot([y_true.min(), y_true.max()],
            [y_true.min(), y_true.max()],
            '--r', linewidth=2)
    ax.scatter(y_true, y_pred, alpha=0.2)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()
    ax.spines['left'].set_position(('outward', 10))
    ax.spines['bottom'].set_position(('outward', 10))
    ax.set_xlim([y_true.min(), y_true.max()])
    ax.set_ylim([y_true.min(), y_true.max()])
    ax.set_xlabel('Measured')
    ax.set_ylabel('Predicted')
    extra = plt.Rectangle((0, 0), 0, 0, fc="w", fill=False, edgecolor='none', linewidth=0)
    ax.legend([extra], [scores], loc='upper left')
    title = title + ' {:.2f} seconds'.format(elapsed_time)
    ax.set_title(title)


fig, axs = plt.subplots(3, 2, figsize=(9, 7))
axs = np.ravel(axs)

for ax, (name, est) in zip(axs, estimators + [('Stacking Regressor', stacking_regressor)]):
    start_time = time.time()
    score = cross_validate(est, X_train, y_train,scoring=['r2', 'neg_mean_absolute_error'],n_jobs=-1, verbose=0)
    elapsed_time = time.time() - start_time

    y_pred = cross_val_predict(est, X_test, y_test, n_jobs=-1, verbose=0)

    plot_regression_results(ax, y_test, y_pred,name, (r'$R^2={:.4f} \pm {:.2f}$' + '\n' + r'$MAE={:.4f} \pm {:.2f}$')
        .format(np.mean(score['test_r2']), np.std(score['test_r2']),-np.mean(score['test_neg_mean_absolute_error']),
                np.std(score['test_neg_mean_absolute_error'])), elapsed_time)

plt.suptitle('Single (linear/ non-linear) versus stacked predictors')
plt.tight_layout()
plt.subplots_adjust(top=0.9)
plt.savefig('stackedsingleper1.png.png', dpi=300, bbox_inches='tight')

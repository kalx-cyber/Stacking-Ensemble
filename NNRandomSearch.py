# -*- coding: utf-8 -*-
"""
Created on Fri Oct 15 22:25:47 2021

@author: Ramanzani S. Kalule
email: kramanzani@gmail.com
"""

import os
import keras
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from keras.optimizers import SGD
from sklearn.utils import shuffle
from sklearn import preprocessing
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from sklearn.model_selection import RandomizedSearchCV
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error

# fix random seed for reproducibility
seed = 123
np.random.seed(seed)
os.getcwd()

def load_data():
    df = pd.read_csv("PoreProps2.csv")
    df = df.drop("FileName",1)
    print(df.info())
    print(df.head(5))
    
    X = df.drop('Permeability', axis=1)
    X = X.drop('Porosity', axis=1)
    X = X.drop('MeanIntensity', axis=1)
    X = X.drop('Area', axis=1)
    
    '''  min-max scaling ''' 
    min_max_scaler = preprocessing.MinMaxScaler()
    X = min_max_scaler.fit_transform(X)
    X = pd.DataFrame(X)


    '''  standardisation ''' 
    #X = preprocessing.scale(X)

 
    y = df[['Porosity','Permeability']]

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
    X, y = shuffle(X, y, random_state=0)

    return X, y #np.log(y)


X, y = load_data()

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler((0,1))
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

y_train = sc.fit_transform(y_train)
y_test = sc.transform(y_test)

#What hyperparameter we want to play with
parameters = {'batch_size': [16, 32, 64, 128],
              'epochs': [30, 50, 80, 100, 120, 150],
              'optimizer': ['SGD', 'RMSprop', 'Adagrad', 'Adadelta', 'Adam', 'Adamax', 'Nadam']}
# Model 1

def buildModel1(optimizer):
    # Initialising the ANN
    regressor1 = Sequential()
    # Adding the input layer and the first hidden layer
    regressor1.add(Dense(units = 128, activation = 'relu'))
    # Adding the output layer
    regressor1.add(Dense(units = 2, activation = 'sigmoid'))
    # Compiling the ANN
    regressor1.compile(loss='mean_squared_error', optimizer=optimizer, metrics=['mean_absolute_error'])
    return regressor1

# Model 2

def buildModel2(optimizer):
    # Initialising the ANN
    regressor2 = Sequential()
    # Adding the input layer and the first hidden layer
    regressor2.add(Dense(units = 128, activation = 'relu'))
    # Adding the second hidden layer
    regressor2.add(Dense(units = 64, activation = 'relu'))
    # Adding the third hidden layer
    regressor2.add(Dense(units = 32, activation = 'relu'))
    # Adding the output layer
    regressor2.add(Dense(units = 2, activation = 'sigmoid'))
    # Compiling the ANN
    regressor2.compile(loss='mean_squared_error', optimizer=optimizer, metrics=['mean_absolute_error'])
    return regressor2

# Model 3

def buildModel3(optimizer):
    # Initialising the ANN
    regressor3 = Sequential()
    # Adding the input layer and the first hidden layer
    regressor3.add(Dense(units = 224, activation = 'relu'))
    # Adding the second hidden layer
    regressor3.add(Dense(units = 128, activation = 'relu'))
    # Adding the third hidden layer
    regressor3.add(Dense(units = 64, activation = 'relu'))
    # Adding the output layer
    regressor3.add(Dense(units = 2, activation = 'sigmoid'))
    # Compiling the ANN
    regressor3.compile(loss='mean_squared_error', optimizer=optimizer, metrics=['mean_absolute_error'])
    return regressor3

# Model 4

def buildModel4(optimizer):
    # Initialising the ANN
    regressor4 = Sequential()
    # Adding the input layer and the first hidden layer
    regressor4.add(Dense(units = 128, activation = 'relu'))
    # Adding the drop_out layer
    regressor4.add(Dropout(0.1))
    # Adding the third hidden layer
    regressor4.add(Dense(units = 64, activation = 'relu'))
    # Adding the drop_out layer
    regressor4.add(Dropout(0.1))
    # Adding the third hidden layer
    regressor4.add(Dense(units = 32, activation = 'relu'))
    # Adding the output layer
    regressor4.add(Dense(units = 2, activation = 'sigmoid'))
    # Compiling the ANN
    regressor4.compile(loss='mean_squared_error', optimizer=optimizer, metrics=['mean_absolute_error'])
    return regressor4

# Model 5

def buildModel5(optimizer):
    # Initialising the ANN
    regressor5 = Sequential()
    # Adding the input layer and the first hidden layer
    regressor5.add(Dense(units = 224, activation = 'relu'))
    # Adding the drop_out layer
    regressor5.add(Dropout(0.2))
    # Adding the second hidden layer
    regressor5.add(Dense(units = 128, activation = 'relu'))
    # Adding the drop_out layer
    regressor5.add(Dropout(0.2))
    # Adding the third hidden layer
    regressor5.add(Dense(units = 64, activation = 'relu'))
    # Adding the output layer
    regressor5.add(Dense(units = 2, activation = 'sigmoid'))
    # Compiling the ANN
    regressor5.compile(loss='mean_squared_error', optimizer=optimizer, metrics=['mean_absolute_error'])
    return regressor5


regressor1 = KerasRegressor(build_fn = buildModel1)
grid_search1 = RandomizedSearchCV(estimator = regressor1, param_distributions = parameters, scoring = 'r2', cv = 5)
grid_search1 = grid_search1.fit(X_train, y_train, verbose = 0)
best_parameters1 = grid_search1.best_params_
best_score1 = grid_search1.best_score_
print("Model 1 Best Parameters: " + str(best_parameters1))
print("Model 1 Best Score: " + str(best_score1))

regressor2 = KerasRegressor(build_fn = buildModel2)
grid_search2 = RandomizedSearchCV(estimator = regressor2, param_distributions = parameters, scoring = 'r2', cv = 7, n_jobs= -1)
grid_search2 = grid_search2.fit(X_train, y_train, verbose = 0)
best_parameters2 = grid_search2.best_params_
best_score2 = grid_search2.best_score_
print("Model 2 Best Parameters: " + str(best_parameters2))
print("Model 2 Best Score: " + str(best_score2))

regressor3 = KerasRegressor(build_fn = buildModel3)
grid_search3 = RandomizedSearchCV(estimator = regressor3, param_distributions = parameters, scoring = 'r2', cv = 7, n_jobs= -1)
grid_search3 = grid_search3.fit(X_train, y_train, verbose = 0)
best_parameters3 = grid_search3.best_params_
best_score3 = grid_search3.best_score_
print("Model 3 Best Parameters: " + str(best_parameters3))
print("Model 3 Best Score: " + str(best_score3))

regressor4 = KerasRegressor(build_fn = buildModel4)
grid_search4 = RandomizedSearchCV(estimator = regressor4, param_distributions = parameters, scoring = 'r2', cv = 7, n_jobs= -1)
grid_search4 = grid_search4.fit(X_train, y_train, verbose = 0)
best_parameters4 = grid_search4.best_params_
best_score4 = grid_search4.best_score_
print("Model 4 Best Parameters: " + str(best_parameters4))
print("Model 4 Best Score: " + str(best_score4))

regressor5 = KerasRegressor(build_fn = buildModel5)
grid_search5 = RandomizedSearchCV(estimator = regressor5, param_distributions = parameters, scoring = 'r2', cv = 7, n_jobs= -1)
grid_search5 = grid_search5.fit(X_train, y_train, verbose = 0)
best_parameters5 = grid_search5.best_params_
best_score5 = grid_search5.best_score_
print("Model 5 Best Parameters: " + str(best_parameters5))
print("Model 5 Best Score: " + str(best_score5))
import pandas as pd
import numpy as np
import random
import pylab as pl
import matplotlib.pyplot as plt
from scipy import optimize
import time
from sklearn import preprocessing, model_selection, svm, metrics, tree
from sklearn.metrics import *
from sklearn.preprocessing import StandardScaler, Imputer
from sklearn.model_selection import train_test_split, KFold, GridSearchCV

from sklearn.ensemble import AdaBoostRegressor, BaggingRegressor,
    ExtraTreesRegressor, RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import ElasticNetCV, HuberRegressor, BayesianRidge,
    LassoLarsCV, LassoCV, RidgeCV, SGDRegressor
from sklearn.svm import SVR, LinearSVR

class Timer(object):
    def __init__(self, name=None):
        self.name = name

    def __enter__(self):
        self.tstart = time.time()
        return self

    def time_check(self):
        return time.time() - self.tstart

    def __exit__(self, type, value, traceback):
        if self.name:
            print("{}:".format(self.name), end = ' ')
print('{:.2} seconds elapsed'.format(time.time() - self.tstart))

def define_model_params():
    mods = {
        'lr' = linear_model.LinearRegression(),
        'rf' = RandomForestRegressor(n_estimators=100, n_jobs = -1)
    }

    params = {

    }
    return mods, params

def model_loop(models_to_run, mods, params, X_train, X_test, y_train, y_test,
    models_to_run, criterion = 'mean_squared_error', cv_folds = 10):
    """
    Returns a dictionary where the keys are model nicknames (strings)
    and the values are regressors with methods predict and fit

    :param dict(str:estimator) mods: models as returned by define_model_params
    :param dict(str:dict) params: grid of regressor hyperparameter options
        to grid search over as returned by define_model_params
    :param pandas.DataFrame X_train: training features for model
    :param pandas.DataFrame X_test: data to predict with same features as
        X_train
    :param pandas.Series y_train: Target variable for X_train
    :param pandas.Series y_test: Target variable for X_test
    :param list[string] models_to_run: which models to actually run
        (e.g. ['ridge', 'RF'])
    :param string criterion: evaluation criterion for model selection on the
        validation set, (e.g. 'mean_squared_error')
    """
    model_grid_results = {}
    with Timer('model comparison loop') as qq:
        for index, model in enumerate([mods[x] for x in models_to_run]):
            model_name = models_to_run[index]
            parameter_values = params[model_name]
            with Timer(model_name) as t:
                estimators = GridSearchCV(model, parameter_values,
                    scoring = criterion, n_jobs = -1, cv = cv_folds)
                estimators.fit(X_train, y_train)
                print("Best estimator found by grid search:")
                print(estimators.best_estimator_)
                print("Best parameters set found on development set:")
                hyperparams = estimators.best_params_
                print(hyperparams)

                print("Cross validation score on development set:")
                cv_score = estimators.cv_results_['mean_test_score']
                print(cv_score)

                print("Test set score using best hyperparameters:")
                test_score = estimators.score(X_test, y_test)
                print(test_score)

                y_pred = estimators.predict(X_test)
                model_grid_results[model_name] = {
                    'cv_score': cv_score,
                    'test_score': test_score,
                    'hyperparams': hyperparams,
                    'test_features': X_test,
                    'predictions': y_pred
                }
    return model_grid_results

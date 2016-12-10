import pandas as pd
import numpy as np
import random
import pylab as pl
import matplotlib.pyplot as plt
from scipy import optimize
import time
from sklearn import preprocessing, model_selection, svm, metrics, tree
from sklearn.metrics import *
from sklearn.model_selection import KFold, GridSearchCV, RandomizedSearchCV

from sklearn.ensemble import AdaBoostRegressor, BaggingRegressor, \
    ExtraTreesRegressor, RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import ElasticNet, HuberRegressor, \
    BayesianRidge, LassoLars, Lasso, Ridge, SGDRegressor, LinearRegression
from sklearn.svm import SVR, LinearSVR

from argparse import ArgumentParser
import warnings
import final_modeling as fm
import final_data_clean as dc
import predict_price_increase as ppi
from custom_scorers import build_tuple_scorer, parse_criterion_string

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
        'lr': LinearRegression(),
        'rf': RandomForestRegressor(n_estimators=50),
        'ada': AdaBoostRegressor(n_estimators=50),
        'bag': BaggingRegressor(n_estimators=50),
        'et': ExtraTreesRegressor(n_estimators=50),
        'gb': GradientBoostingRegressor(n_estimators=50),
        'en': ElasticNet(),
        'hr': HuberRegressor(),
        'br': BayesianRidge(n_iter=100),
        'll': LassoLars(),
        'lasso': Lasso(),
        'ridge': Ridge(),
        'sgd': SGDRegressor(),
        'svr': SVR(),
        'linsvr': LinearSVR()
    }

    params = {
        'rf' : {
            "max_depth": [1, 3, 5, 10, 20, 50, 100],
            "max_features": [0.3, 0.4, 0.6, 0.8],
            "min_samples_split": [1, 3, 10, 20],
            "min_samples_leaf": [1, 10, 20, 30],
            "bootstrap": ["True", "False"]},
        'ada' : {
            "learning_rate": [0.1, 0.5, 1.0, 1.5],
            "loss" : ["linear", "square", "exponential"]},
        'bag' : {
            "max_features": [0.3, 0.5, 0.8, 1.0],
            "max_samples": [0.3, 0.5, 1.0],
            "bootstrap_features": ["True", "False"]},
        'et' : {
            "max_features": [0.5, 0.8, "auto", "sqrt"],
            "max_depth": [1, 3, 5, 10, 20, 50, 100],
            "min_samples_leaf": [1, 10, 20, 30],
            "min_samples_split": [1, 3, 10, 20]},
        'gb' : {
            "loss": ["ls", "lad", "huber"],
            "learning_rate": [0.1, 0.2, 0.5, 1.0],
            "max_depth": [1, 3, 5, 10, 20, 50, 100],
            "max_features": [0.3, 0.4, 0.8],
            "min_samples_split": [1, 3, 10, 20],
            "min_samples_leaf": [1, 10, 20, 30]},
        'en' : {
            "l1_ratio": [0.5, 0.7, 0.9, 1.0],
            "alpha": [0.2, 0.5, 0.8, 1.0, 2.0]},
        'hr' : {
            "epsilon": [1.35, 1.6, 2.0, 2.5],
            "alpha" : [0.0001, 0.001, 0.005]},
        'br' : {
            "alpha_1": [1e-06, 1e-05],
            "alpha_2": [1e-06, 1e-05],
            "lambda_1": [1e-06, 1e-05],
            "lambda_2": [1e-06, 1e-05]},
        'lasso' : {
            "alpha": np.logspace(-6, 3, 15),
            "selection": ["random", "cyclic"]},

        'ridge' : {
            "alpha": np.logspace(-6, 3, 15)},
        'sgd' : {
            "loss": ["squared_loss", "huber", "epsilon_insensitive"],
            "penalty": ["none", "12", "l1", "elasticnet"],
            "learning_rate": ["optimal"]},
        'svr' : {
            "C": [0.0001, 0.001, 0.01, 0.1, 1, 10],
            "kernel": ["rbf", "poly", "sigmoid"],
            "epsilon": [0.05, 0.1, 0.2, 0.5]},
        'linsvr' : {
            "C": [0.0001, 0.001, 0.01, 0.1, 1, 10],
            "epsilon": [0.05, 0.1, 0.2, 0.5]},
        'll' : {},
        'lr' : {}
    }
    return mods, params

def model_loop(models_to_run, mods, params, X_train, X_test, y_train, y_test,
        criterion_list = ['median_absolute_err', 'mean_absolute_err',
            'accuracy_5', 'accuracy_10'], cv_folds = 5, max_per_grid = 50,
        output_dir = 'data/results'):
    """
    Returns a dictionary where the keys are model nicknames (strings)
    and the values are regressors with methods predict and fit

    Args:
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
    Returns:
        dictionary + model
    """
    model_grid_results = {}
    tuple_score = build_tuple_scorer(criterion_list)
    cv_scorer = parse_criterion_string(criterion_list[0])

    with Timer('model comparison loop') as qq:
        for index, model in enumerate([mods[x] for x in models_to_run]):
            model_name = models_to_run[index]
            parameter_values = params[model_name]
            param_size = [len(a) for a in parameter_values.values()]
            param_size = min(np.prod(param_size), max_per_grid)
            with Timer(model_name) as t:
                estimators = RandomizedSearchCV(model, parameter_values,
                    scoring = cv_scorer, n_jobs = -1, cv = cv_folds,
                    random_state = 300, n_iter = param_size)
                estimators.fit(X_train, y_train)
                print("Best estimator found by grid search:")
                print(estimators.best_estimator_)
                print("Best parameters set found on development set:")
                print(estimators.best_params_)

                print("Cross validation score on development set:")
                cv_score = np.abs(estimators.best_score_)
                print(cv_score)

                print("Test set score using best hyperparameters:")
                test_score = np.abs(list(tuple_score(
                    estimators.best_estimator_, X_test, y_test)))
                print(test_score)

                y_pred = estimators.predict(X_test)
                model_grid_results[model_name] = {
                    'cv_score': cv_score,
                    'test_score': test_score,
                    'hyperparams': estimators.best_params_,
                    'predictions': y_pred,
                    'model': estimators.best_estimator_
                }
    print("Models fitted: {}".format(model_grid_results.keys()))
    return model_grid_results


def get_best_model(model_grid_results):
    best_model_name, best_cv_score = "", np.Inf
    for model_name, model in model_grid_results.items():
        pass
    return None



def apply_model_to_lightrail(data, model, model_name, output_dir,
        bbl_path = "data/subway_bbls/Queens Light Rail BBL.csv"):
    # Apply fitted model to affected properties near the Queens Light Rail
    affected_properties, data = dc.extract_affected_properties(
        data, bbl_path)

    X_updated, X_updated_for_modeling, y_orig = ppi.prepare_data(
        affected_properties)

    ppi.make_prediction(X_updated, X_updated_for_modeling, y_orig, model,
        "{}/price_increase_{}.csv".format(output_dir, model_name))

def main():
    warnings.filterwarnings("ignore")

    # Set up input option parsing for model type and data path
    parser = ArgumentParser(description =
        "Run a cross-validated grid search over a model or list of models")
    parser.add_argument("--model", dest = "model_type", nargs="*",
        help = "Defines the type of model to be built. Not case sensitive")
    parser.add_argument("--data", dest = "data_path",
        help = "Path to csv file on which you want to fit a model.")
    parser.set_defaults(model_type = 'lr',
        data_path = "data/merged/queens_2003_2016.csv")
    args = parser.parse_args()

    # LR, ElasticNet, HuberRegressor, BayesianRidge, LassoLars, Lasso, Ridge,
    # SGD, LinearSVR taking a very long time
    model_type, data_path = args.model_type, args.data_path
    model_type = [m.lower() for m in model_type]

    print("Reading in data from %s" % data_path)
    data = fm.get_data_for_model(data_path)

    print("Creating target variable")
    X, y = fm.create_target_var(data, 'price_per_sqft')

    print("Splitting data into training and test sets")
    X_train, X_test, y_train, y_test = dc.split_data(X, y)
    print("Train: %s, Test: %s" % (X_train.shape, X_test.shape))
    print("Train y: %s, Test y: %s" % (y_train.shape, y_test.shape))

    print("Imputing missing values")
    X_train, X_test = dc.fill_na(X_train, X_test)

    print("Normalizing data")
    X_train, X_test = dc.normalize(X_train, X_test)

    print("Fitting models")
    mods, params = define_model_params()
    model_results = model_loop(model_type, mods, params,
        X_train, X_test, y_train, y_test)
    print(model_results)

    # apply_model_to_lightrail(estimators.best_estimator_,
    #    model_name, output_dir)

if __name__ == '__main__':
    main()

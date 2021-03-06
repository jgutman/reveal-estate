import pandas as pd
import numpy as np
from scipy import optimize
import os, time, pickle
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
            print(self.time_convert())

    def time_convert(self):
        total_sec = self.time_check()
        hours = '{:0.0f}'.format(np.floor(total_sec / 3600.))
        minutes = '{:0.0f}'.format(np.floor((total_sec % 3600.) / 60.))
        seconds = '{:0.2f}'.format(total_sec % 60.)
        str_time = '{} h, {} min, {} sec'.format(hours, minutes, seconds)
        return(str_time)

def define_model_params():
    mods = {
        'lr': LinearRegression(),
        'rf': RandomForestRegressor(random_state = 300),
        'ada': AdaBoostRegressor(random_state = 300),
        'bag': BaggingRegressor(random_state = 300),
        'et': ExtraTreesRegressor(random_state = 300),
        'gb': GradientBoostingRegressor(random_state = 300),
        'en': ElasticNet(random_state = 300),
        'hr': HuberRegressor(),
        'br': BayesianRidge(n_iter = 100),
        'll': LassoLars(),
        'lasso': Lasso(random_state = 300),
        'ridge': Ridge(random_state = 300),
        'sgd': SGDRegressor(random_state = 300),
        'svr': SVR(),
        'linsvr': LinearSVR(random_state = 300)
    }

    params = {
        'rf' : {
            "max_depth": [1, 3, 5, 10, 20, 50, 100],
            "max_features": [0.3, 0.4, 0.6, 0.8],
            "min_samples_split": [2, 4, 8, 10, 20],
            "min_samples_leaf": [1, 10, 20, 30],
            "bootstrap": ["True", "False"],
            "n_estimators": [10, 20, 50, 100]},
        'ada' : {
            "learning_rate": [0.1, 0.5, 1.0, 1.5],
            "loss" : ["linear", "square", "exponential"],
            "n_estimators": [10, 20, 50, 100]},
        'bag' : {
            "max_features": [0.3, 0.5, 0.8, 1.0],
            "max_samples": [0.3, 0.5, 1.0],
            "bootstrap_features": ["True", "False"],
            "n_estimators": [10, 20, 50, 100]},
        'et' : {
            "max_features": [0.5, 0.8, "auto", "sqrt"],
            "max_depth": [1, 3, 5, 10, 20, 50],
            "min_samples_leaf": [1, 10, 20],
            "min_samples_split": [2, 3, 5, 8, 10, 20],
            "n_estimators": [10, 20, 50, 100]},
        'gb' : {
            "loss": ["ls", "lad", "huber"],
            "learning_rate": [0.1, 0.2, 0.5, 1.0],
            "max_depth": [1, 3, 5, 10, 20],
            "max_features": [0.3, 0.4, 0.8],
            "min_samples_split": [2, 3, 5, 8, 10, 20],
            "min_samples_leaf": [1, 10, 20, 30],
            "n_estimators": [10, 20, 50]},
        'en' : {
            "l1_ratio": [0.2, 0.3, 0.4, 0.5, 0.7, 0.9, 1.0],
            "alpha": np.logspace(-6, 3, 15)},
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
        'll' : {"alpha": np.logspace(-6, 3, 15)},
        'lr' : {"fit_intercept": [True],
		"normalize": [True, False]}
    }
    return mods, params

def model_loop(models_to_run, mods, params, X_train, X_test, y_train, y_test,
        criterion_list = ['median_absolute_err', 'mean_absolute_err',
            'accuracy_10', 'accuracy_15'], cv_folds = 5, max_per_grid = 2,
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
                    # for more verbosity, set n_jobs = 1, verbose = 3? 10?
                    random_state = 300, n_iter = param_size, verbose = 1)
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

                y_pred_test = estimators.predict(X_test)
                y_pred_train = estimators.predict(X_train)

                model_grid_results[model_name] = {
                    'cv_score': cv_score,
                    'test_score': test_score,
                    'hyperparams': estimators.best_params_,
                    'predictions_test': y_pred_test,
                    'predictions_train': y_pred_train,
                    'model': estimators.best_estimator_
                }
    print("Models fitted: {}".format(model_grid_results.keys()))
    return model_grid_results


def get_best_model(model_grid_results):
    zipped_results = [(model_name, m['cv_score']) for model_name, m in
        model_grid_results.items()]
    zipped_results.sort(key = lambda x: x[1])
    best_model_name = zipped_results[0][0]
    best_model = model_grid_results[best_model_name]['model']
    return best_model_name, best_model


def write_dict_to_df(model_grid_results):
    model_results_string = {k: {sub_k: str(sub_v)
            for sub_k, sub_v in v.items()}
        for k, v in model_grid_results.items()}

    output = pd.DataFrame.from_dict(model_results_string).transpose()
    return output


def output_results(model_grid_results, output_dir, y_train, y_test,
    bbl_train, bbl_test):
    model_results_df = write_dict_to_df(model_grid_results)
    model_name, best_model = get_best_model(model_grid_results)
    y_pred_test = model_grid_results[model_name]['predictions_test']
    y_pred_train = model_grid_results[model_name]['predictions_train']

    model_results_df.to_csv(os.path.join(output_dir,
        'results_dict_{}.csv'.format(model_name)),
        columns = ['cv_score', 'test_score', 'hyperparams'])
    output_test = os.path.join(output_dir,
        'results_predictions_{}_{}.csv'.format(model_name, 'test'))
    output_train = os.path.join(output_dir,
            'results_predictions_{}_{}.csv'.format(model_name, 'train'))

    pd.DataFrame({'bbl': bbl_test, 'y_true': y_test,
        'y_pred': y_pred_test}).to_csv(output_test, index = False)
    pd.DataFrame({'bbl': bbl_train, 'y_true': y_train,
        'y_pred': y_pred_train}).to_csv(output_train, index = False)
    return model_name, best_model

def main():
    warnings.filterwarnings("ignore")

    # Set up input option parsing for model type and data path
    parser = ArgumentParser(description =
        "Run a cross-validated grid search over a model or list of models")
    parser.add_argument("--model", dest = "model_type", nargs="*",
        help = "Defines the type of model to be built. Not case sensitive")
    parser.add_argument("--data", dest = "data_path",
        help = "Path to csv file on which you want to fit a model.")
    parser.add_argument("--iters", dest = "max_per_grid", type = int,
        help = "Max number of grid search iterations per model type")
    parser.add_argument("--output", dest = "output_dir",
        help = "Path to directory for writing output files to")
    parser.set_defaults(model_type = ['rf'], max_per_grid = 2,
        data_path = "data/merged/queens_2003_2016.csv",
        output_dir = "data/results")
    args = parser.parse_args()

    # LR, ElasticNet, HuberRegressor, BayesianRidge, LassoLars, Lasso, Ridge,
    # SGD, LinearSVR taking a very long time
    model_type, data_path, max_per_grid, output_dir = args.model_type, \
        args.data_path, args.max_per_grid, args.output_dir
    if type(model_type) == str:
        model_type = [model_type]
    model_type = [m.lower() for m in model_type]

    print("Reading in data from %s" % data_path)
    data_with_bbl = fm.get_data_for_model(data_path)
    data = data_with_bbl.drop('bbl', axis=1)

    print("Preprocessing data")
    X_train_raw, X_train, X_test, y_train, y_test = fm.preprocess_data(data)
    train_bbl = data_with_bbl['bbl'].loc[y_train.index]
    test_bbl = data_with_bbl['bbl'].loc[y_test.index]

    print("Fitting models")
    mods, params = define_model_params()
    model_results = model_loop(model_type, mods, params,
        X_train, X_test, y_train, y_test,
        max_per_grid = max_per_grid, output_dir = output_dir)
    print(model_results)

    model_name, best_model = output_results(model_results, output_dir, y_train,
        y_test, train_bbl, test_bbl)
    path = os.path.join(output_dir, 'pkl_models')
    filename = '{}_{}.pkl'.format(model_name, max_per_grid)
    with open(os.path.join(path, filename), 'wb') as f:
        pickle.dump(best_model, f)

    ppi.apply_model_to_lightrail(data_with_bbl, X_train_raw, best_model,
        model_name, output_dir = output_dir)

if __name__ == '__main__':
    main()

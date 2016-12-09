import pandas as pd
from pyproj import Proj
import numpy as np
import math
import matplotlib.pyplot as plt
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from argparse import ArgumentParser
from final_data_clean import *
from merge_pluto_finance_new import *
from predict_price_increase import *

import seaborn as sns
sns.set(color_codes=True)
import warnings


def create_target_var(data, target_name):
    '''
    Separates X and y variables for data.
    Args:
        data: Pandas dataframe containing all data.
        target_name: column name of target variable
    Returns:
        X: Pandas dataframe with training features
        y: Target variable for X as Pandas series.
    '''
    # Convert int64 to float64
    data = data.astype(float)
    # Drop NaN for crucial columns
    data = data.replace({np.Inf:np.nan, -np.Inf:np.nan})
    data = data.dropna(how = 'any', subset = [target_name])
    # Split data into X and y
    X = data.drop(target_name, axis=1)
    y = data.loc[:,target_name]
    return X, y


def get_data_for_model(data_path = \
        'bronx_brooklyn_manhattan_queens_statenisland_2003_2016.csv'):
    df = pd.read_csv(data_path, low_memory = True)
    # drop columns that are not needed or are redundant
    df = drop_cols(df, ['sale_date', 'sale_price', 'public_recycling_bins_dist'])
    return df


def fit_LR(X_train, X_test, y_train, y_test):
    '''
    Fits Linear Regression model to Pandas dataframes (X_train, X_test, y_train, y_test).
    Args:
        X_train: Pandas dataframe with training features.
        X_test: Pandas dataframe with same features as X_train.
        y_train: Target variable for X_train.
        y_test: Target varibale for X_test.

    Returns:
        lin_reg: Linear Regression model
    '''
    regr = linear_model.LinearRegression()
    regr.fit(X_train, y_train)
    mse = mean_squared_error(y_test, regr.predict(X_test))
    predicted = regr.predict(X_test)
    percent_diff = 100*(np.abs(predicted - y_test).astype(float) / y_test)
    acc = 100 * (sum(i < 10. for i in percent_diff)/ len(percent_diff))
    print('Mean squared error for Linear Regression model: ', mse)
    print('\nAccuracy (within 10% of true value): ', acc)
    return regr


def fit_RF(X_train, X_test, y_train, y_test):
    RF_reg_final = RandomForestRegressor(n_estimators=100, n_jobs = -1)
    RF_reg_final.fit(X_train, y_train)
    predicted = RF_reg_final.predict(X_test)
    percent_diff = 100*(np.abs(predicted - y_test).astype(float) / y_test)
    acc = 100 * (sum(i < 10. for i in percent_diff)/ len(percent_diff))
    print('Mean squared error for Random Forest model: ', mean_squared_error(y_test, RF_reg_final.predict(X_test)))
    print('\nAccuracy (within 10% of true value): ', acc)
    return RF_reg_final



def main():
    warnings.filterwarnings("ignore")

    # Set up input option parsing for model type and data path
    parser = ArgumentParser(description =
        "Model type (Linear Regression LR or Random Forest RF)")
    parser.add_argument("--model", dest="model_type", type = str,
        help="Defines the type of model to be built. Acceptable options include LR (linear regression) or RF (random forest). Not case sensitive")
    parser.add_argument("--data", dest="data_path",type = str,
        help="Path to csv file on which you want to fit a model.")
    parser.set_defaults(model_type = 'lr',
        data_path = "data/merged/individual/bronx_2010_2010.csv")
    args = parser.parse_args()
    model_type, data_path = args.model_type, args.data_path
    model_type = model_type.lower()

    print("Reading in data from %s" % data_path)
    data = get_data_for_model(data_path)

    affected_properties, data = extract_affected_properties(data, "data/subway_bbls/Queens Light Rail BBL.csv")
    updated_affected_properties = affected_properties.drop(['subwaydist'], axis =1)
    updated_affected_properties = bbl_dist_to_subway(updated_affected_properties, filepath = "data/open_nyc/updatedsubwaydist.csv")



    print("Creating target variable")
    X, y = create_target_var(data, 'price_per_sqft')

    print("Splitting data into training and test sets")
    X_train, X_test, y_train, y_test = split_data(X, y)
    print("Train: %s, Test: %s" % (X_train.shape, X_test.shape))
    print("Train y: %s, Test y: %s" % (y_train.shape, y_test.shape))

    print("Imputing missing values")
    X_train, X_test = fill_na(X_train, X_test)

    print("Normalizing data")
    X_train, X_test = normalize(X_train, X_test)

    X_orig, X_updated, y_orig = prepare_data(affected_properties,updated_affected_properties)

    if model_type == 'lr':
        print("Fitting Linear Regression model")
        linear_reg = fit_LR(X_train, X_test, y_train, y_test)
        make_prediction(X_orig, X_updated, y_orig, linear_reg)
    elif model_type == 'rf':
        print("Fitting Random Forest model")
        random_forest = fit_RF(X_train, X_test, y_train, y_test)
        make_prediction(X_orig, X_updated, y_orig, random_forest)
    else:
        print("Please enter a valid model name (LR for Linear Regression or RF for Random Forest.")


if __name__ == '__main__':
    main()

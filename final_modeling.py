import pandas as pd
from pyproj import Proj
import numpy as np
import math
import matplotlib.pyplot as plt
from sklearn import datasets, linear_model
#from sklearn import model_selection
from sklearn import model_selection as cross_validation
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from argparse import ArgumentParser
from final_data_clean import *

import seaborn as sns
sns.set(color_codes=True)

import warnings


def create_target_var(data_train, data_test, target_name):
    '''
    Separates X and y variables for training and testing data.
    Args:
        data_train: Pandas dataframe containing training data.
        data_test: Pandas dataframe containing test data.
        target_name: column name of target variable
    Returns:
        X_train: Pandas dataframe with training features.
        X_test: Pandas dataframe with same features as X_train.
        y_train: Target variable for X_train.
        y_test: Target varibale for X_test.
    '''

    cols = list(data_train.columns.values) #Make a list of all of the columns in data_train
    cols.pop(cols.index(target_name))
    # Put target at the end of data_train
    data_train = data_train[cols+[target_name]]
    # Put target at the end of data_test
    data_test = data_test[cols+[target_name]]
    X_train = data_train.ix[:,:-1]
    y_train = data_train.ix[:,-1]
    X_test = data_test.ix[:,:-1]
    y_test = data_test.ix[:,-1]

    return X_train, X_test, y_train, y_test



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


#def visualize_feature_importance(model):
    #print()


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
        data_path = "data/merged/bronx_2010_2010.csv")
    args = parser.parse_args()
    model_type, data_path = args.model_type, args.data_path
    model_type = model_type.lower()
    output_dir = "data/merged"
    print("Reading in data from %s" % data_path)
    df = pd.read_csv(data_path, low_memory = True)
    # drop columns that are not needed or are redundant
    df = drop_cols(df, ['zonemap','sale_date','sale_price','year_built','latitude','longitude','bbl','bbl_x','bbl_y','public_recycling_bins_dist'])
    df = as_float(df)

    print("Splitting data into training and test sets")
    data_train, data_test = split_data(df)
    print("Cleaning data train and data test: %s, %s" % (data_train.shape, data_test.shape))
    data_train, data_test = fill_na(data_train, data_test)

    print("Creating target variable")
    X_train, X_test = normalize(data_train, data_test)
    
    print("Normalizing data")
    X_train, X_test, y_train, y_test = create_target_var(data_train, data_test, 'price_per_sqft')
    if model_type == 'lr':
        print("Fitting Linear Regression model")
        linear_reg = fit_LR(X_train, X_test, y_train, y_test)
    elif model_type == 'rf':
        print("Fitting Random Forest model")
        random_forest = fit_RF(X_train, X_test, y_train, y_test)
    else:
        print("Please enter a valid model name (LR for Linear Regression or RF for Random Forest.")


if __name__ == '__main__':
    main()

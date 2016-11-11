import pandas as pd
from pyproj import Proj
import numpy as np
import math
import matplotlib.pyplot as plt
from sklearn import datasets, linear_model
from sklearn import cross_validation
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from argparse import ArgumentParser

import seaborn as sns
sns.set(color_codes=True)


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
    data_train = data_train[cols+[target_name]] #Put target at the end of data_train
    data_test = data_test[cols+[target_name]] #Put target at the end of data_test
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
    print('Mean squared error for Linear Regression model: ', mse)
    return regr

def fit_RF(X_train, X_test, y_train, y_test):
    RF_reg_final = RandomForestRegressor(n_estimators=100, n_jobs = -1)
    RF_reg_final.fit(X_train, y_train)
    print('Mean squared error for Random Forest model: ', mean_squared_error(y_test, RF_reg_final.predict(X_test)))
    return RF_reg_final


#def visualize_feature_importance(model):
    #print()


def main():
    # Set up input option parsing for model type and data path
    parser = ArgumentParser(description =
        "Model type (Linear Regression LR or Random Forest RF)")
    parser.add_argument("--model", dest="model_type", type = str,
        help="Defines the type of model to be built. Acceptable options include LR (linear regression) or RF (random forest). Not case sensitive")
    parser.add_argument("--data", dest="data_path",type = str,
        help="Path to training and testing csv files, named data_train and data_test.")
    parser.set_defaults(model_type = 'lr',
        data_path = "data/merged/")
    args = parser.parse_args()
    model_type, data_path = args.model_type, args.data_path
    model_type = model_type.lower()
    data_train = pd.read_csv((data_path + "/data_train"))
    data_test = pd.read_csv((data_path + "/data_test"))
    X_train, X_test, y_train, y_test = create_target_var(data_train, data_test, 'price_per_sqft')
    
    if model_type == 'lr':
        linear_reg = fit_LR(X_train, X_test, y_train, y_test)
    elif model_type == 'rf':
        random_forest = fit_RF(X_train, X_test, y_train, y_test)
    else:
        print("Please enter a valid model name (LR for Linear Regression or RF for Random Forest.")
    

if __name__ == '__main__':
    main()  
    
    
    
    
    
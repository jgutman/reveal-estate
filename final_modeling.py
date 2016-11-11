import pandas as pd
from pyproj import Proj
import numpy as np
import math
import matplotlib.pyplot as plt
from sklearn import datasets, linear_model
from sklearn import cross_validation
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor

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
    cols.pop(cols.index('target_name'))
    data_train = data_train[cols+target_name] #Put target at the end of data_train
    data_test = data_test[cols+target_name] #Put target at the end of data_test
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
    print('Mean_squared_error', mse)
    return regr




def main():
    data_path = "data/merged"
    data_train = pd.read_csv((data_path + "/data_train"))
    data_test = pd.read_csv((data_path + "/data_test"))
    X_train, X_test, y_train, y_test = create_target_var(data_train, data_test, 'price_per_sqft')
    linear_reg = fit_LR(X_train, X_test, y_train, y_test)
    

if __name__ == '__main__':
    main()  
    
    
    
    
    
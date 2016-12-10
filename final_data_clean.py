import pandas as pd
import numpy as np
import math
from sklearn import model_selection
from argparse import ArgumentParser
from sklearn.preprocessing import Imputer, MinMaxScaler
import merge_pluto_finance_new as mpf

'''
Module to do final cleaning and train/test split of the data before modeling.
'''

def drop_cols(data, cols, target_var = "price_per_sqft",
        lower_limit = 20, upper_limit = 5000):
    '''
    Drop unnecessary columns from data before modeling.
    Censors data by removing rows where the target variable (price per sqft)
    is outside of the specified range.
    '''
    data = data.loc[data[target_var] >= lower_limit]
    data = data.loc[data[target_var] <= upper_limit]
    return data.drop(cols, axis = 1)


def split_data(X, y):
    '''
    Splits data into training and test sets (0.8/0.2)
        Args:
            X: Pandas dataframe
            y: Pandas Series
        Returns:
            X_train: Pandas dataframe of features used for training
            X_test: Pandas dataframe of features used for testing
            y_train: Pandas Series of labels used for training
            y_test: Pandas Series of labels used for testing
    '''
    X_train, X_test, y_train, y_test = model_selection.train_test_split(
        X, y, test_size=0.20, random_state=42)
    X_train.reset_index(drop=True, inplace=True)
    X_test.reset_index(drop=True, inplace=True)
    y_train.reset_index(drop=True, inplace=True)
    y_test.reset_index(drop=True, inplace=True)
    return X_train, X_test, y_train, y_test


def fill_na(X_train, X_test):
    '''
    Fills NaN values with the mean of the column. Note we have already created
    dummy variables for columns with missing values.

    Args:
        X_train: numpy ndarray used for training.
        X_test: numpy ndarray used for testing.
    Returns:
        X_train: numpy ndarray with no NaN values, ready for modeling.
        X_test: numpy ndarray with no NaN values, ready for testing.

    '''
    missing_imputer = Imputer(strategy = "mean", axis=0)
    X_train = missing_imputer.fit_transform(X_train)
    X_test = missing_imputer.transform(X_test)
    return X_train, X_test


def normalize(X_train, X_test):
    '''
    Transforms features by scaling each feature to (0,1) range.
    Args:
        X_train: numpy ndarray used for training.
        X_test: numpy ndarray used for testing.
    Returns:
        X_train: numpy ndarray with scaled features for modeling.
        X_test: numpy ndarray with scaled features for modeling.
    '''
    min_max_scaler = MinMaxScaler()
    X_train = min_max_scaler.fit_transform(X_train)
    X_test = min_max_scaler.transform(X_test)
    return X_train, X_test


def extract_affected_properties(df, path_to_bbls,
        path_to_subway_dist = "data/open_nyc/updatedsubwaydist.csv"):
    '''
    Extracts the properties affected by a subway line, based on bbl.
    Then alters their distance to subway according to the updated subway
    distances in path_to_subway_dist file.

    Args:
        df: Pandas dataframe with bbl column and all properties
        path_to_bbls: filepath to csv containing a list of BBLs within a 0.5 mi
            radius of proposed Queens lightrail
        path_to_subway_dist: filepath to csv containing altered distance to
            subway for affected properties (and original distance for all
            unaffected properties)
    Returns:
        affected_properties: Pandas dataframe with original dist to subway
        updated_affected_properties: Pandas dataframe with altered distance
            to subway (0.5 mi)
    '''
    bbls = pd.read_csv(path_to_bbls, header = None)
    bbls = bbls.iloc[:,0]
    affected_properties = df.loc[df['bbl'].isin(bbls)]
    affected_properties.reset_index(drop = True, inplace = True)

    updated_affected_properties = affected_properties.drop(
        'subwaydist', axis = 1)
    updated_affected_properties = mpf.bbl_dist_to_subway(
        updated_affected_properties, filepath = path_to_subway_dist)

    affected_properties.drop('bbl', axis = 1, inplace = True)
    updated_affected_properties.drop('bbl', axis = 1, inplace = True)

    return affected_properties, updated_affected_properties

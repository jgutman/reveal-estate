import pandas as pd
import numpy as np
import math
from sklearn import model_selection
from argparse import ArgumentParser
from sklearn import preprocessing

'''
Module to do final cleaning and train/test split of the data before modeling.

'''

def drop_cols(data, cols):
    data = data[data["price_per_sqft"] >= 20]
    data = data[data["price_per_sqft"] <= 5000]
    return data.drop(cols, axis = 1)



def split_data(data):
    '''
    Splits data into training and test sets (0.8/0.2)
        Args: 
            data: Pandas dataframe
        Returns:
            data_train: Pandas dataframe used for training
            data_test: Pandas dataframe used for testing
    
    '''
    #Convert 'int64' into float; otherwise, sklearn throws a warning message
    columns = data.columns.values
    non_float = []
    for col in columns:
        if data[col].dtype != np.float64:
            non_float.append(col)
    for col in non_float:
        data[col] = data[col].astype(float)
    #drop NaN for crucial columns
    data= data.dropna(how = 'any', subset = ['price_per_sqft'])   
    #Split the data
    rs = model_selection.ShuffleSplit(train_size = 0.8, test_size=.2, random_state = 1, n_splits = 1)

    for train, test in rs.split(data):
        train_index = train
        test_index = test
    data_train = data.ix[train_index,:]
    data_test = data.ix[test_index,:]
    data_train.reset_index(drop=True, inplace=True)
    data_test.reset_index(drop=True, inplace=True)
    return data_train, data_test




def fill_na(data_train, data_test):
    '''
    Fills NaN values with the mean of the column. Note we have already created dummy variables
    for columns with missing values.
    
    Args:
        data_train: Pandas dataframe used for training.
        data_test: Pandas dataframe used for testing.
    Returns:
        data_train: Pandas dataframe with no NaN values, ready for modeling.
        data_test: Pandas dataframe with no NaN values, ready for testing.
    
    '''
    data_train = data_train.apply(lambda x: x.fillna(x.mean()),axis=0)
    data_test = data_test.apply(lambda x: x.fillna(x.mean()),axis=0)
    return data_train, data_test


def normalize(data_train, data_test):
    '''Transforms features by scaling each feature to (0,1) range.'''
    min_max_scaler = preprocessing.MinMaxScaler()
    data_train = min_max_scaler.fit_transform(data_train)
    data_test = min_max_scaler.transform(data_test)
    return data_train, data_test


def as_float(data):
    for col in data.columns.values:
        data[col] = data[col].astype('float32')
    return data

    
    
    
    
    
    
    
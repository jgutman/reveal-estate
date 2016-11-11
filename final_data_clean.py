import pandas as pd
import numpy as np
import math
from sklearn import cross_validation

'''
Module to do final cleaning and train/test split of the data before modeling.

'''

def drop_cols(data, cols):
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
    data= data.dropna(how = 'any', subset = ['latitude','longitude','price_per_sqft'])   
    #Split the data
    split = cross_validation.ShuffleSplit(data.shape[0], n_iter=1, train_size = 0.7, test_size=.3, random_state = 1)

    for train, test in split:
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



def main():
    data_path = "data/merged/bronx_2010_2010.csv"
    output_dir = "data/merged"
    print("Reading in data from %s" % data_path)
    df = pd.read_csv(data_path, low_memory = True, error_bad_lines=False)
    df = drop_cols(df, ['zonemap','sale_date','sale_price'])
    data_train, data_test = split_data(df)
    print("Cleaning data train and data test")
    data_train, data_test = fill_na(data_train, data_test)
    print("Saving training data to %s/%s" % (output_dir, "data_train"))
    data_train.to_csv((output_dir + "/data_train"), index = False, chunksize=1e4)
    print("Saving test data to %s/%s" % (output_dir, "data_test"))
    data_test.to_csv((output_dir + "/data_test"), index = False, chunksize=1e4)
    
if __name__ == '__main__':
    main()    
    
    
    
    
    
    
    
    
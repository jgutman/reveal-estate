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
from final_modeling import *
from merge_pluto_finance_new import *




def prepare_data(df_orig, df_updated_subways):
    
    '''
    Prepares the dataframe of affected properties to have the model applied.
    
    Args:
        df: Pandas dataframe of affected properties
        
    Returns:
        X_orig: Original features for the affected properties (without added subway information)
        y_orig: price_per_sqft target variable for the affected properties (without added subway information)
        X_updated: Updated features for affected properties (with added subway information)
        
    '''
    print("Creating target variable")
    
    df_updated_subways = drop_cols(df_updated_subways, ['latitude', 'longitude'])
    print(df_updated_subways.shape)
    X_orig, y_orig = create_target_var(df_orig, 'price_per_sqft')
    X_updated, _ = create_target_var(df_updated_subways, 'price_per_sqft')
    
    _, X_updated = fill_na(X_updated, X_updated)
    
    print("Normalizing data")
    _, X_updated = normalize(X_updated, X_updated)

    return X_orig, X_updated, y_orig



def make_prediction(X_orig, X_updated, y_orig, model):
    '''
    Predicts price_per_sqft for the dataframe with updated subway information, and creates
    Pandas dataframe with
        
    '''
    #bbls_to_identify = X_updated['bbl']
    predicted = model.predict(X_updated.drop(['bbl'], axis=1))
    X_orig['y_pred'] = predicted
    X_orig['y_original'] = y_orig
    X_orig.to_csv('price_increase.csv')











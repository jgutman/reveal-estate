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




def prepare_data(df_updated_subways):

    '''
    Prepares the dataframe of affected properties to have the model applied.

    Args:
        df_orig: Pandas dataframe of affected properties with updated subway features

    Returns:
        X_orig: Original features for the affected properties (without added subway information)
        y_orig: price_per_sqft target variable for the affected properties (without added subway information)
        X_updated: Updated features for affected properties (with added subway information)

    '''
    #print("Creating target variable")

    df_updated_subways = drop_cols(df_updated_subways, ['latitude', 'longitude'])
    #print(df_updated_subways.shape, df_orig.shape)
    X_updated, y_orig = create_target_var(df_updated_subways, 'price_per_sqft')

    X_updated_for_modeling = X_updated.drop(['bbl'], axis = 1)
    _, X_updated_for_modeling = fill_na(X_updated_for_modeling, X_updated_for_modeling)

    #print("Normalizing data")
    _, X_updated_for_modeling = normalize(X_updated_for_modeling, X_updated_for_modeling)

    return X_updated, X_updated_for_modeling, y_orig



def make_prediction(X_updated, X_updated_for_modeling, y_orig, model,
    output = "price_increase.csv"):
    '''
    Predicts price_per_sqft for the dataframe with updated subway information,
    and creates Pandas dataframe with affected BBLs, the original predictions for price per square feet under the true features in the data, and the new predictions for price per square feet with subway distances reduced to 0.5

    '''
    #print(X_updated_for_modeling.shape)
    predicted = model.predict(X_updated_for_modeling)
    X_updated['y_pred'] = predicted
    X_updated['y_original'] = y_orig
    X_updated[['bbl','y_original','y_pred']].to_csv(output)

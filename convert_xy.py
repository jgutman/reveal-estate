from pyproj import Proj
import pandas as pd
import numpy as np
import math

'''
Module used to transform NY State X,Y Coordinates to Latitude and Longitude. Requirements: pyproj (https://pypi.python.org/pypi/pyproj)
'''


NYSP1983 = Proj(init="ESRI:102718", preserve_units=True)
def convert(x,y):
    '''
    Does the actual conversion of tuple (x,y) to (lat,long)
    
    Args:
        x : x coordinate
        y : y coordinate
    Returns:
        the latitude and longitude of the (x,y) location
    '''
    if math.isnan(x):
        return np.nan, np.nan
    elif math.isnan(y):
        return np.nan, np.nan
    else:
        lat, long = NYSP1983(x, y, inverse=True)
        return lat, long
    
def convert_df(df, col_x, col_y):
    '''
    Adds two new columns, latitude and longitude to a dataframe
    
    Args: 
    
        df = dataframe you want to alter
        col_x: name of the column that contains X coordinate
        col_y: name of the column that contains Y coordinate
        
    Returns:
    
        dataframe with new latitude and longitude columns
    '''
    
    x_coord = df[col_x]
    y_coord = df[col_y]
    lats = []
    longs = []
    for x,y in zip(x_coord,y_coord):
        longit, lat = convert(x,y)
        lats.append(lat)
        longs.append(longit)
    df['latitude'] = lats
    df['longitude'] = longs
    return df
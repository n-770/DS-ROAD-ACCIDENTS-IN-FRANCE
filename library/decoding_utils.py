
# -------------------------------------------------------------------------------------------------
# Import
# -------------------------------------------------------------------------------------------------

import pandas as pd
import numpy as np

from feature_engine.creation import CyclicalFeatures
from category_encoders import CatBoostEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import SplineTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import OrdinalEncoder

# -------------------------------------------------------------------------------------------------
# decode cyclic features: eg months, hours
# -------------------------------------------------------------------------------------------------

def decode_cyclic_feature(df, col_name, period, rangeStart=0): #period: eg 24 hours 12 months
    
    sin_col_name = f'{col_name}_sin'
    cos_col_name = f'{col_name}_cos'
    
    angle_radians = np.arctan2(df[sin_col_name], df[cos_col_name])
    decoded_col = (angle_radians / (2 * np.pi)) * period
    decoded_col = (decoded_col + period) % period
    
    df_decoded = pd.DataFrame()
    df_decoded[col_name] = np.round(decoded_col).astype(int)
    
    return df_decoded+rangeStart

# -------------------------------------------------------------------------------------------------
# decode one hot encoding
# -------------------------------------------------------------------------------------------------

def decode_one_hot(df, encoded_feature_columns, feature_name):
    
    encoded_df = df[encoded_feature_columns].copy()
    
    original_col = encoded_df.idxmax(axis=1)
    all_zeros = encoded_df.sum(axis=1) == 0
    
    prefix = f'{feature_name}_'
    missing_category = '1.0'
    
    original_col = np.where(all_zeros, f'{prefix}{missing_category}', original_col)
    original_col = pd.Series(original_col).str.replace(prefix, '')
    
    decoded_df = pd.DataFrame()
    decoded_df[feature_name] = original_col
    
    decoded_df[feature_name] = decoded_df[feature_name].astype(float).astype(int)
    
    return decoded_df

def decode_one_hot_v2(df, encoded_feature_columns, org_feature_name, encoded_prefix = 'onehot__', missing_category_ending = ''):

    encoded_df = df[encoded_feature_columns].copy()
    
    original_col = encoded_df.idxmax(axis=1)
    prefix = f'{encoded_prefix}{org_feature_name}_'
    
    if missing_category_ending:
        
        all_zeros = encoded_df.sum(axis=1) == 0
        missing_category = missing_category_ending #eg '1.0'
        original_col = np.where(all_zeros, f'{prefix}{missing_category}', original_col)
        
    original_col = pd.Series(original_col).str.replace(prefix, '')
    
    decoded_df = pd.DataFrame()
    decoded_df[org_feature_name] = original_col
    
    decoded_df[org_feature_name] = decoded_df[org_feature_name].astype(float).astype(int)
    
    return decoded_df

# -------------------------------------------------------------------------------------------------

# -------------------------------------------------------------------------------------------------
# Shap: aggregate_shap_values
# -------------------------------------------------------------------------------------------------

import pandas as pd
import numpy as np

import copy
import re

import sys
from time import sleep
from tqdm import tqdm

import shap

import decoding_utils

# -------------------------------------------------------------------------------------------------

def get_col_names(col_name, start_index, end_index, isFloat=True):

    col_names = []
    if isFloat:
        float_range = np.arange(start_index, end_index, 1.0)
        for i in float_range:
            col_names.append(f'{col_name}_{i}')
    else:
        for i in range(start_index, end_index):
            col_names.append(f'{col_name}_{i}')
    
    return col_names

# -------------------------------------------------------------------------------------------------

def aggregate_shap_values(columns_dict, shap_values):
    
    '''
    Aggregate shap_values
    
    Parameters:
    - columns_dict
    - shap_values
    
    Example of a columns dictionary which has to be passed as a parameter:
    
    acc_columns_dict = {
        'acc_year': {'length': 1, 'encoding': 'none'},
        'acc_municipality': {'length': 1, 'encoding': 'none'},
        'acc_hour': {'length': 2, 'encoding': 'hour'},
        'acc_ambient_lightning': {'length': 4, 'encoding': 'ohe'},
        'acc_atmosphere': {'length': 8, 'encoding': 'ohe'},
        'acc_urbanization_level': {'length': 1, 'encoding': 'ohe_one'},
        'acc_intersection': {'length': 8, 'encoding': 'ohe'},
        'acc_collision_type': {'length': 6, 'encoding': 'ohe'},
    }
    
    Example of function call:
    shap_values_r, df_shap_data, df_new_shap_values, org_columns, col_bounderies = aggregate_shap_values(acc_columns_dict, shap_values)
    '''
    
    #get bounderies of encoded columns
    org_columns = []
    encodings = []
    encoded_lengths = []
    
    for col_id, col_details in columns_dict.items():
        
        org_columns.append(col_id)
        
        for key, value in col_details.items():
        
            if key == 'encoding':
                encodings.append(value)
            elif key == 'length': 
                encoded_lengths.append(value)
    
    col_bounderies = np.cumsum(encoded_lengths)
    
    
    #create aggregated shap values
    new_shap_values = []
    for values in shap_values.values:
        
        #split shap values into a list for each feature
        values_split = np.split(values , col_bounderies)
        values_split.pop() #remove empty array at end
        #sum values within each list
        values_sum = [sum(l) for l in values_split]
        new_shap_values.append(values_sum)
    
    #create df with aggregated shap values
    df_new_shap_values = pd.DataFrame(new_shap_values, columns=org_columns)
    
    #create stacked aggregated shap values array
    new_shap_values_stacked_rows = [np.stack([df_new_shap_values.iloc[i][col] for col in df_new_shap_values.columns]) for i in range(len(df_new_shap_values))]
    new_shap_values_stacked_array = np.array(new_shap_values_stacked_rows)
    
    
    #create decoded shap_values.data df
    df_shap_data = pd.DataFrame()
    
    b=0
    for i, c, e, col in zip(col_bounderies, encoded_lengths, encodings, org_columns):
        
        #print(b, i, c, e, col)
        shap_data = shap_values.data[:,b:i]
        
        if e == 'ohe':
            cols = get_col_names(col, 2.0, round(float(c+2), 1))
            df_r = pd.DataFrame(data=shap_data, columns=cols)
            df_shap_data[col] = decoding_utils.decode_one_hot(df_r, cols, col)
            
        elif e == 'ohe_one':
            df_r = pd.DataFrame(data=shap_data, columns=[col])
            df_r+=1
            df_shap_data[col] = df_r.astype(int)
            
        elif e == 'hour':
            df_hour = pd.DataFrame(data=shap_data, columns=['acc_hour_sin', 'acc_hour_cos'])
            df_shap_data['acc_hour'] = decoding_utils.decode_cyclic_feature(df_hour, 'acc_hour', 24)
            
        else:
            df_shap_data[col] = pd.DataFrame(data=shap_data, columns=[col])
        b=i
    
    
    #create aggregated shap_values object
    shap_values_r = copy.deepcopy(shap_values)
    
    shap_values_r.values = new_shap_values_stacked_array
    shap_values_r.data = np.array(np.array(df_shap_data))
    shap_values_r.feature_names = list(df_shap_data.columns)
    
    
    return shap_values_r, df_shap_data, df_new_shap_values, org_columns, col_bounderies
    
# -------------------------------------------------------------------------------------------------
# Shap aggregation (v2)
# -------------------------------------------------------------------------------------------------

def create_features_dict(feature_names):
    
    # ---------------------------------------------------------------------------------------------
    # create encoded features dictionary 
    
    # dictionary entries have the form:
    # 'acc_hour': {'prefix' : 'cyclical', 'encoding': 'hour', 'length': 2, 'endings': ['sin','cos']}
    # ---------------------------------------------------------------------------------------------
    
    feature_cols = np.append(feature_names, 'prefix__none')
    
    col_dict = {}
    rows = []
    
    new_col = True
    
    col_org_name = ''
    col_endings = []
    row = []
    
    for col in feature_cols:
        
        # get column attributes
        result = col.split('__')
        result_ = []
        if ('_sin' in result[1]):
            result_ = [result[1].removesuffix('_sin'), 'sin']
        elif ('_cos' in result[1]):
            result_ = [result[1].removesuffix('_cos'), 'cos']
        elif ('_-1' in result[1]):
            result_ = [result[1].removesuffix('_-1'), '-1']
        else:
            # split at position before first digit after _
            result_ = re.split(r'_(?=\d)', result[1], maxsplit=1)
            result_[0] = result_[0].removesuffix('_')
        
        #check for new column name
        if result_[0] != col_org_name:
            new_col = True
        else:
            new_col = False
        
        #add dict entry if feature has a new name
        if new_col and row and col_org_name != '':
        
            row_enc = ''
            if row[0] == 'cyclical' and '_hour' in row[1]:
                row_enc = 'hour'
            elif row[0] == 'cyclical' and '_month' in row[1]:
                row_enc = 'month'
            elif row[0] == 'onehot':
                row_enc = 'ohe'
            else:
                row_enc = 'none'
            
            col_length = len(col_endings)
            
            if col_length == 1:
                col_endings = []
            
            dict_entry = {row[1]:{'prefix':row[0],'encoding':row_enc,'length':col_length,'endings':col_endings}}
            col_dict.update(dict_entry)
            
            col_endings = []
        
        # break if dummy feature found
        if result_[0] == 'none':
            break;
        
        #get new row values: prefix colname ending
        row = []
        if len(result_) > 1:
            row = [result[0], result_[0], result_[1]]
        else:
            row = [result[0], result_[0], '']
        
        #set column name
        col_org_name = row[1]
        
        #add column ending
        col_endings.append(row[2])
        
        #append row
        rows.append(row)
    
    return col_dict, rows

# -------------------------------------------------------------------------------------------------

def get_col_attributes(columns_dict):
    
    org_columns = []
    encodings = []
    encoded_lengths = []
    prefixes = []
    endings = []
    
    for col_id, col_details in columns_dict.items():
        
        org_columns.append(col_id)
        
        for key, value in col_details.items():
            if key == 'encoding':
                encodings.append(value)
            elif key == 'length': 
                encoded_lengths.append(value)
            elif key == 'prefix': 
                prefixes.append(value)
            elif key == 'endings':
                cat_endings = []
                for v in value:
                    cat_endings.append(v)
                endings.append(cat_endings)
    
    return org_columns, encodings, encoded_lengths, prefixes, endings

# -------------------------------------------------------------------------------------------------

def get_col_names_v2(org_col_name, prefix, endings):

    col_names = []
    col = f'{prefix}__{org_col_name}'
    
    if endings:
        for e in endings:
            col_e = col + f'_{e}'
            col_names.append(col_e)
    else:
        col_names.append(col)
    
    return col_names

# -------------------------------------------------------------------------------------------------

def get_decoded_shap_data(shap_values, col_bounderies, encoded_lengths, encodings, org_columns, prefixes, endings):
    
    # -----------------------------------------------------------
    # create decoded shap_values.data df
    # -----------------------------------------------------------
    #
    # note: 
    # cyclic and ohe encoded features are being decoded
    #
    # features with one encoded column are not decoded
    # catbost cannot be decoded
    # to decode MinMaxScaler the scaler object must be accessible
    # -----------------------------------------------------------
    
    df_shap_data = pd.DataFrame()
    
    b=0
    for i, c, e, col, pre, end in zip(col_bounderies, encoded_lengths, encodings, org_columns, prefixes, endings):
        
        #print(b, i, c, e, col)
        shap_data = shap_values.data[:,b:i]
        
        if e == 'ohe':
            cols = get_col_names_v2(col, pre, end)
            df_r = pd.DataFrame(data=shap_data, columns=cols)
            df_shap_data[col] = decoding_utils.decode_one_hot_v2(df_r, cols, col)
            
        elif e == 'ohe_one':
            df_r = pd.DataFrame(data=shap_data, columns=[col])
            df_r+=1
            df_shap_data[col] = df_r.astype(int)
            
        elif e == 'hour':
            df_hour = pd.DataFrame(data=shap_data, columns=['cyclical__acc_hour_sin', 'cyclical__acc_hour_cos'])
            df_shap_data['acc_hour'] = decoding_utils.decode_cyclic_feature(df_hour, 'cyclical__acc_hour', 24)
            
        elif e == 'month':
            df_hour = pd.DataFrame(data=shap_data, columns=['cyclical__acc_month_sin', 'cyclical__acc_month_cos'])
            df_shap_data['acc_month'] = decoding_utils.decode_cyclic_feature(df_hour, 'cyclical__acc_month', 12)
            
        else:
            df_shap_data[col] = pd.DataFrame(data=shap_data, columns=[col])
        b=i
    
    return df_shap_data

# -------------------------------------------------------------------------------------------------

def get_agg_shap_values(shap_values, org_columns, col_bounderies):
    
    new_shap_values = []
    
    #for values in shap_values.values:
    for values in tqdm(shap_values.values, desc="get_agg_shap_values: Processing items"):
    
        #split shap values into a list for each feature
        values_split = np.split(values , col_bounderies)
        
        values_split.pop() #remove empty array at end #check
        
        #sum values within each list
        values_sum = [sum(l) for l in values_split]
        new_shap_values.append(values_sum)
    
    #create df with aggregated shap values
    df_new_shap_values = pd.DataFrame(new_shap_values, columns=org_columns)
    df_new_shap_values.shape #(9989, 21)
    
    #create stacked aggregated shap values array
    new_shap_values_stacked_rows = [np.stack([df_new_shap_values.iloc[i][col] for col in df_new_shap_values.columns]) for i in range(len(df_new_shap_values))]
    new_shap_values_stacked_array = np.array(new_shap_values_stacked_rows)
    new_shap_values_stacked_array
    
    return df_new_shap_values, new_shap_values_stacked_array, new_shap_values

# -------------------------------------------------------------------------------------------------

def get_agg_shap_values_obj(shap_values, feature_names):

    columns_dict, _ = create_features_dict(feature_names)
    
    org_columns, encodings, encoded_lengths, prefixes, endings = get_col_attributes(columns_dict)
    col_bounderies = np.cumsum(encoded_lengths)
    
    df_new_shap_data = get_decoded_shap_data(shap_values, col_bounderies, encoded_lengths, encodings, org_columns, prefixes, endings)
    
    df_new_shap_values, new_shap_values_stacked_array, new_shap_values = get_agg_shap_values(shap_values, org_columns, col_bounderies)
    
    shap_values_r = copy.deepcopy(shap_values)
    shap_values_r.values = new_shap_values_stacked_array
    shap_values_r.data = np.array(df_new_shap_data)
    shap_values_r.feature_names = list(df_new_shap_data.columns)
    
    return shap_values_r, df_new_shap_values, df_new_shap_data, columns_dict

# -------------------------------------------------------------------------------------------------
# New Shap interface aggregation
#
# author: Christian Leibold
# -------------------------------------------------------------------------------------------------

class ShapOneHotAggregator:
    """
    Aggregate SHAP values of one-hot and cyclical encoded features back to their original feature.
    
    example call:
    aggregator = ShapOneHotAggregator(shap_values_new.feature_names)
    explainer_agg, mapping = aggregator.aggregate(shap_values_new.values, shap_values_new.data)
    """
    
    def __init__(self, feature_names, prefix="onehot__", cyclical_prefix="cyclical__"):
        self.feature_names = feature_names
        self.prefix = prefix
        self.cyclical_prefix = cyclical_prefix
        self.groups = self._build_groups()
        
    def _build_groups(self):
        groups = {}
        for fname in self.feature_names:
            if fname.startswith(self.prefix):
                # One-hot encoded
                base = fname[len(self.prefix):]
                parent = base.rsplit("_", 1)[0]
                groups.setdefault(parent, []).append(fname)
            elif fname.startswith(self.cyclical_prefix):
                # Cyclical encoded (_sin/_cos)
                parent = fname.rsplit("_", 1)[0]  # remove _sin/_cos
                groups.setdefault(parent, []).append(fname)
            else:
                # Continuous or already atomic
                groups.setdefault(fname, [fname])
        return groups

    def aggregate(self, shap_values, data, base_values=None):
        has_classes = shap_values.ndim == 3
        agg_list, agg_feature_names = [], []

        for parent, children in self.groups.items():
            idxs = [self.feature_names.index(ch) for ch in children]
            if has_classes:
                agg_vals = shap_values[:, idxs, :].sum(axis=1)
            else:
                agg_vals = shap_values[:, idxs].sum(axis=1)
            agg_list.append(agg_vals)
            agg_feature_names.append(parent)

        agg_values = np.stack(agg_list, axis=1) if has_classes else np.column_stack(agg_list)

        # Collapse data
        X_df = pd.DataFrame(data, columns=self.feature_names)
        agg_data = pd.DataFrame(index=X_df.index)
        for parent, children in self.groups.items():
            if parent.startswith("cyclical__") and len(children) == 2:
                # Keep both sin/cos values but collapse into one parent vector
                agg_data[parent] = np.sqrt(X_df[children[0]]**2 + X_df[children[1]]**2)
            elif len(children) > 1 and children[0].startswith(self.prefix):
                # One-hot collapse
                agg_data[parent] = X_df[children].idxmax(axis=1).str.replace(f"{parent}_", "")
            else:
                agg_data[parent] = X_df[children[0]]

        explainer_agg = shap.Explanation(
            values=agg_values,
            base_values=base_values,
            data=agg_data.values,
            feature_names=agg_feature_names
        )
        return explainer_agg, self.groups

# -------------------------------------------------------------------------------------------------

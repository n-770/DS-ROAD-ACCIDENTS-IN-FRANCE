# ----------------------------------------------
# import modules
# ----------------------------------------------

# classic packages
import pandas as pd
import numpy as np

# io, load, dump
import io
import joblib
from joblib import dump, load
from pathlib import Path

# visualization
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objs as go
import altair as alt
from PIL import Image

# stats
from scipy.stats import chi2_contingency

# custom libraries
import sys
sys.path.append('../../library')

#import shap_values_aggr
#from cleaning_utils import distinguish_cols, print_col_categories
import streamlit_functions_explore as st_funct

# helper
import time
import random

# warnings
import warnings
warnings.filterwarnings("ignore")

# streamlit
import streamlit as st

# ----------------------------------------------
# load objects
# ----------------------------------------------

@st.cache_data
def load_df(df_joblib_file_name):
    df = joblib.load(df_joblib_file_name)
    return df

# ----------------------------------------------
# lookups
# ----------------------------------------------

@st.cache_data
def veh_cat_lookup():

    veh_cat_lookup_data = {
        'veh_cat': [
                0,
                1,
                2,
                3,
                7,
                10,
                13,
                14,
                15,
                16,
                17,
                20,
                21,
                30,
                31,
                32,
                33,
                34,
                35,
                36,
                37,
                38,
                39,
                40,
                41,
                42,
                43,
                50,
                60,
                80,
                99
            ],
        'veh_cat_name': [
            'Undeterminable',
            'Bicycle',
            'Moped',
            'Small car',
            'Light vehicle',
            'Light vehicle (1.5T - 3.5T)',
            'HGV (3.5T - 7.5T)',
            'HGV (> 7.5T)',
            'HGV + trailer (> 3.5T)',
            'Road tractor',
            'Road tractor + trailer ',
            'Special engine vehicle',
            'Agricultural tractor',
            'Scooter (< 50 cc)',
            'Motorcycle (50 cc - 125 cc)',
            'Scooter (50 cc - 125 cc)',
            'Motorcycle (> 125 cc)',
            'Scooter (> 125 cc)',
            'Light quad',
            'Heavy quad',
            'Bus',
            'Coach',
            'Train',
            'Tram',
            '3WD (<= 50 cc)',
            '3WD (50 cc - 125 cc)',
            '3WD (> 125 cc)',
            'EDP with engine',
            'EDP without engine',
            'E-bike',
            'Other vehicle'],
        }
    
    veh_cat_lookup_df = pd.DataFrame(veh_cat_lookup_data)
    return veh_cat_lookup_df

def create_veh_cat_df(df):

    veh_cat_code_to_name = {

        0:'Undeterminable',
        1:'Bicycle',
        2:'Moped',
        3:'Small car',
        7:'Light vehicle',
        10:'Light vehicle (1.5T - 3.5T)',
        13:'HGV (3.5T - 7.5T)',
        14:'HGV (> 7.5T)',
        15:'HGV + trailer (> 3.5T)',
        16:'Road tractor',
        17:'Road tractor + trailer ',
        20:'Special engine vehicle',
        21:'Agricultural tractor',
        30:'Scooter (< 50 cc)',
        31:'Motorcycle (50 cc - 125 cc)',
        32:'Scooter (50 cc - 125 cc)',
        33:'Motorcycle (> 125 cc)',
        34:'Scooter (> 125 cc)',
        35:'Light quad',
        36:'Heavy quad',
        37:'Bus',
        38:'Coach',
        39:'Train',
        40:'Tram',
        41:'3WD (<= 50 cc)',
        42:'3WD (50 cc - 125 cc)',
        43:'3WD (> 125 cc)',
        50:'EDP with engine',
        60:'EDP without engine',
        80:'E-bike',
        99:'Other vehicle'
    }

    veh_cat_df = df[['veh_cat', 'ind_severity', 'acc_year']]
    veh_cat_df['veh_cat_name'] = veh_cat_df['veh_cat'].map(veh_cat_code_to_name)

    return veh_cat_df

# ----------------------------------------------

@st.cache_data
def ind_secu1_lookup():

    ind_secu1_keys = [
        -1,
        0,
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9
    ]
    ind_secu1_labels = [
        'Not reported',
        'No equipment',
        'Seatbelt',
        'Helmet',
        'Child restraint',
        'Reflective vest',
        'Airbag (multi wheel vehicles)',
        'Gloves (motorcycles)',
        'Gloves + Airbag (motorcycles)',
        'Unknown',
        'Other'
    ]

    ind_secu1_lookup_data = {
        'ind_secu1': ind_secu1_keys,
        'ind_secu1_name': ind_secu1_labels
        }
    ind_secu1_lookup_df = pd.DataFrame(ind_secu1_lookup_data)
    
    return ind_secu1_lookup_df, ind_secu1_keys, ind_secu1_labels

@st.cache_data
def create_ind_secu1_df(df):

    secu_code_to_name = {
        -1:'Not reported',
        0:'No equipment',
        1:'Seatbelt',
        2:'Helmet',
        3:'Child restraint',
        4:'Reflective vest',
        5:'Airbag (multi‑wheel vehicles)',
        6:'Gloves (motorcycles)',
        7:'Gloves + Airbag (motorcycles)',
        8:'Unknown',
        9:'Other'
    }

    ind_secu_df = df[['ind_secu1', 'ind_severity', 'acc_year']]
    ind_secu_df['ind_secu1_name'] = ind_secu_df['ind_secu1'].map(secu_code_to_name)

    return ind_secu_df

# ----------------------------------------------

@st.cache_data
def acc_coll_type_lookup():

    acc_coll_type_keys = [
        1,
        2,
        3,
        4,
        5,
        6,
        7
    ]
    acc_coll_type_labels = [
        'Two vehicles frontal',
        'Two vehicles from the rear',
        'Two vehicles by the side',
        'Three vehicles and more in chain',
        'Three or more vehicles multiple collisions',
        'Other collision',
        'Without collision'
    ]

    col_labels_alt = [
        #'Not Specified', #q
        'Two Vehicles - Head-on', 
        'Two Vehicles - Rear-End',
        'Two Vehicles - Side-End',
        'Three or More Vehicles - Chain Collision',
        'Three or More Vehicles - Multiple Collisions',
        'Other Collision',
        'No Collision'
    ]
    
    acc_coll_type_lookup_data = {
        'acc_collision_type': acc_coll_type_keys,
        'acc_collision_type_name': acc_coll_type_labels
        }
    acc_coll_type_lookup_df = pd.DataFrame(acc_coll_type_lookup_data)

    return acc_coll_type_lookup_df, acc_coll_type_keys, acc_coll_type_labels

@st.cache_data
def create_acc_coll_type_df(df):

    acc_coll_type_code_to_name = {
        1:'Two vehicles frontal',
        2:'Two vehicles from the rear',
        3:'Two vehicles by the side',
        4:'Three vehicles and more in chain',
        5:'Three or more vehicles multiple collisions',
        6:'Other collision',
        7:'Without collision'
    }

    acc_coll_type_df = df[['acc_collision_type', 'ind_severity', 'acc_year']]
    acc_coll_type_df['acc_collision_type_name'] = acc_coll_type_df['acc_collision_type'].map(acc_coll_type_code_to_name)

    return acc_coll_type_df

# ----------------------------------------------

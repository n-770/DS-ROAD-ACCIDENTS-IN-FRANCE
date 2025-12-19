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

# make custom libraries importable
import sys
sys.path.append('../../library')

#import shap_values_aggr
#from cleaning_utils import distinguish_cols, print_col_categories

# helper
import time
import random

# warnings
import warnings
warnings.filterwarnings("ignore")

# streamlit
import streamlit as st

# ----------------------------------------------
# functions
# ----------------------------------------------

@st.cache_data
def load_df(df_joblib_file_name):
    df = joblib.load(df_joblib_file_name)
    return df

'''
@st.cache_data
def load_shap(shap_joblib_file_name):
    shap_values = load(shap_joblib_file_name)
    return shap_values

@st.cache_data
def get_feature_dict(shap_features):
    dict, _  = shap_values_aggr.create_features_dict(shap_features)
    return dict

@st.cache_data
def shap_plot_feature(_shap_values, feature_prefix): #dont hash shap_values

    shap_values = _shap_values
    shap_features = shap_values.feature_names

    # Filter names that start with prefix
    prefix = feature_prefix
    selected_features = [f for f in shap_features if f.startswith(prefix)]

    # Get indices of the selected features
    selected_idx = [shap_features.index(f) for f in selected_features]

    # Create df from shap_values.data
    shap_values_df = pd.DataFrame(shap_values.data, columns=shap_features)

    for i in range (4):
        # Slice SHAP values for class i and only those features
        shap_subset = shap_values.values[:, selected_idx, i]
        X_subset = shap_values_df[selected_features]
        fig = plt.figure()
        shap.summary_plot(
            shap_subset,
            features=X_subset,
            feature_names=selected_features
        )
        st.pyplot(fig)
'''
# ----------------------------------------------

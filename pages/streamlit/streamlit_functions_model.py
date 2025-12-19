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

# modelling
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
import shap

# helper
import time
import random
import json

# custom libraries
import sys
sys.path.append('../../library')

import shap_values_aggr

# warnings
import warnings
warnings.filterwarnings("ignore")

# streamlit
import streamlit as st

# ----------------------------------------------
# functions
# ----------------------------------------------

# Load Dataframe
@st.cache_data
def load_df(df_joblib_file_name):
    df = joblib.load(df_joblib_file_name)
    return df

# Load SHAP Explanation object
@st.cache_data
def load_shap_obj(shap_obj_name):
    shap_exp = joblib.load(shap_obj_name)
    return shap_exp

# Load load_metrics
@st.cache_data
def load_metrics(model_path, X_test, y_test):
    """Load a model from joblib and compute metrics on test data."""
    model = joblib.load(model_path)
    y_pred = model.predict(X_test)

    metrics = {
        "test": {
            "accuracy": accuracy_score(y_test, y_pred),
            "macro_f1": f1_score(y_test, y_pred, average="macro"),
            "per_class": classification_report(y_test, y_pred, output_dict=True),
        },
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist()
    }
    return metrics

# ----------------------------------------------

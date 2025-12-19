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

# helper
import time
import random
import json

# custom libraries
import sys
sys.path.append('../../library')

import shap_values_aggr

#streamlit
import streamlit as st

import streamlit_functions_model as st_funct

# warnings
import warnings
warnings.filterwarnings("ignore")

# ----------------------------------------------
# Page: Conclusions & Next Steps
# ----------------------------------------------

def run_page_5():
    st.title("Conclusions & Next Steps")

    st.subheader("Scientific Conclusion")
    st.markdown(
        """
        - **Multi-class models**: struggled with rare severe classes (fatalities: F1 ≈ 0.20-0.27)  
        - **PCA projections**: revealed heavy class overlap with no distinct clusters -> simple geometric separation is impossible  
        - **Binary-class reformulation**: "uninjured" vs "injured/killed" substantially improved performance:
            - Marcro F1: ~0.8
            - Per-class F1: ~0.8
            - Overall accuracy: ~0.8
        """
    )

    st.subheader("Business Conclusion")
    st.markdown(
        """
        - **Key Takeaway**: Exposure & protection factors play dominant role
        - **Identified Key Drivers**: 
            1. Municipality
            1. Safety Equipment
            1. Vehicle Category
            1. Urbanization
            1. Obstacles
        """
    )

    st.subheader("Future Work & Modeling Outlook")
    st.markdown(
        """
        - **Advanced Feature Engineering**: 
            - Deeper investigation of feature interactions & contextual dependencies
            - Exploring polynomial & non-linear transformations to highlight hidden dependencies
        - **Deeper SHAP Analysis**: 
            - Dependency plots & interaction analysis for feature relationships
            - Developing interactive dashboards for SHAP value exploration
        - **Noise Reduction**:
            - Refining imputation methods & encoding
            - Investigating dimensionality reduction to filter irrelevant / redundant signals
        """
    )
    
    # ------------------------------------------
    #spacer at page bottom
    # ------------------------------------------
    st.write( '' )
    st.write( '' )
    st.write( '' )
    st.write( '' )

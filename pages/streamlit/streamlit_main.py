# ----------------------------------------------
# Run streamlit app with:

# cd <Directory>
# streamlit run streamlit_main.py
# ----------------------------------------------

# ----------------------------------------------
# import modules
# ----------------------------------------------

# classic packages
import pandas as pd
import numpy as np

from joblib import dump, load

# visualization
import matplotlib.pyplot as plt
import seaborn as sns

# evaluation
import shap

# helper
import time
import random

# custom libraries
import sys
sys.path.append('../../library')

import shap_values_aggr

# warnings
import warnings
warnings.filterwarnings("ignore")

#streamlit
import streamlit as st

from streamlit_home import run_home_page

from streamlit_0 import run_page_0
from streamlit_1 import run_page_1
from streamlit_2 import run_page_2
from streamlit_3 import run_page_3
from streamlit_4 import run_page_4
from streamlit_5 import run_page_5

# ----------------------------------------------
# Configuration, CSS
# ----------------------------------------------

st.set_page_config(page_title="Road Accidents: Multi-Class classification project") #, layout="wide"

st.markdown("""
<style>
    .block-container {
        padding-top: 1rem; /* Adjust this value to control the space (e.g., 0rem for minimal) */
        padding-bottom: 0rem;
        padding-left: 1rem;
        padding-right: 1rem;
    
    .single-space {
        line-height: 1.5; /* Adjust this value as needed */
        margin-top: 5px; /* Removes top margin of the paragraph */
        margin-bottom: 5px; /* Removes bottom margin of the paragraph */
    }
</style>
""", unsafe_allow_html=True)

# ----------------------------------------------
# Navigation
#
# Define pages using st.Page with custom titles and icons
# ----------------------------------------------

#material icons: eg 
# analytics bar_chart code computer insights database
# arrow_forward_ios trending_up

#Home: Context & Objectives
home_page = st.Page(run_home_page, title="Home", icon=":material/home:")

#Sections
page_0 = st.Page(run_page_0, title="Dataset", icon=":material/database:")
page_1 = st.Page(run_page_1, title="Data Exploration & Visualization", icon=":material/bar_chart:")
page_2 = st.Page(run_page_2, title="Preprocessing & Feature Engineering", icon=":material/insights:")
page_3 = st.Page(run_page_3, title="Modeling & Evaluation", icon=":material/model_training:")
page_4 = st.Page(run_page_4, title="Shap-Values Analysis", icon=":material/analytics:")
page_5 = st.Page(run_page_5, title="Conclusions & Next Steps", icon=":material/trending_up:")

# Group pages into a list or a dictionary for sections
pages = [home_page, page_0, page_1, page_2, page_3, page_4, page_5]

# Create the navigation menu
pg = st.navigation(pages, position="sidebar") # Position can be "sidebar" or "hidden"

# Run the selected page
pg.run()

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
# Page: Preprocessing & Feature Engineering
# ----------------------------------------------

def run_page_2():
    
    st.title("Preprocessing & Feature Engineering")
    tab1, tab2, tab3 = st.tabs(["Feature Engineering", "Preprocessing Pipeline", "Preprocessing Steps"])

    # ------------------------------------------
    # Tab: Feature Engineering
    # ------------------------------------------
    with tab1:
        st.markdown("""
                        - **Intersection flag** (loca_is_intersection): 
                        Flags if a location is an intersection with more than one road.
                    
                        - **Count of roads** (loca_road_count): 
                        Counts the number of rows for an intersection.

                        - **Speed limit span** (loca_max_speed_dif): 
                        Difference of highest and lowest value in speed limit for an intersection.  
                    
                        - **Age Group** (ind_age_group):
                        Age ranges derived from the birthyear of an individual (0-17, 18-24, 25-44, 45-64, 65+). 
                    
                        - **Speed limit ranges**  (loca_max_speed):
                        Cutting maximum speed limits per road into speed ranges (<50, 50-90, >90).
                    
                        - **Count of lanes ranges** (loca_road_lanes):
                        Cutting count of lanes per road into ranges (1-2, 3-4, >4).
            """)
	    
		

    # ------------------------------------------
    # Tab: Preprocessing Pipeline
    # ------------------------------------------
    with tab2:
        #st.subheader("Preprocessing Pipeline")
	    st.image("art__pic_pipeline.png")
		
		
	# ------------------------------------------
    # Tab: Preprocessing Steps
    # ------------------------------------------
    with tab3:

        
        # 1 cleaning
        # ------------------------------------------
        st.write("#### 1 Data cleaning")

        st.markdown("""
                        Dropping irrelevant variables:
                        - **ID columns:** acc_num, veh_id, veh_num
                        - **Substitutes:** acc_department / acc_long/lat (-> acc_municipality), acc_date (-> acc_year/month/hour)
                        - **Missing values:** ind_action, ind_secu2, ind_secu3, ind_location 
                         
            """)
       
        # 2 preparation
        # ------------------------------------------
        st.write("#### 2 Data preparation")            
            
        st.markdown("""
                        **Reducing modalities**
                        by aggregating to higher-tier groups for categorical variables with high cardinality 
            """)
        
        # 3 imputing
        # ------------------------------------------
        st.write("#### 3 Imputing missing values")
          
        st.markdown("""
                        - **Quantitative variables:**
                        Linear Regression towards target + KNNImputer

                        - **Cateogrical variables:**
                        Imputation per category + mode
            """)
            
        # 4 scaling
        # ------------------------------------------
        st.write("#### 4 Scaling quantiative features")
       
        st.markdown("""
                        - **Normalization MinMax:**
                        loca_road_count, loca_max_speed, loca_max_speed_dif

                        - **Normalization robust:**
                        loca_road_lanes
            """)
            

        # 5 encoding
        # ------------------------------------------
        st.write("#### 5 Encoding categorical features")
       
        st.markdown("""
                        - **OneHot**
                        
                        - **Ordinal**
                        
                        - **Cyclic**
                    
                        - **Catboost/Target**
            """)

    # ------------------------------------------
    #spacer at page bottom
    # ------------------------------------------
    st.write(' ')
    st.write(' ')
    st.write(' ')
    st.write(' ')

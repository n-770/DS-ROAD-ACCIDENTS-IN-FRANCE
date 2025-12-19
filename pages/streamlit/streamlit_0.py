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

#import shap_values_aggr

#streamlit
import streamlit as st

import streamlit_functions as st_funct

# warnings
import warnings
warnings.filterwarnings("ignore")

# ----------------------------------------------
# function, attributes
# ----------------------------------------------

# raw data #q:use sample data

#df_acc_raw_filename = '../../src/streamlit/_local/df_accidents_raw.joblib'
#df_loca_raw_filename = '../../src/streamlit/_local/df_locations_raw.joblib'
#df_veh_raw_filename = '../../src/streamlit/_local/df_vehicles_raw.joblib'
#df_ind_raw_filename = '../../src/streamlit/_local/df_individuals_raw.joblib'

df_acc_raw_filename = '../../data/processed/1_exploration/1.1-leibold-data-exploration_accidents.joblib'
df_loca_raw_filename = '../../data/processed/1_exploration/1.1-munz-data-exploration_locations.joblib'
df_veh_raw_filename = '../../data/processed/1_exploration/1.1-leibold-data-exploration_vehicles.joblib'
df_ind_raw_filename = '../../data/processed/1_exploration/1.0-becker-data-exploration-raw_usagers.joblib'

df_acc_raw = pd.DataFrame()
df_loca_raw = pd.DataFrame()
df_veh_raw = pd.DataFrame()
df_ind_raw = pd.DataFrame()

def load_raw_data(use_sample):

    global df_acc_raw
    global df_loca_raw
    global df_veh_raw
    global df_ind_raw

    if (use_sample):
        df_acc_raw = st_funct.load_df(df_acc_raw_filename.replace(".joblib", "")+'_sample_1000.joblib')
        df_loca_raw = st_funct.load_df(df_loca_raw_filename.replace(".joblib", "")+'_sample_1000.joblib')
        df_veh_raw = st_funct.load_df(df_veh_raw_filename.replace(".joblib", "")+'_sample_1000.joblib')
        df_ind_raw = st_funct.load_df(df_ind_raw_filename.replace(".joblib", "")+'_sample_1000.joblib')
    else:
        df_acc_raw = st_funct.load_df(df_acc_raw_filename)
        df_loca_raw = st_funct.load_df(df_loca_raw_filename)
        df_veh_raw = st_funct.load_df(df_veh_raw_filename)
        df_ind_raw = st_funct.load_df(df_ind_raw_filename)

# ----------------------------------------------
# Page: Dataset
# ----------------------------------------------

def run_page_0():
    
    # ------------------------------------------
    # load data
    # ------------------------------------------

    # load raw data
    use_sample=True
    load_raw_data(use_sample)
    
    # load aggregated data: load from local path!
    data_joblib_file_name = "../../data/processed/2_preprocessing/1.0-leibold-data-preprocessing_aggr.gc"
    df = st_funct.load_df(data_joblib_file_name)

    # ------------------------------------------
    # page header, tabs
    # ------------------------------------------

    st.title("Dataset")
    tab1, tab2 = st.tabs(["Raw Data", "Consolidated Data"])

    # ------------------------------------------
    # Tab: Raw data
    # ------------------------------------------
    with tab1:

        sub_tab0, sub_tab1, sub_tab2, sub_tab3, sub_tab4, sub_tab5 = st.tabs(["Relationship", "Accidents", "Locations", "Vehicles", "Individuals", "Issues"])

        #FR:
        #Caracteristiques
        #Lieux
        #Vehicules
        #Usagers
        
        # ------------------------------------------
        # Relationship
        # ------------------------------------------

        with sub_tab0:

            st.write("### Relationship between source data")

            with st.container(height=500):
                st.image("art__src-data-table_joins_v2.png")

        # ------------------------------------------
        # Accidents
        # ------------------------------------------

        with sub_tab1:
            
            #sub-header with data-load checkbox
            #st.write("### Accident Characteristics (Caracteristiques)")
            col1, col2 = st.columns([5, 5])
            
            with col1: 
                st.write("### Accident Characteristics")
            
            with col2.container(key="col2_container_acc"):
            
                st.markdown("""
                    <style>
                            .st-key-col2_container_acc div.stCheckbox {float: right;} 
                    </style>
                """, unsafe_allow_html=True)

                if st.checkbox("Load full dataset", value=not use_sample, key="id_load_acc"):
                    use_sample=False
                    load_raw_data(use_sample)
                else:
                    use_sample=True
                    load_raw_data(use_sample)
            
            #df sample rows
            st.write("#### Sample rows")
            st.dataframe(df_acc_raw.head(10), use_container_width=True)
            st.write("Dataframe size:", df_acc_raw.shape)

            #df summary
            if st.checkbox("Accident Summary Information", value=False):
            
                st.write("#### Summary Information")
                summary_acc_raw = pd.DataFrame({
                    "Column": df_acc_raw.columns,
                    "Non-Null Count": df_acc_raw.notna().sum().values,
                    "Missing": df_acc_raw.isna().sum().values,
                    "Dtype": df_acc_raw.dtypes.astype(str).values
                    })
                
                st.dataframe(summary_acc_raw, use_container_width=True)
        
        # ------------------------------------------
        # Locations
        # ------------------------------------------

        with sub_tab2:

            #sub-header with data-load checkbox
            #st.write("### Locations (Lieux)")
            col1, col2 = st.columns([5, 5])
            
            with col1: 
                st.write("### Locations")
            
            with col2.container(key="col2_container_loca"):
            
                st.markdown("""
                    <style>
                            .st-key-col2_container_loca div.stCheckbox {float: right;} 
                    </style>
                """, unsafe_allow_html=True)

                if st.checkbox("Load full dataset", value=not use_sample, key="id_load_loca"):
                    use_sample=False
                    load_raw_data(use_sample)
                else:
                    use_sample=True
                    load_raw_data(use_sample)
            
            st.write("#### Sample rows")
            st.dataframe(df_loca_raw.head(10), use_container_width=True)
            st.write("Dataframe size:", df_loca_raw.shape)

            if st.checkbox("Locations Summary Information", value=False):

                st.write("#### Summary Information")
                summary_loca_raw = pd.DataFrame({
                    "Column": df_loca_raw.columns,
                    "Non-Null Count": df_loca_raw.notna().sum().values,
                    "Missing": df_loca_raw.isna().sum().values,
                    "Dtype": df_loca_raw.dtypes.astype(str).values
                    })
                
                st.dataframe(summary_loca_raw, use_container_width=True)

        # ------------------------------------------
        # Vehicles
        # ------------------------------------------

        with sub_tab3:

            #sub-header with data-load checkbox
            #st.write("### Vehicles (Vehicules)")
            col1, col2 = st.columns([5, 5])
            
            with col1: 
                st.write("### Vehicles")
            
            with col2.container(key="col2_container_veh"):
            
                st.markdown("""
                    <style>
                            .st-key-col2_container_veh div.stCheckbox {float: right;} 
                    </style>
                """, unsafe_allow_html=True)

                if st.checkbox("Load full dataset", value=not use_sample, key="id_load_veh"):
                    use_sample=False
                    load_raw_data(use_sample)
                else:
                    use_sample=True
                    load_raw_data(use_sample)
            
            st.write("#### Sample rows")
            st.dataframe(df_veh_raw.head(10), use_container_width=True)
            st.write("Dataframe size:", df_veh_raw.shape)

            if st.checkbox("Vehicles Summary Information", value=False):

                st.write("#### Summary Information")
                summary_veh_raw = pd.DataFrame({
                    "Column": df_veh_raw.columns,
                    "Non-Null Count": df_veh_raw.notna().sum().values,
                    "Missing": df_veh_raw.isna().sum().values,
                    "Dtype": df_veh_raw.dtypes.astype(str).values
                    })
                
                st.dataframe(summary_veh_raw, use_container_width=True)

        # ------------------------------------------
        # Individuals
        # ------------------------------------------

        with sub_tab4:
            
            #sub-header with data-load checkbox
            #st.write("### Individuals (Usagers)")
            col1, col2 = st.columns([5, 5])
            
            with col1: 
                st.write("### Individuals")
            
            with col2.container(key="col2_container_ind"):
            
                st.markdown("""
                    <style>
                            .st-key-col2_container_ind div.stCheckbox {float: right;} 
                    </style>
                """, unsafe_allow_html=True)

                if st.checkbox("Load full dataset", value=not use_sample, key="id_load_ind"):
                    use_sample=False
                    load_raw_data(use_sample)
                else:
                    use_sample=True
                    load_raw_data(use_sample)

            st.write("#### Sample rows")
            st.dataframe(df_ind_raw.head(10), use_container_width=True)
            st.write("Dataframe size:", df_ind_raw.shape)

            if st.checkbox("Individuals Summary Information", value=False):

                st.write("#### Summary Information")
                summary_ind_raw = pd.DataFrame({
                    "Column": df_ind_raw.columns,
                    "Non-Null Count": df_ind_raw.notna().sum().values,
                    "Missing": df_ind_raw.isna().sum().values,
                    "Dtype": df_ind_raw.dtypes.astype(str).values
                    })
                
                st.dataframe(summary_ind_raw, use_container_width=True)

        # ------------------------------------------
        # Issues
        # ------------------------------------------

        with sub_tab5:

            st.write("#### Issues occuring during exploration of source data")
            
            st.markdown("""
                        - **Changing features:**
                        In 2018, some features were re-engineered and new features were added to the data source.
                        For this reason, source data were only used from 2019 onwards.
            """)
            st.markdown("""
                        - **N:M relationship:**
                        Source data didn't contain a key between locations (lieux) and vehicles (vehicules) resulting in a N:M relationship.
                        Location data had to be aggregated to avoid row duplications.
            """)
            st.markdown("""
                        - **Missing data:**
                        Several features contained a large proportion of null values and had to be removed.
            """)
            st.markdown("""
                        - **Limitations:**
                        To avoid bias due to potentially differing conditions in overseas territories, data were limited to Metropolitan France.
            """)

    # ------------------------------------------
    # Tab: consolidated data
    # ------------------------------------------
    with tab2:
        st.subheader("Consolidated Data")

        if df.empty:
            st.info("No dataset found")

        # Build a compact summary table 
        summary = pd.DataFrame({
            "Column": df.columns,
            "Non-Null Count": df.notna().sum().values,
            "Missing": df.isna().sum().values,
            "Dtype": df.dtypes.astype(str).values
            })
        
        st.dataframe(summary, use_container_width=True)

        st.write("### Sample rows")
        st.dataframe(df.head(10), use_container_width=True)
        st.write("Dataframe size:", df.shape)

        # move text to relationship section and data exploration page #td
        #st.subheader("Key variables")
        #st.markdown("""
        #- **Target:** Accident severity (4 classes: uninjured, slightly injured, hospitalized, fatalities).
        #- **Explanatory:** Datetime (e.g. hour, month, date), accident characteristics such as weather & light conditions, vehicle type, road category, traffic circulation, speed limit, etc.
        #- **Limitations:** Missing values of some features (e.g. trip reason, GPS), extreme outliers in speed, exclusion of overseas territories.
        #""")
    
    # ------------------------------------------
    #spacer at page bottom
    # ------------------------------------------
    st.write(' ')
    st.write(' ')
    st.write(' ')
    st.write(' ')

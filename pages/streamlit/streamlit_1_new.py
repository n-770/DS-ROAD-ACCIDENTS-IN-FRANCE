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
from PIL import Image

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

import streamlit_functions_explore as st_funct

# warnings
import warnings
warnings.filterwarnings("ignore")

# ----------------------------------------------
# functions
# ----------------------------------------------

def key_to_str(dict_int_key): 
    return {str(key): value for key, value in dict_int_key.items()}

# ----------------------------------------------
# Page: Data Exploration & Visualization
# ----------------------------------------------

def run_page_1():

    # ----------------------------------------------
    # load local data-preprocessing file
    # ----------------------------------------------

    data_joblib_file_name = "_local/1.0-leibold-data-preprocessing_aggr.joblib"
    df = st_funct.load_df(data_joblib_file_name)

    df_dep_pop_filename = "../../data/processed/1_exploration/1.0-simmler-dep_pop_fr_2019.joblib"
    
    # ----------------------------------------------
    # variables
    # ----------------------------------------------

    colors_severity_multi_class = {1: 'green', 2: 'yellowgreen', 3: 'orange', 4: 'darkred'}
    plot_color = 'deepskyblue'

    year_options = ['2019', '2020', '2021', '2022', '2023', '2024']
    year_options_all = ['All', '2019', '2020', '2021', '2022', '2023', '2024']

    # ----------------------------------------------
    # title
    # ----------------------------------------------

    st.title("Data Exploration & Visualization")

    if df.empty:
        st.info("No dataset found")
    
    st.markdown("""
        <style>
        .stTabs [data-baseweb="tab-list"] {
            gap: 15px;
            flex-wrap: wrap;
        }
        .stTabs [data-baseweb="tab"] {
            height: 18px;
            white-space: pre-wrap;
            padding-bottom: 8px;
        }
        </style>
        """, unsafe_allow_html=True)
    
    tab_list = ["Target Distribution", "Time", "Location", "Weather", "Vehicle Type", 
                "Collision Type", "Security", "Takeaways"]
    
    tabs = st.tabs(tab_list)

    with tabs[0]:

        # ----------------------------------------------
        # Plot: Target distribution
        # ----------------------------------------------

        st.write("### Target distribution")

        '''
        st.write("### Distribution of severity classes")
        severity_counts = df["ind_severity"].value_counts().sort_index()
        fig = px.bar(
        x=severity_counts.index,
        y=severity_counts.values,
        labels={"x": "Severity", "y": "Count"},
        title="Severity distribution (2019-2024)")
        st.plotly_chart(fig, use_container_width=True)
        '''

        with st.container(height=550):
        
            palette_color = colors_severity_multi_class
            st_funct.severity_distribution_multi_class(df, palette_color)

    # ----------------------------------------------
    
    with tabs[1]:
        #st.write("### Time patterns")

        sub_tab1, sub_tab2, sub_tab3 = st.tabs(["Accidents 2019-2024", "by Month", "by Hour"])

        # accidents per year
        with sub_tab1:

            #st_funct.sns_countplot(df, 'acc_year', 'Number of Accidents 2019-2024')
            st_funct.feature_distribution(df, "acc_year", "Year", 
                                          "Accidents", "Number of Accidents 2019-2024")
        
        # accidents per month
        with sub_tab2:

            select_year = st.selectbox(
                "Select year:", 
                options = year_options_all,
                index = 0, key = 'selectbox_month')
            
            st_funct.feature_distribution(df, "acc_month", "Month", "Accidents", 
                                          "Accidents per Month", year=select_year)
            
            st_funct.severity_cat_barplot(df, 'acc_month', 'Month', list(range(0, 12, 1)), year=select_year)
            #st_funct.severity_cat_barplot_alt(df, 'acc_month', 'Month', year=select_year)

            st_funct.severity_cat_cross_heatmap(df, 'acc_month', 'Month', 'Month vs. Severity', 
                                            yticks=list(range(0, 12, 1)), year=select_year)

        # accidents per hour
        with sub_tab3:
        
            select_year = st.selectbox(
                "Select year:", 
                options = year_options_all,
                index = 0, key = 'selectbox_hour')
            
            st_funct.feature_distribution(df, "acc_hour", "Hour", "Accidents", 
                                          "Accidents per Hour", year=select_year)
            
            st_funct.severity_cat_barplot(df, 'acc_hour', 'Hour', list(range(0, 24, 1)), year=select_year)
            #st_funct.severity_cat_barplot_alt(df, 'acc_hour', 'Hour', year=select_year)

            st_funct.severity_cat_cross_heatmap(df, 'acc_hour', 'Hour', 'Hour vs. Severity', 
                                            yticks=list(range(0, 24, 1)), year=select_year)
        
    # ----------------------------------------------

    with tabs[2]:
        #st.write("Location Patterns");
        
        sub_loc_tab1, sub_loc_tab2 = st.tabs(["Departments", "Municipalities"])

        # departments
        with sub_loc_tab1:
        
            select_year = st.selectbox(
                "Select year:", 
                options = year_options,
                index = 0, key = 'selectbox_dep')
            
            st_funct.feature_distribution(df, "acc_department", "Department", "Accidents", 
                                          "Accidents per Department", year=select_year)
            
            #dep_labels = np.array(df['acc_department'].unique()).tolist()
            #st_funct.severity_cat_barplot(df, 'acc_department', 'Department', dep_labels, year=select_year)

            st_funct.severity_cat_barplot_alt(df, 'acc_department', 'Department', year=select_year)

            #st.write('Crosstab:')
            #st_funct.severity_cat_crosstab(df, 'acc_department', year=select_year)

            #8: Ardennes
            #39: Jura

            #todo: add pop density

            #df_dep_pop = st_funct.load_dep_pop_data(df_dep_pop_filename)
            #df_pop = st_funct.merge_dep_pop_data(df, df_dep_pop)
            #st_funct.severity_cat_crosstab(df_pop, 'acc_department', year=select_year)
        
        # municipalities
        with sub_loc_tab2:
        
            select_year = st.selectbox(
                "Select year:", 
                options = year_options,
                index = 0, key = 'selectbox_zip')
            
            st_funct.feature_distribution(df, "acc_municipality", "Municipality", "Accidents",
                                          "Accidents per Municipality", year=select_year)
            
            st_funct.severity_cat_barplot_alt(df, 'acc_municipality', 'Municipality', year=select_year)

            #st.write('Crosstab:')
            #st_funct.severity_cat_crosstab(df, 'acc_municipality', year=select_year)
    
    # ----------------------------------------------

    with tabs[3]:
        st.write("Weather & Light"); #td
    
    # ----------------------------------------------

    with tabs[4]:
        st.write("Vehicle Type"); #td
    
    # ----------------------------------------------

    with tabs[5]:
        st.write("Collision Type"); #td
    
    # ----------------------------------------------

    with tabs[6]:
        st.write("Security equipment"); #td
    
    # ----------------------------------------------

    with tabs[7]:

        st.subheader("Takeaways from data exploration")
        #st.write("### Takeaways")
        
        # Button 1: Target takeaway
        #if st.button("Show target takeaway"):
        st.markdown("""
            #### Imbalanced target distribution
            Accident severity is highly imbalanced within the modalities. \n\n
            The majority of accidents lead to no or minor injuries; severe and and fatal outcomes are rare.
            """)
        
        # Button 2: Explanatory takeaways
        #if st.button("Show key takeaways"):
        st.markdown("""
            #### Key Takeaways
            - **Time patterns:** Most accidents occur during rush hours (8 AM, 4-6 PM); night-time accidents are less frequent, but have a worse outcome.  
            - **Location patterns:** Most accidents happen in urban areas, with Paris taking the lead; severe outcomes are spread across all departments and municipalities.
            - **Weather & light:** Most accidents happen in daylight and normal weather. Rare weather conditions such as fog and storm come with a higher amount of severe outcomes.
            - **Vehicle types:** Two-wheelers and light vehicles show the highest average severity; heavy vehicles the lowest.
            - **Collision types:** Head-on and off-road crashes carry the highest severity; multi-vehicle collisions often have a less severe outcome.
            - **Security equipment:** The existence of security equipment, such as wearing a seatbelt or helm, has a positive influence on the outcome of an accident.
            """)
            #not related to severity
            #**Road & speed conditions:** Accidents cluster in 30-50 km/h zones, especially on metropolitan and county roads.
            #td
            #, with a higher frequency outside of urban centers; remote and mountainous areas tend to have higher amounts of severe outcomes.
        
    # ------------------------------------------
    #spacer at page bottom
    # ------------------------------------------
    st.write(' ')
    st.write(' ')
    st.write(' ')
    st.write(' ')

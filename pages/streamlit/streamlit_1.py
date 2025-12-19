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
import streamlit_functions_explore_lookup as st_lookup

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

    data_joblib_file_name = "../../data/processed/2_preprocessing/1.0-leibold-data-preprocessing_aggr.gc"
    df = st_funct.load_df(data_joblib_file_name)

    df_dep_pop_filename = "../../data/processed/1_exploration/1.0-simmler-dep_pop_fr_2019.joblib"
    
    # ----------------------------------------------
    # variables
    # ----------------------------------------------

    colors_severity_multi_class = {1: 'green', 2: 'yellowgreen', 3: 'orange', 4: 'darkred'}
    plot_color = 'deepskyblue'

    year_options = ['2019', '2020', '2021', '2022', '2023', '2024']
    year_options_all = ['All', '2019', '2020', '2021', '2022', '2023', '2024']

    #select_year = 2019

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
    
    tab_list = ["Target Distribution", "Temporal Distribution", "Local Distribution", "Collision Type", #"Weather", 
                "Vehicle Type", "Security", "Takeaways"]
    
    tabs = st.tabs(tab_list)

    with tabs[0]:

        # ----------------------------------------------
        # Plot: Target distribution
        # ----------------------------------------------

        st.write("### Target distribution")

        with st.container(height=550):
        
            palette_color = colors_severity_multi_class
            st_funct.severity_distribution_multi_class(df, palette_color)

    # ----------------------------------------------
    
    with tabs[1]:
        #st.write("Temporal Distribution");

        sub_tab1, sub_tab2, sub_tab3 = st.tabs(["Accidents 2019-2024", "by Month", "by Hour"])

        # accidents per year
        with sub_tab1:

            #st_funct.sns_countplot(df, 'acc_year', 'Number of Accidents 2019-2024')
            st_funct.feature_distribution(df, "acc_year", "Year", 
                                          "Accidents", "Number of Accidents 2019-2024")
        
        # accidents per month
        with sub_tab2:

            select_year_month = st.selectbox(
                "Select year:", 
                options = year_options_all,
                index = 0, key = 'selectbox_month')
            
            st_funct.feature_distribution(df, "acc_month", "Month", "Accidents", 
                                          "Accidents per Month", year=select_year_month)
            
            st_funct.severity_cat_barplot(df, 'acc_month', 'Month', list(range(0, 12, 1)), year=select_year_month)
            #st_funct.severity_cat_barplot_alt(df, 'acc_month', 'Month', year=select_year_month)

            #st_funct.severity_cat_cross_heatmap(df, 'acc_month', 'Month', 'Month vs. Severity', yticks=list(range(0, 12, 1)), year=select_year_month)

        # accidents per hour
        with sub_tab3:
        
            select_year_hour = st.selectbox(
                "Select year:", 
                options = year_options_all,
                index = 0, key = 'selectbox_hour')
            
            st_funct.feature_distribution(df, "acc_hour", "Hour", "Accidents", 
                                          "Accidents per Hour", year=select_year_hour)
            
            st_funct.severity_cat_barplot(df, 'acc_hour', 'Hour', list(range(0, 24, 1)), year=select_year_hour)
            #st_funct.severity_cat_barplot_alt(df, 'acc_hour', 'Hour', year=select_year_hour)

            #st_funct.severity_cat_cross_heatmap(df, 'acc_hour', 'Hour', 'Hour vs. Severity', yticks=list(range(0, 24, 1)), year=select_year_hour)
        
    # ----------------------------------------------

    with tabs[2]:
        #st.write("Local Distribution");

        sub1_loc_tab_options = ["Departments", "Distribution"]
        sub1_loc_tabs = st.tabs(sub1_loc_tab_options)

        with sub1_loc_tabs[0]:
            #st.write("##### Departments")

            select_year_dep = 2019

            sub2_loc_tab_options = ["Distribution", "Crosstab", "Urban vs. Rural", "Index"]
            sub2_loc_tabs = st.tabs(sub2_loc_tab_options)

            with sub2_loc_tabs[0]:
                #st.write('Distribution:')

                select_year_dep = st.selectbox(
                "Select year:", 
                options = year_options,
                index = 0, key = 'selectbox_dep_')

                st_funct.feature_distribution(df, "acc_department", "Department", "Accidents", 
                                            "Accidents per Department", year=select_year_dep, show_all_ticks=False)
                
                #dep_labels = np.array(df['acc_department'].unique()).tolist()
                #st_funct.severity_cat_barplot(df, 'acc_department', 'Department', dep_labels, year=select_year_dep)

                st_funct.severity_cat_barplot_alt(df, 'acc_department', 'Department', year=select_year_dep)

            with sub2_loc_tabs[1]:
                st.write('Crosstab:')

                st_funct.severity_cat_crosstab(df, 'acc_department', year=select_year_dep) #normalize: index

            with sub2_loc_tabs[2]:
                st.write('Urban vs. Rural:')

                img_paris = st_funct.load_image('art__dep_75_paris_wide.png')
                img_ardennen = st_funct.load_image('art__dep_08_ardennes_wide.png')

                loc_pic_col1, loc_pic_col2 = st.columns([5, 5])
                with loc_pic_col1: 
                    st.image(img_paris, caption='Paris - Department 75', width=350)
                with loc_pic_col2:
                    st.image(img_ardennen, caption='Ardennes - Department 08', width=350)

            with sub2_loc_tabs[3]:
                st.write('Department index:')

                dep_pop_df = st_funct.dep_pop_data_merge_reduced(df, df_dep_pop_filename).sort_values(by='acc_department')
                st.dataframe(dep_pop_df.head(200), use_container_width=True)
        
        with sub1_loc_tabs[1]:
            #st.write("##### Distribution")

            img_spread = st_funct.load_image('art__road-accidents_longlat.png')
            img_spread_class_3_4 = st_funct.load_image('art__road-accidents_longlat_hue_3_4.png')

            loc_spread_col1, loc_spread_col2 = st.columns([5, 5])
            
            with loc_spread_col1: 
                #img_1_slider = st.slider('Adjust image width (px)', 200, 500, 350,  key="img_1_slider_") # Min, max, default values
                st.image(img_spread, caption='Distribution of Road Accidents', width=370) #width=350 , #img_1_slider use_container_width =True
            with loc_spread_col2: 
                #img_2_slider = st.slider('Adjust image width (px)', 200, 500, 350,  key="img_2_slider_")
                st.image(img_spread_class_3_4, caption='Distribution of Road Accidents - High Severity', width=350)

    # ----------------------------------------------
    '''
    with tabs[3]:
        #st.write("Weather & Light");

        sub_atm_tab1, sub_atm_tab2 = st.tabs(["Weather Conditions", "Ambient Lightning"])

        with sub_atm_tab1:

            select_year = st.selectbox(
                "Select year:", 
                options = year_options_all,
                index = 0, key = 'selectbox_atm')
            
            st_funct.feature_distribution(df, "acc_atmosphere", "Weather Conditions", "Accidents", 
                                          "Accidents by Weather Condition", year=select_year)
            
            #atm_labels = np.array(df['acc_atmosphere'].unique()).tolist()
            atm_labels_ = ['Normal', 'Light Rain', 'Heavy Rain', 'Snow - Hail', 'Fog - Smoke', 
              'Strong Wind - Storm', 'Dazzling Weather', 'Overcast Weather', 'Other']
            
            st_funct.severity_cat_barplot(df, 'acc_atmosphere', 'Weather Conditions', atm_labels_, year=select_year)
            #st_funct.severity_cat_barplot_alt(df, 'acc_atmosphere', 'Weather Condition', year=select_year)

        with sub_atm_tab2:

            select_year = st.selectbox(
                "Select year:", 
                options = year_options_all,
                index = 0, key = 'selectbox_light')
            
            st_funct.feature_distribution(df, "acc_ambient_lightning", "Lightning Conditions", "Accidents", 
                                          "Accidents by Lightning Condition", year=select_year)
            
            #lum_labels = np.array(df['acc_ambient_lightning'].unique()).tolist()
            lum_labels_ = ['Broad daylight','Dusk or dawn','Night without street lighting', 
                          'Night with street lighting not on','Night with street lighting on']
            
            st_funct.severity_cat_barplot(df, 'acc_ambient_lightning', 'Lightning Conditions', lum_labels_, year=select_year)
            #st_funct.severity_cat_barplot_alt(df, 'acc_ambient_lightning', 'Lightning Condition', year=select_year)
    '''
    # ----------------------------------------------

    with tabs[3]:
        st.write("##### Collision Type");

        sub1_coll_tab_options = ["Distribution", "Index"]
        sub1_coll_tabs = st.tabs(sub1_coll_tab_options)

        with sub1_coll_tabs[0]:
        
            select_year_coll = st.selectbox(
                    "Select year:", 
                    options = year_options_all,
                    index = 0, key = 'selectbox_col')
            
            st_funct.feature_distribution(df, "acc_collision_type", "Collision Type", "Accidents", 
                                            "Accidents by Collision Type", year=select_year_coll)
            
            cols = df[['acc_collision_type', 'ind_severity', 'acc_year']]
            cols = cols.dropna()
            cols = cols[cols['acc_collision_type'] > -1]

            select_x_label_coll_type = st.radio(
                "Options",
                ('Show Keys', 'Show Labels'), 
                label_visibility = "collapsed",
                index = 0, 
                key = 'select_x_label_coll_type_'
            )

            if select_x_label_coll_type == 'Show Keys':
            
                col_key_labels = np.array(df['acc_collision_type'].unique()).tolist()
                st_funct.severity_cat_barplot(cols, 'acc_collision_type', 'Collision Type', col_key_labels, year=select_year_coll)

            elif select_x_label_coll_type == 'Show Labels':
            
                st_funct.severity_cat_barplot(cols, 'acc_collision_type', 'Collision Type', acc_coll_type_labels, year=select_year_coll)

            #acc_coll_type_df = st_lookup.create_acc_coll_type_df(df)
            #st_funct.severity_cat_barplot_alt(acc_coll_type_df, 'acc_collision_type_name', 'Collision Type', year=select_year_coll, remove_null=True, angle=90)

        with sub1_coll_tabs[1]:
            
            st.write("Collision type index:");

            acc_coll_type_lookup_df, acc_coll_type_keys, acc_coll_type_labels = st_lookup.acc_coll_type_lookup()
            st.dataframe(acc_coll_type_lookup_df.head(40), use_container_width=True)
    
    # ----------------------------------------------

    with tabs[4]:
        st.write("##### Vehicle Type");

        sub1_veh_tab_options = ["Distribution", "Index"]
        sub1_veh_tabs = st.tabs(sub1_veh_tab_options)

        with sub1_veh_tabs[0]:
    
            select_year_veh = st.selectbox(
                    "Select year:", 
                    options = year_options, #_all,
                    index = 0, key = 'selectbox_veh_')
            
            st_funct.feature_distribution(df, "veh_cat", "Vehicle Type", "Accidents", 
                                            "Accidents by Vehicle Type", year=select_year_veh, show_all_ticks=False)
            
            veh_labels = np.array(df['veh_cat'].unique()).tolist()
            st_funct.severity_cat_barplot(df, 'veh_cat', 'Vehicle Type', 
                                        veh_labels, year=select_year_veh, categorical_type=True)
            
            '''
            select_x_veh_cat = st.radio(
                "Options",
                ('Show Keys', 'Show Labels'), 
                label_visibility = "collapsed",
                index = 0, 
                key = 'select_x_veh_cat_'
            )

            if select_x_veh_cat == 'Show Keys':

                #eg remove_null=False, categorical=False, angle=90
                st_funct.severity_cat_barplot_alt(df, 'veh_cat', 'Vehicle Type', 
                                                year=select_year_veh, remove_null=True, categorical=True, angle=0)
            
            elif select_x_veh_cat == 'Show Labels':

                veh_cat_df = st_funct.create_veh_cat_df(df)
                st_funct.severity_cat_barplot_alt(veh_cat_df, 'veh_cat_name', 'Vehicle Type', 
                                                year=select_year_veh, remove_null=True, angle=90)
            '''

        with sub1_veh_tabs[1]:

            st.write("Vehicle type index:");
            veh_lookup_df = st_lookup.veh_cat_lookup()
            st.dataframe(veh_lookup_df.head(40), use_container_width=True)
    
    # ----------------------------------------------

    with tabs[5]:
        st.write("##### Security equipment");

        sub1_sec_tab_options = ["Distribution", "Index"]
        sub1_sec_tabs = st.tabs(sub1_sec_tab_options)

        with sub1_sec_tabs[0]:

            select_year_sec = st.selectbox(
                "Select year:", 
                options = year_options, #_all,
                index = 0, key = 'selectbox_secu')
            
            st_funct.feature_distribution(df, "ind_secu1", "Security equipment", "Accidents", 
                                            "Accidents by Security equipment", year=select_year_sec)
            
            #secu_index_keys = np.array(df['ind_secu1'].unique()).tolist()
            #st_funct.severity_cat_barplot(df, 'ind_secu1', 'Security equipment', secu_index_keys, year=select_year_sec) 
            #, categorical_type=True

            select_x_secu = st.radio(
                "Options",
                ('Show Keys', 'Show Labels'), 
                label_visibility = "collapsed",
                index = 0, 
                key = 'select_x_label_secu_'
            )

            if select_x_secu == 'Show Keys':

                st_funct.severity_cat_barplot_alt(df, 'ind_secu1', 'Security equipment', 
                                                year=select_year_sec, remove_null=True, angle=0)
            
            elif select_x_secu == 'Show Labels':
            
                ind_secu_df = st_lookup.create_ind_secu1_df(df)
                st_funct.severity_cat_barplot_alt(ind_secu_df, 'ind_secu1_name', 'Security equipment', 
                                                year=select_year_sec, remove_null=True, angle=90)
        
        with sub1_sec_tabs[1]:

            st.write("Security equipment index:")
            secu_lookup_df, secu_keys_, secu_labels = st_lookup.ind_secu1_lookup()
            st.dataframe(secu_lookup_df.head(40), use_container_width=True)

    # ----------------------------------------------

    with tabs[6]:
        st.subheader("Takeaways from data exploration")
        #st.write("### Takeaways")
        
        st.markdown("""
            #### Imbalanced target distribution
            Accident severity is highly imbalanced within the modalities. \n\n
            The majority of accidents lead to no or minor injuries; severe and and fatal outcomes are rare.
            """)
        
        st.markdown("""
            #### Key Takeaways
            - **Temporal Distribution:** Most accidents occur during rush hours (8 AM, 4-6 PM); night-time accidents are less frequent, but have a worse outcome.  
            - **Geographical Distribution:** Most accidents happen in urban areas, with Paris taking the lead; severe outcomes are distributed across all departments and municipalities; rural and mountainous areas tend to have a higher amount of severe outcomes.
            - **Vehicle type:** Two-wheelers and light vehicles show the highest average severity; heavy vehicles the lowest.
            - **Collision type:** Head-on and off-road crashes carry the highest severity; multi-vehicle collisions often have a less severe outcome.
            - **Security equipment:** The existence of security equipment, such as wearing a seatbelt or helm, has a positive influence on the outcome of an accident.
            """)
            #not related to severity
            #**Road & speed conditions:** Accidents cluster in 30-50 km/h zones, especially on metropolitan and county roads.
            #removed
            #- **Weather & Light:** 
            # Most accidents happen in daylight and normal weather. Rare weather conditions such as fog and storm come with a higher amount of severe outcomes.
        
    # ------------------------------------------
    #spacer at page bottom
    # ------------------------------------------
    st.write(' ')
    st.write(' ')
    st.write(' ')
    st.write(' ')

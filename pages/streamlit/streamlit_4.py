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
import copy

# custom libraries
import sys
sys.path.append('../../library')

import shap_values_aggr
#from cleaning_utils import distinguish_cols, print_col_categories

# warnings
import warnings
warnings.filterwarnings("ignore")

#streamlit
import streamlit as st
import streamlit.components.v1 as components

import streamlit_functions_model as st_funct_model
import streamlit_functions_shap as st_funct_shap

# ----------------------------------------------
# functions, variables
# ----------------------------------------------

def st_shap(plot, height=None):
    # Get the HTML and JS for the SHAP plot
    shap_html = f"<head>{shap.getjs()}</head><body>{plot.html()}</body>"
    # Render the HTML using components.v1.html
    components.html(shap_html, height=height)

@st.cache_data
def get_target_class_idx(class_label, shift=0):

    class_idx = -1
    
    if class_label=="Class 1":
        class_idx=0+shift
    elif class_label=="Class 2":
        class_idx=1+shift
    elif class_label=="Class 3":
        class_idx=2+shift
    elif class_label=="Class 4":
        class_idx=3+shift
    
    return class_idx

target_classes = ["Class 1", "Class 2", "Class 3", "Class 4", "All Classes"]

# ----------------------------------------------
# Page: Shap-Values Analysis
# ----------------------------------------------

def run_page_4():

    st.title("Shap-Values Analysis")

    # ------------------------------------------
    # loading saved shap values
    # ------------------------------------------
    shap_values_path = '../../models/shap_values_rf_multiclass_final.joblib'
    shap_values_2 = st_funct_shap.load(shap_values_path)

    # store feature_names in object + make a list object for indexing later
    feature_names_2 = shap_values_2.feature_names
    feature_names_list_2 = list(feature_names_2)

    # Load shap values
    shap_values = st_funct_shap.load_shap("streamlit_shap_values__test.joblib")
    shap_features = shap_values.feature_names
    
    checkbox_global_feature_importance = True
    checkbox_inspect_single_rows = False
    checkbox_beeswarm_feature_subsets = False
    checkbox_single_class_mean_impact = False
    checkbox_dependence_plot = False
    checkbox_aggregated_features = False

    # ------------------------------------------
    # Tabs
    # ------------------------------------------
    tab_list = [
        "Global Feature Importance",
        "Beeswarm",
        "Interaction Dependence",
        "Force Plot",
        "Aggregation Barplot"
    ]
    
    tabs = st.tabs(tab_list)

    # ------------------------------------------
    # Plot: global feature importance on prediction
    # ------------------------------------------

    with tabs[0]:
    #st.subheader("Global feature importance")
    #if st.checkbox("Global feature importance", value=True):
        
        with st.container(height=900):

            gfi_max_display_opt = 5
            gfi_range_start = 5
            gfi_max_display_opt = st.selectbox(
                "How many features would you like to see?",
                range(gfi_range_start, len(feature_names_2)+1),
                index = gfi_max_display_opt - gfi_range_start #eg 5 corresponds to default 10
                )
            st_funct_shap.shap_summary_plot(shap_values_2, 
                                            max_display=gfi_max_display_opt,
                                            width=6, height=8)
    
    # ------------------------------------------
    # Plot: beeswarm for all classes and feature subsets
    # ------------------------------------------

    with tabs[1]:
    #st.subheader("Inspect encoded multi-columns features")
    #if st.checkbox("Inspect feature subset"):
    
        with st.container(height=1500):

            dict = st_funct_shap.get_feature_dict(feature_names_2) #shap_features feature_names_2

            features_list = ['no selected']
            for col_id, col_details in dict.items():
                for key, value in col_details.items():
                    if (key == 'prefix') and ((value == 'onehot') or (value == 'cyclical')):
                        features_list.append(col_id)
            
            # Create the selectbox
            selected_option = ''
            selected_option = st.selectbox(
                "What feature would you like to explore?",
                features_list
            )
            st.write("You selected:", selected_option)

            if (selected_option != '') and (selected_option != 'no selected'):
            
                feature_name_prefix = ''
                for col_id, col_details in dict.items():
                    if selected_option in col_id:
                        for key, value in col_details.items():
                            if key == 'prefix': 
                                feature_name_prefix = f"{value}__{col_id}_"
                
                st.write(feature_name_prefix)
                st_funct_shap.shap_plot_feature(shap_values_2, feature_name_prefix) #shap_values shap_values_2
    
    # ------------------------------------------
    # Plot: bar plot for single class with mean impact 
    # ------------------------------------------
    '''
    st.subheader("Single class with mean impact")
    if st.checkbox("Single class with mean impact"):
    
        with st.container(height=900):

            sci_class_idx_length = shap_values_2.values.shape[2]

            sci_class_selection_idx = st.selectbox(
                "Select the target class:",
                range(-1, sci_class_idx_length), #-1:all
                index = 0
                )
            
            st_funct_shap.get_single_class_mean(shap_values_2, sci_class_selection_idx, sci_class_idx_length)
    '''
    # ------------------------------------------
    # Plot: dependency plot
    # -----------------------------------------

    with tabs[2]:
    #st.subheader("Dependency plot")
    #if st.checkbox("Dependency plot"):

        #todo:callback

        feature_list = copy.deepcopy(feature_names_list_2)
        feature_list.insert(0, 'No feature selected')

        interaction_list = copy.deepcopy(feature_names_list_2)
        interaction_list.insert(0, 'No interaction selected')

        feature_index = ''
        feature_index = st.selectbox(
            "Select feature:",
            feature_list
        )
        
        interaction_index = ''
        interaction_index = st.selectbox(
            "Select interaction:",
            interaction_list
        )
        st.write('If no interaction is selected, the most related feature will be chosen')

        dep_class_idx_length = shap_values_2.values.shape[2]

        dep_class_selection_idx = st.selectbox(
            "Select the target class:",
            range(-1, dep_class_idx_length),
            index = 1, key="dep_class_selection_idx_"
            )
        
        #with st.container(height=600):

        st_funct_shap.shap_dependence_plot(shap_values_2, feature_index, interaction_index, 
                                           target_class_idx=dep_class_selection_idx, target_class_idx_length=dep_class_idx_length)
    
    # ------------------------------------------
    # Plot: force plot to inspect single rows
    # ------------------------------------------

    with tabs[3]:
    #st.subheader("Inspect single rows")
    #if st.checkbox("Inspect single rows"):

        with st.container(height=900):

            fps_max_row_idx = shap_values_2.data.shape[0]-1 #eg 1234

            fps_row_idx = st.number_input(
                "Enter the row ID you would like to inspect:",
                min_value=0,
                max_value=fps_max_row_idx,
                value=fps_max_row_idx,
                step=1
            )

            fps_class_idx_length = shap_values_2.values.shape[2]

            fps_class_selection_idx = st.selectbox(
                "Select the target class:",
                range(-1, fps_class_idx_length),
                index = 0, key="fps_class_selection_idx_"
                )
            
            #fps_class_selection = st.selectbox(
            #    "Select the target class:", 
            #    options = target_classes, 
            #    index = 0)
            #fps_class_idx = get_target_class_idx(fps_class_selection, shift=0)

            st_funct_shap.init_shap_force_plot(shap_values_2, 
                                            fps_row_idx, 
                                            fps_class_selection_idx, fps_class_idx_length,
                                            use_comp=True)
    
    # ------------------------------------------
    # Plot: summary plot with aggregated features
    # -----------------------------------------
    with tabs[4]:
    #st.subheader("Aggregated shap values")
    #if st.checkbox("Aggregated shap values"):
    
        with st.container(height=900):

            #shap_values_agg = st_funct_shap.load_shap("streamlit_shap_values_agg__test.joblib")
            #fig = plt.figure()
            #shap.summary_plot(shap_values_agg, shap_values_agg.data, plot_type="bar", max_display=30)
            #st.pyplot(fig)

            explainer_agg, _ = st_funct_shap.get_aggregated_shap_object(shap_values_2, feature_names_list_2)
            
            max_display=50
            st_funct_shap.shap_summary_plot_v2(explainer_agg, max_display)

    # ------------------------------------------
    #spacer at page bottom
    # ------------------------------------------
    st.write(' ')
    st.write(' ')
    st.write(' ')
    st.write(' ')

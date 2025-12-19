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

# make custom libraries importable
import sys
sys.path.append('../../library')

import shap_values_aggr
#from cleaning_utils import distinguish_cols, print_col_categories

# evaluation
import shap

# helper
import time
import random

# warnings
import warnings
warnings.filterwarnings("ignore")

# streamlit
import streamlit as st
import streamlit.components.v1 as components

# ----------------------------------------------
# functions: shap values
# ----------------------------------------------

@st.cache_data
def load_shap(shap_joblib_file_name):
    shap_values = load(shap_joblib_file_name)
    return shap_values

@st.cache_data
def get_feature_dict(shap_features):
    dict, _  = shap_values_aggr.create_features_dict(shap_features)
    return dict

@st.cache_data
def get_aggregated_shap_object(_shap_values, feature_names_list):

    aggregator = shap_values_aggr.ShapOneHotAggregator(feature_names_list)
    explainer_agg, mapping = aggregator.aggregate(_shap_values.values, _shap_values.data)

    return explainer_agg, mapping

# ----------------------------------------------

def st_shap(plot, height=None):

    # Get the HTML and JS for the SHAP plot
    shap_html = f"<head>{shap.getjs()}</head><body>{plot.html()}</body>"
    # Render the HTML using components.v1.html
    components.html(shap_html, height=height)

# ----------------------------------------------

@st.cache_data
def shap_summary_plot(_shap_values, max_display=10, width=10, height=10):

    shap_values = _shap_values
    
    fig = plt.figure(figsize=(width, height))

    shap.summary_plot(shap_values.values[:, :, :], shap_values.data, 
                        feature_names=shap_values.feature_names, 
                        plot_type = 'bar', max_display = max_display)
    
    st.pyplot(fig, use_container_width=False)

@st.cache_data
def shap_summary_plot_v2(_shap_explainer, max_display):

    shap_explainer = _shap_explainer

    fig = plt.figure()
    shap.summary_plot(
        shap_explainer.values,
        shap_explainer.data,
        feature_names=shap_explainer.feature_names,
        plot_type="bar",
        max_display=max_display
    )
    st.pyplot(fig)

# ----------------------------------------------

@st.cache_data
def get_single_class_mean(_shap_values, sci_class_selection_idx, sci_class_idx_length):

    shap_values=_shap_values

    if sci_class_selection_idx >=0:
        st.write('Selected target class index:',sci_class_selection_idx)
        fig = plt.figure()
        shap.plots.bar(shap_values[:, :, sci_class_selection_idx].mean(0))
        st.pyplot(fig)
    else:
        st.write('All target classes:')
        for i in range(0,sci_class_idx_length):
            fig = plt.figure()
            shap.plots.bar(shap_values[:, :, i].mean(0))
            st.pyplot(fig)

# ----------------------------------------------

@st.cache_data
def init_shap_force_plot(_shap_values, row_idx, class_idx, class_idx_length, use_comp=True):

    if class_idx >= 0:
        shap_force_plot(_shap_values, row_idx, class_idx, use_comp)
    else:
        for i in range(0, class_idx_length):
            shap_force_plot(_shap_values, row_idx, i, use_comp)

# ----------------------------------------------

@st.cache_data
def shap_force_plot(_shap_values, row_idx, class_idx, use_comp=True):

    shap_values = _shap_values

    shap.initjs()

    # Correct slicing
    base_val = shap_values.base_values[row_idx, class_idx]
    shap_val = shap_values.values[row_idx, :, class_idx]
    features = shap_values.data[row_idx]
    
    # Force plot
    if use_comp:
        plot_shap = shap.force_plot(base_val, shap_val, features, 
                            feature_names=shap_values.feature_names)
        st_shap(plot_shap)
        
    else:
        plot_plt = shap.force_plot(base_val, shap_val, features, 
                                feature_names=shap_values.feature_names, 
                                matplotlib=True, show=False)
        st.pyplot(plot_plt)
        plt.close()

# ----------------------------------------------

@st.cache_data
def shap_plot_feature(_shap_values, feature_prefix): #dont hash shap_values

    shap_values = _shap_values
    shap_features = shap_values.feature_names

    # Filter names that start with prefix
    prefix = feature_prefix
    selected_features = [f for f in shap_features if f.startswith(prefix)]

    # Get indices of the selected features
    selected_idx = 0
    try:
        selected_idx = [shap_features.index(f) for f in selected_features]
    except Exception as e:
        feature_names_list = list(shap_features)
        selected_idx = [feature_names_list.index(f) for f in selected_features]
    
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

# ----------------------------------------------

@st.cache_data
def shap_dependence_plot(_shap_values, feature_index, interaction_index='', target_class_idx=0, target_class_idx_length=4):

    shap_values = _shap_values

    if (feature_index) and (feature_index != 'No feature selected'):

        for i in range(target_class_idx_length):

            index=i
            if target_class_idx>=0:index=target_class_idx

            if (interaction_index) and (interaction_index != 'No interaction selected'):
            
                shap.dependence_plot(
                    feature_index,
                    shap_values.values[:, :, index],
                    shap_values.data,
                    feature_names=shap_values.feature_names,
                    interaction_index=interaction_index,
                    show=False
                )
            else:

                # if no interaction_index is given, the most related feature will be chosen
                shap.dependence_plot(
                    feature_index,
                    shap_values.values[:, :, index],
                    shap_values.data,
                    feature_names=shap_values.feature_names,
                    show=False
                    )
            
            st.write('class idx:', index)
            
            ax = plt.gca()
            ax.set_xticks([0, 1])
            ax.set_xlim(-0.5, 1.5)
            plt.gcf().set_size_inches(8, 8) # 4 4
            plt.show()

            st.pyplot(plt)
            plt.close()

            if target_class_idx>=0:break;

# ----------------------------------------------

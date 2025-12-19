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
import plotly.graph_objs as go
import altair as alt
from PIL import Image

# stats
from scipy.stats import chi2_contingency

# custom libraries
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

# ----------------------------------------------
# load objects
# ----------------------------------------------

@st.cache_data
def load_df(df_joblib_file_name):
    df = joblib.load(df_joblib_file_name)
    return df

@st.cache_data
def load_dep_pop_data(df_dep_pop_filename):

    df_dep_pop = joblib.load(df_dep_pop_filename)
    df_dep_pop['dep_no'] = df_dep_pop['dep_no'].astype(str).str.zfill(2) #zfill
    return df_dep_pop

@st.cache_data
def merge_dep_pop_data(df, df_dep_pop):

    df_pop = pd.merge(df, df_dep_pop, left_on='acc_department', right_on='dep_no', how='left')

    df_copy = df.copy()
    df_copy['acc_dep_name'] = df_pop['dep_name']
    df_copy['acc_region'] = df_pop['region_1']
    return df_copy

@st.cache_data
def dep_pop_data_reduced(df_dep_pop_filename):

    df_pop = load_dep_pop_data(df_dep_pop_filename)

    df_pop['acc_department'] = df_pop['dep_no']
    df_pop['acc_dep_name'] = df_pop['dep_name']
    df_pop['acc_region'] = df_pop['region_1']

    df_dep_reduced = df_pop[['acc_department','acc_dep_name', 'acc_region']]
    return df_dep_reduced

@st.cache_data
def dep_pop_data_merge_reduced(df, df_dep_pop_filename):

    df_pop = load_dep_pop_data(df_dep_pop_filename)

    df_acc_dep = pd.DataFrame(df['acc_department'].unique().tolist(), columns=['acc_department'])

    df_pop_merge = pd.merge(df_acc_dep, df_pop, left_on='acc_department', right_on='dep_no', how='left')
    
    df_pop_merge['acc_dep_name'] = df_pop_merge['dep_name']
    df_pop_merge['acc_region'] = df_pop_merge['region_1']

    df_dep_reduced = df_pop_merge[['acc_department','acc_dep_name', 'acc_region']]
    return df_dep_reduced

@st.cache_resource
def load_image(image_path):
    try:
        image = Image.open(image_path)
        return image
    except FileNotFoundError:
        st.error(f"Error: Image not found at {image_path}")
        return None

# ----------------------------------------------
# plots
# ----------------------------------------------

def key_to_str(dict_int_key): 
    return {str(key): value for key, value in dict_int_key.items()}

#sns: severity distribution
@st.cache_data
def severity_distribution_multi_class(df, palette_color):

    custom_labels = ['uninjured','slightly injured','hospitalized','fatalities']

    data_label_counts = df['ind_severity'].value_counts()

    fig = plt.figure(figsize=(8, 6))
    #ax = 
    sns.countplot(x='ind_severity', data=df, order=data_label_counts.index, palette=key_to_str(palette_color))
    tick_positions = np.arange(len(custom_labels))
    plt.xticks(tick_positions, labels=custom_labels, rotation=0)
    #plt.xticks(rotation=rotation)
    plt.xlabel('Severity')
    plt.ylabel('Count')
    plt.title("Severity Distribution") #Multi-Class
    plt.tight_layout()
    plt.show()
    st.pyplot(fig) #plt
    #plt.close()

# ----------------------------------------------

#plotly: countplot
@st.cache_data
def feature_distribution(df, feature, label_x, label_y, title, year='', show_all_ticks=True):

    df_feature = df
    if (year != '') and (year != 'All'):
        df_feature = df[df['acc_year'] == int(year)]
    
    counts = df_feature[feature].value_counts().sort_index()
    fig2 = px.bar(
    x=counts.index,
    y=counts.values,
    labels={"x": label_x, "y": label_y},
    title=title)
    
    if show_all_ticks:

        fig2.update_xaxes(
            tickmode='linear'
        )

    st.plotly_chart(fig2, use_container_width=True)

def feature_distribution_(df, feature, label_x, label_y, title, year=''):

    df_feature = df
    if (year != '') and (year != 'All'):
        df_feature = df[df['acc_year'] == int(year)]
    
    counts = df_feature[feature].value_counts().sort_index()
    fig2 = px.bar(
    x=counts.index,
    y=counts.values,
    labels={"x": label_x, "y": label_y},
    title=title)
    st.plotly_chart(fig2, use_container_width=True)

# ----------------------------------------------

#sns: countplot
@st.cache_data
def sns_countplot(df, column_name, title):
    
    plot_color = 'deepskyblue'
    bar_width=0.8

    fig = plt.figure(figsize=(8, 6))
    #ax = 
    sns.countplot(x=column_name, data=df, color=plot_color, width=bar_width)
    plt.title(title)
    plt.xlabel(column_name)
    plt.ylabel('Count')
    plt.tight_layout()
    plt.show()
    st.pyplot(fig)

@st.cache_data
def sns_countplot_(df, column_name, title, year=0):

    title_postfix =''
    if year > 0:
        char = df[df['acc_year'] == year]
        title_postfix += f" ({year})"
    else:
        char=df
    
    plot_color = 'deepskyblue'
    bar_width=0.8

    fig = plt.figure(figsize=(8, 6))
    #ax = 
    sns.countplot(x=column_name, data=char, color=plot_color, width=bar_width)
    plt.title(f"{title} {title_postfix}")
    plt.xlabel(column_name)
    plt.ylabel('Count')
    plt.tight_layout()
    plt.show()
    st.pyplot(fig)

# ----------------------------------------------

#altair: barplot stacked severity per category (liminted df size)
@st.cache_data
def severity_cat_barplot_alt(df, col_name, col_title, year='', remove_null=False, categorical=False, angle=90):

    #df_copy = df.copy()
    df_copy = df[[col_name,'ind_severity','acc_year']]

    if (year != '') and (year != 'All'):
        df_filter = df_copy[df_copy['acc_year'] == int(year)]
    else:
        df_filter = df_copy
    
    if remove_null:
        #df_filter = df_filter.dropna(subset=[col_name], inplace=True)
        #df_filter = df_filter[df_filter[col_name].notna()]
        df_filter = df_filter.dropna()

    if categorical:
        df_filter[col_name]= df_filter[col_name].astype('category') #q
    
    domain_ = [1,2,3,4]
    range_ = ['green','yellowgreen','orange','darkred']
    
    st.altair_chart(alt.Chart(df_filter)
        .mark_bar()
        .encode(
            alt.X(f"{col_name}:O", title=col_title, axis=alt.Axis(labelAngle=angle)), #eg 45 #O N
            alt.Y("count():Q", title="Accidents").stack("normalize"),
            alt.Color("ind_severity:N", scale=alt.Scale(domain=domain_, range=range_)),
        )
        .configure_legend(orient="bottom")
        )

#plotly: barplot stacked severity per category
@st.cache_data
def severity_cat_barplot(df, col_name, col_title, col_labels, exclude_0=False, remove_0=False, year='', categorical_type=False):

    df_feature = df
    if (year != '') and (year != 'All'):
        try:
            df_feature = df[df['acc_year'] == int(year)]
        except Exception as e:
            year=2019
            df_feature = df[df['acc_year'] == int(year)]
    
    data = df_feature[[col_name, 'ind_severity']]
    data = data.dropna()
    
    if remove_0: 
        data = data[data[col_name] > -1]

    if categorical_type:
        data[col_name]= data[col_name].astype('category') #td:label
    
    colors_grav_px = {1:'green', 2:'yellowgreen', 3:'orange', 4:'darkred'}
    keydict = {1:'Uninjured', 2:'Slightly injured', 3: 'Hospitalized', 4: 'Killed'}
    traces = []
    
    for key, grp in data.groupby(data.ind_severity):
        if (key != 0):
            
            #count = data[col_name].count()
            aggregated = (grp[col_name].value_counts()).sort_index()
        
            x_values = aggregated.index.tolist()
            y_values = (aggregated.values/data[col_name].value_counts().sort_index().values*100).tolist()
            
            if exclude_0:
                x1,y1 = x_values[1:], y_values[1:] #q
            else:
                x1,y1 = x_values[0:], y_values[0:] #dont exclude values
            
            x1 = col_labels
            
            trace1 = go.Bar(x=x1, y=y1, opacity=0.75, name = keydict[key], marker_color = colors_grav_px[key])
            
            layout = dict(height=400, barmode = 'stack', 
                          title=f"Relationship between {col_title} and Severity of Accident", 
                          yaxis = dict(title = 'Percentage'), xaxis = dict(title = col_title));
            
            traces.append(trace1)
            
    fig = go.Figure(data=traces, layout=layout)
    #fig.show()
    #iplot(fig)
    st.plotly_chart(fig, use_container_width=True)

# ----------------------------------------------

#crosstab
@st.cache_data
def severity_cat_crosstab(df, col_name, year='', normalize_type='index'):

    df_feature = df
    if (year != '') and (year != 'All'):
        df_feature = df[df['acc_year'] == int(year)]
    
    data = df_feature[[col_name, 'ind_severity']]
    data = data.dropna()
    
    #crosstab
    crosstab_table = pd.crosstab(data[col_name], data['ind_severity'], normalize=normalize_type)
    st.write(crosstab_table)

#heatmap
@st.cache_data
def severity_cat_cross_heatmap(df, col_name, col_title, title, yticks, year=''):

    df_feature = df
    if (year != '') and (year != 'All'):
        df_feature = df[df['acc_year'] == int(year)]
    
    data = df_feature[[col_name, 'ind_severity']]
    data = data.dropna()
    
    #crosstab
    crosstab_table = pd.crosstab(data[col_name], data['ind_severity'], normalize='index')
    
    #heatmap
    fig = plt.figure(figsize=(8, 6))
    
    xticks = ['Uninjured', 'Slightly injured', 'Hospitalized', 'Killed']
    yticks = yticks
    
    ax = sns.heatmap(crosstab_table, annot=True, cmap='Blues', linewidths=.5, 
                     yticklabels = yticks, xticklabels = xticks)
    
    ax.invert_yaxis()
    ax.tick_params(axis='x', labelrotation=45)
    
    plt.xlabel('Severity')
    plt.ylabel(col_title)
    plt.title(title)
    
    plt.tight_layout()
    plt.show()
    st.pyplot(fig)

# ----------------------------------------------

#chi #td
def grav_cat_chi2(df, col_name, cramer=False, cramer_evaluation=False):
    
    data = df[[col_name, 'ind_severity']]
    data = data.dropna()
    
    crosstab_table = pd.crosstab(data[col_name], data['ind_severity'], normalize='index')
    st.write(crosstab_table)
    
    #H0: The variable is independent of grav
    #H1: The variable is not independent of grav
    
    res = chi2_contingency(crosstab_table)
    
    statistic_ch2 = res[0]
    pvalue_ch2 = res[1]
    
    print(f"Chi-square Statistic: {res[0]}")
    print(f"P-value: {res[1]}")
    
    if pvalue_ch2 > 0.05: 
        print('H0 is rejected, since the p-value is large.')
    else:
        print('H0 is not rejected.')
    
    if (pvalue_ch2 > 0.05) and (cramer):
        
        v_cramer = np.sqrt(statistic_ch2 / crosstab_table.values.sum()) #q:normalize
        
        print()
        print('V Cramer:', v_cramer)
        
        if cramer_evaluation:
            
            print()
            if v_cramer < 0.08: 
                print('V Cramer result is very weak.')
            elif (v_cramer >= 0.08) & (v_cramer < 0.2): 
                print('V Cramer result is weak.')
            elif (v_cramer >= 0.2) & (v_cramer < 0.45):
                print('V-Cramer result is medium.')
            else:
                print('V-Cramer result is High.')
            
            print()
            print('V-Cramer evaluation:')
            print('Weak : Value around 0.1')
            print('Medium : Value around 0.3')
            print('High : Value around and larger than 0.5')

# ----------------------------------------------

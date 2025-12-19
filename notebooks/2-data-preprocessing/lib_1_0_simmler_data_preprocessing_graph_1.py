##########

import pandas as pd
import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Circle
from matplotlib.lines import Line2D

from plotly.offline import init_notebook_mode, iplot
import plotly.graph_objs as go
from plotly import tools
import plotly.express as py

##########

#plot color
plot_color = 'deepskyblue' #darkturquoise deepskyblue dodgerblue cornflowerblue

#colors
colors_grav_org_order = {1: 'green', 2: 'darkred', 3: 'orange', 4: 'yellowgreen'}
colors_grav_reorder = {1: 'green', 2: 'yellowgreen', 3: 'orange', 4: 'darkred'}
colors_grav = colors_grav_org_order

#plotly
colors_1 = py.colors.qualitative.Plotly
colors_2 = py.colors.qualitative.D3
colors_3 = py.colors.qualitative.Set2
colors_4 = py.colors.qualitative.Pastel1
colors_5 = py.colors.qualitative.Pastel2

c_uninjured = colors_2[2] #green
c_slighty_injured = colors_1[7] #light-green/yellow  #colors_1[9] colors_1[7] #colors_5[4]
c_hospitalized = colors_2[1] #orange
c_killed = 'darkred' #red

colors_grav_px_1 = {1:'green', 2:'yellowgreen', 3:'orange', 4:'darkred'}
colors_grav_px = {1: c_uninjured, 2: c_slighty_injured, 3: c_hospitalized, 4: c_killed}

##########

def sns_countplot_years(char, column_name, column_title='', title='', title_postfix='', 
                           bar_width=0.8, xtick_freq=1, xtick_rotate=0, ordered=False):
    
    col_label = column_title if column_title else column_name
    graph_title = title if title else 'Number of Accidents per ' + col_label
    
    fig, axes = plt.subplots(3, 2, figsize=(10, 12))
    axes_flat = axes.flatten()
    
    for i, ax in enumerate(axes_flat):
    
        year = i + 2019
        char_year = char[char['acc_year'] == year]
        
        if ordered:
            col_counts = char_year[column_name].value_counts()
            col_counts_order = col_counts.index
            
            sns.countplot(ax=ax, x=column_name, data=char_year, order=col_counts_order, color=plot_color, width=bar_width) #0.5
        else:
            sns.countplot(ax=ax, x=column_name, data=char_year, color=plot_color, width=bar_width)
        
        if xtick_freq > 1:
            current_ticks = ax.get_xticks()
            new_ticks = current_ticks[::xtick_freq]
            ax.set_xticks(new_ticks)
        
        if xtick_rotate > 0:
            ax.tick_params(axis='x', labelrotation=xtick_rotate)
            #plt.xticks(rotation=xtick_rotate)
        
        #ax.set_xlabel(col_label)
        ax.set_ylabel('Count')
        ax.set_title(year)
    
    plt.suptitle(f"{graph_title} {title_postfix}")
    plt.tight_layout()
    plt.show()

def sns_countplot(char, column_name, column_title='', title='', title_postfix='', year=0, 
                  bar_width=0.8, xtick_freq=1, xtick_rotate=0, ordered=False):
    
    if year > 0:
        char = char[char['acc_year'] == year]
        title_postfix += f" ({year})"
    
    col_label = column_title if column_title else column_name
    graph_title = title if title else 'Number of Accidents per ' + col_label
    
    plt.figure(figsize=(8, 6))
    
    if ordered:
        col_counts = char[column_name].value_counts()
        col_counts_order = col_counts.index
        ax = sns.countplot(x=column_name, data=char, order=col_counts_order, color=plot_color, width=bar_width) #0.5
    else:
        ax = sns.countplot(x=column_name, data=char, color=plot_color, width=bar_width)
    
    if xtick_freq > 1:
        current_ticks = ax.get_xticks()
        new_ticks = current_ticks[::xtick_freq] #3
        ax.set_xticks(new_ticks)
    
    if xtick_rotate > 0:
        plt.xticks(rotation=xtick_rotate) #45
    
    plt.title(f"{graph_title} {title_postfix}")
    plt.xlabel(col_label)
    plt.ylabel('Count')
    
    plt.tight_layout()
    plt.show()

def countplot(char, column_name, column_title='', title='', title_postfix=''):

    col_label = column_title if column_title else column_name
    graph_title = title if title else 'Number of Accidents per ' + col_label
    
    char_col_counts = char[column_name].value_counts()
    
    plt.figure(figsize=(8, 6))
    
    char_col_counts.plot(kind='bar', color=plot_color)
    
    plt.title(f"{graph_title} {title_postfix}")
    plt.xlabel(col_label)
    plt.ylabel('Count')
    plt.tight_layout()
    plt.show()

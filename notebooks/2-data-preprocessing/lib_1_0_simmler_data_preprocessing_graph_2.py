##########

import pandas as pd
import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Circle
from matplotlib.lines import Line2D

import plotly.graph_objs as go
from plotly import tools
import plotly.express as py
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode(connected=True)

from scipy.stats import chi2_contingency

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

def key_to_str(dict_int_key): 
    return {str(key): value for key, value in dict_int_key.items()}

##########

def hallo(): print('hallo')

##########

#countplots
def countplot(char, column_name, column_title='', title='', title_postfix='', plot_color='deepskyblue'):

    col_label = column_title if column_title else column_name
    graph_title = title if title else 'Number of Accidents per ' + col_label
    
    char_col_counts = char[column_name].value_counts()
    char_col_counts.plot(kind='bar', color=plot_color)
    
    plt.title(f"{graph_title} {title_postfix}")
    plt.xlabel(col_label)
    plt.ylabel('Count')
    plt.tight_layout()
    plt.show()

def sns_countplot_grav_(df):
    
    plt.figure(figsize=(8, 6))
    
    try:
        sns.countplot(x='ind_severity', data=df, palette=colors_grav_reorder, width=0.8)
    except ValueError as e: 
        #type(e).__name__
        if 'palette dictionary is missing key' in str(e):
            sns.countplot(x='ind_severity', data=df, palette=key_to_str(colors_grav_reorder), width=0.8)
        else: raise e
    
    plt.title(f"Distribution of severity")
    plt.xlabel('Severity')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.show()

def sns_countplot(char, column_name, column_title='', title='', title_postfix='', 
                  plot_color='deepskyblue', palette_color=None, bar_width=0.8, xtick_freq=1, xtick_rotate=0, ordered=False):
    
    col_label = column_title if column_title else column_name
    graph_title = title if title else 'Number of Accidents per ' + col_label
    if palette_color: plot_color=None
    
    if ordered:
        col_counts = char[column_name].value_counts()
        col_counts_order = col_counts.index
    else: 
        col_counts_order=None
    
    plt.figure(figsize=(8, 6))
    
    try:
        ax = sns.countplot(x=column_name, data=char, order=col_counts_order, color=plot_color, palette=palette_color, width=bar_width) #0.5
    except ValueError as e:
        if 'palette dictionary is missing key' in str(e):
            ax = sns.countplot(x=column_name, data=char, order=col_counts_order, 
                               color=plot_color, palette=key_to_str(palette_color), width=bar_width) #0.5
        else: raise e
    
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

def sns_countplot_labels(df, cat_name, label_dict, rotation=45):
    
    data = df.copy()
    data = data.dropna()
    
    data_cat_counts = data[cat_name].value_counts()
    
    data['cat_label'] = data[cat_name].map(label_dict)
    data_label_counts = data['cat_label'].value_counts()
    
    plt.figure(figsize=(8, 6))
    ax = sns.countplot(x='cat_label', data=data, order=data_label_counts.index, color=plot_color)
    plt.xticks(rotation=rotation)
    plt.xlabel(cat_name)
    plt.ylabel('Count')
    plt.title(f"Number of Accidents per {cat_name}")
    plt.tight_layout()
    plt.show()

def sns_countplot_labels_2(df, cat_name, cat_title, label_dict, rotation=45):

    if not cat_title: 
        cat_title=cat_name
    
    data = df.copy()
    data = data.dropna()
    
    data_cat_counts = data[cat_name].value_counts()
    
    data['cat_label'] = data[cat_name].map(label_dict)
    data_label_counts = data['cat_label'].value_counts()
    
    plt.figure(figsize=(8, 6))
    ax = sns.countplot(x='cat_label', data=data, order=data_label_counts.index, color=plot_color)
    plt.xticks(rotation=rotation)
    plt.xlabel(cat_name)
    plt.ylabel('Count')
    plt.title(f"Number of Accidents per {cat_title}")
    plt.tight_layout()
    plt.show()
    
##########

# aggregate over col and return keys and values
def create_aggregated_data(col, df):
    aggregated = df[col].value_counts().sort_index()
    x_values = aggregated.index.tolist()
    y_values = aggregated.values.tolist()
    return x_values, y_values

#barplot stacked severity per category
def grav_cat_barplot(df, col_name, col_title, col_labels, exclude_0=False, remove_0=False):

    data = df[[col_name, 'ind_severity']]
    data = data.dropna()
    
    if remove_0: 
        data = data[data[col_name] > -1] #td
    
    keydict = {1:'Uninjured', 2:'Slightly injured', 3: 'Hospitalized', 4: 'Killed'}
    traces = []
    
    for key, grp in data.groupby(data.ind_severity):
        if (key != 0):
            
            count = data[col_name].count()
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
    fig.show()
    #iplot(fig)

##########

#heatmap
def grav_cat_cross_heatmap(df, col_name, col_title, title, yticks):
    
    data = df[[col_name, 'ind_severity']]
    data = data.dropna()
    
    #crosstab
    crosstab_table = pd.crosstab(data[col_name], data['ind_severity'], normalize='index')
    #display(crosstab_table)
    
    #heatmap
    plt.figure(figsize=(8, 6))
    
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

def grav_cat_chi2__backup(df, col_name):
    
    data = df[[col_name, 'ind_severity']]
    data = data.dropna()
    
    crosstab_table = pd.crosstab(data[col_name], data['ind_severity'], normalize='index')
    display(crosstab_table)
    
    #H0: The variable is independent of grav
    #H1: The variable is not independent of grav
    
    from scipy.stats import chi2_contingency
    res = chi2_contingency(crosstab_table)
    
    statistic_ch2 = res[0]
    pvalue_ch2 = res[1]
    
    print(f"Chi-square Statistic: {res[0]}")
    print(f"P-value: {res[1]}")
    
    if res[1] > 0.05: 
        print('H0 is rejected, since the p-value is large.')
    else:
        print('H0 is not rejected.')

def grav_cat_chi2(df, col_name, cramer=False, cramer_evaluation=False):
    
    data = df[[col_name, 'ind_severity']]
    data = data.dropna()
    
    crosstab_table = pd.crosstab(data[col_name], data['ind_severity'], normalize='index')
    display(crosstab_table)
    
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

def df_chi2_cramer(df, features, target, show_all=False, pvalue_threshold = 0.05, vcramer_threshold = 0.08):
    
    data = df.dropna()
    
    chi2_column_names = ['feature', 'chi2_stat', 'p_value', 'v_cramer', 'v_cramer_evaluation']
    chi2_df = pd.DataFrame(columns = chi2_column_names)
    
    for col_name in features:
        
        crosstab_table = pd.crosstab(data[col_name], data[target], normalize='index')
        #display(crosstab_table)
        
        res = chi2_contingency(crosstab_table)
        
        statistic_ch2 = res[0]
        p_value_ch2 = res[1]
        
        v_cramer = np.sqrt(statistic_ch2 / crosstab_table.values.sum()) #q:normalize
        
        #P-Value
        #H0: The variable is independent of target < 0.05
        #H1: The variable is not independent of target > 0.05
        
        #V-Cramer
        #Weak : Value around 0.1
        #Medium : Value around 0.3
        #High : Value around and larger than 0.5
        
        if (show_all) | ((p_value_ch2 >= pvalue_threshold) and (v_cramer >= vcramer_threshold)):
        
            #print(f"Chi-square Statistic: {statistic_ch2}")
            #print(f"P-value: {p_value_ch2}")
            #print('V-Cramer:', v_cramer)
            
            if v_cramer < vcramer_threshold: 
                evaluation = 'Below Threshold'
            elif (v_cramer >= vcramer_threshold) & (v_cramer < 0.2): 
                evaluation = 'Weak'
            elif (v_cramer >= 0.2) & (v_cramer < 0.45):
                evaluation = 'Medium'
            else:
                evaluation = 'High'
            
            chi2_df.loc[len(chi2_df)] = [col_name, statistic_ch2, p_value_ch2, v_cramer, evaluation]
    
    return chi2_df

##########

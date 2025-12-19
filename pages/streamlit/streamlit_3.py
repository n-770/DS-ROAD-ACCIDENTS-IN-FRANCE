# ----------------------------------------------
# import modules
# ----------------------------------------------
# app.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import json
from pathlib import Path
import joblib
import io
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
import shap

# ----------------------------------------------
# Page: Modeling & Evaluation
# ----------------------------------------------
@st.cache_data
def load_metrics(model_path):
    # Load precomputed metrics stored via joblib
    return joblib.load(model_path)
def kpi_card(title, value, color="#2E86C1"):
    st.markdown(
        f"""
        <div style="background-color:{color}; padding:15px; border-radius:10px; text-align:center; color:white;">
            <h4 style="margin:0;">{title}</h4>
            <p style="font-size:24px; margin:0;"><b>{value}</b></p>
        </div>
        """,
        unsafe_allow_html=True
    )

def show_model_dashboard(metrics, model_name):
    st.subheader(model_name)

    # KPI Cards 
    col1, col2, col3, col4 = st.columns(4)

    # Macro F1 (Train/Test) → green
    with col1: kpi_card("Train Macro F1", f"{metrics['train']['macro_f1']:.3f}", "#27AE60")
    with col3: kpi_card("Test Macro F1", f"{metrics['test']['macro_f1']:.3f}", "#27AE60")

    # Recall (Train/Test) → blue
    with col2: kpi_card("Train Recall", f"{metrics['train']['recall']:.3f}", "#2980B9")
    with col4: kpi_card("Test Recall", f"{metrics['test']['recall']:.3f}", "#2980B9")

    # Tabs for Details
    detail_tabs = st.tabs(["Per-Class Report", "Feature Importance","Confusion Matrix"])

    # Per-Class Report
    with detail_tabs[0]:
        per_class = metrics["test"].get("per_class", {})
        if per_class:
            df_pc = pd.DataFrame(per_class).T
            df_pc_rounded = df_pc.round(2)
            styled_df = df_pc_rounded.style.format("{:.2f}", na_rep="-")
            st.dataframe(styled_df, use_container_width=True)

    # Feature Importance 
    with detail_tabs[1]:
    
        fi = metrics.get("feature_importance", {})
        if fi:
            fi = pd.DataFrame(fi)
            fi = fi.sort_values("importance", ascending=False).head(20)
            fig = px.bar(fi, x="importance", y="feature", orientation="h",
                     title=f"Top Feature Importances – {model_name}", template="plotly_white",
                     color="importance", color_continuous_scale="Viridis")
            fig.update_layout(yaxis=dict(autorange="reversed"))
            st.plotly_chart(fig, use_container_width=True)
    # Confusion Matrix
    with detail_tabs[2]:
        cm = metrics.get("confusion_matrix", None)
        if cm is not None:
            cm_df = pd.DataFrame(cm)
            fig = px.imshow(cm_df, text_auto=True, color_continuous_scale="Blues",
                        title=f"Confusion Matrix – {model_name}", template="plotly_white")
            st.plotly_chart(fig, use_container_width=True)
            
def run_page_3():
    
    st.title("Modeling & Evaluation")

    #model_files = {
    #"RandomForest(multiclass)":"/Users/tchokomeny/Downloads/sep25_bds_int_road_accidents/project_team/SEP25-BDS-Road-Accidents/src/streamlit/art_metrics_rf_multiclass.joblib", 
    #"RandomForest(binary)":"/Users/tchokomeny/Downloads/sep25_bds_int_road_accidents/project_team/SEP25-BDS-Road-Accidents/src/streamlit/metrics_rf_binary_final.joblib",
    #"XGBClassifier (multiclass)": "/Users/tchokomeny/Downloads/sep25_bds_int_road_accidents/project_team/SEP25-BDS-Road-Accidents/src/streamlit/art_metrics_xgb_multiclass.joblib",
    #"XGBClassifier (binary)": "/Users/tchokomeny/Downloads/sep25_bds_int_road_accidents/project_team/SEP25-BDS-Road-Accidents/src/streamlit/art_metrics_xgb_binary.joblib"}
    
    model_files = {
    "RandomForest (multiclass)":"art_metrics_rf_multiclass.joblib", 
    "XGBClassifier (multiclass)":"art_metrics_xgb_multiclass.joblib",
    "RandomForest (binary)":"metrics_rf_binary_final.joblib",
    "XGBClassifier (binary)": "art_metrics_xgb_binary.joblib"}

    tabs = st.tabs(list(model_files.keys()))
    for tab, (model_name, model_path) in zip(tabs, model_files.items()):
        with tab:
            metrics = load_metrics(model_path)
            show_model_dashboard(metrics, model_name)
 
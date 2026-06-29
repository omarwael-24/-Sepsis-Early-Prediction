import os
import sys
import joblib
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parents[1]))

# Direct imports from the root-appended path
from predict import SepsisPredictor
from data_preprocessing import DataLoader, SepsisDataProcessor, DropUnwantedColumns
from feature_engineering import HourFeature, ExtractingRollingFeatures

# Page configuration
st.set_page_config(page_title="Sepsis Early Warning System", layout="wide")

st.title("🏥 Sepsis Early Warning Dashboard")
st.markdown("Expert Clinical Decision Support System for Early Sepsis Detection in ICU Patients")

@st.cache_data
def load_data():
    raw_df = DataLoader.load_clean()
    if isinstance(raw_df.index, pd.MultiIndex):
        raw_df = raw_df.reset_index()
    elif raw_df.index.name is not None:
        raw_df = raw_df.reset_index()
    return raw_df

try:
    raw_df = load_data()
    predictor = SepsisPredictor()
    
    # Patient Selection
    patient_ids = raw_df['Patient_ID'].unique()
    selected_patient = st.sidebar.selectbox("Select Patient ID:", patient_ids)
    
    patient_data = raw_df[raw_df['Patient_ID'] == selected_patient].copy()
    st.sidebar.metric(label="Total Monitored Hours", value=f"{len(patient_data)} hrs")
    
    # Run Preprocessing Pipeline
    df_processed = HourFeature.transform_hour_column(patient_data.copy())
    df_processed = SepsisDataProcessor.SetMultIndexGroup(df_processed)
    df_processed = SepsisDataProcessor.impute_missing_values(df_processed)
    df_processed = ExtractingRollingFeatures.extracting_rolling_features(df_processed)
    df_processed = DropUnwantedColumns.drop_unwanted_columns(df_processed)
    
    df_processed = df_processed.reset_index()

    # Timeline Slider
    selected_hour_idx = st.slider("Select Evaluation Hour:", 0, len(df_processed)-1, len(df_processed)-1)
    
    # Extract Target Row for Inference
    target_row = df_processed.iloc[[selected_hour_idx]].copy()
    
    columns_to_drop = [col for col in ['SepsisLabel', 'Patient_ID', 'Hour'] if col in target_row.columns]
    target_row = target_row.drop(columns=columns_to_drop)
        
    # Model Prediction
    prob, alarm = predictor.predict_action(target_row)
    sepsis_prob_pct = prob[0] * 100

    # Display Metrics
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Current Risk Assessment")
        if alarm[0] == 1:
            st.error(f"🚨 ALERT: High Risk of Sepsis! (Probability: {sepsis_prob_pct:.2f}%)")
        else:
            st.success(f"✅ STATUS: Patient is stable. (Probability: {sepsis_prob_pct:.2f}%)")
            
    with col2:
        st.subheader("Threshold Comparison")
        st.metric(label="Sepsis Probability", value=f"{sepsis_prob_pct:.2f}%")
        st.caption(f"Model Critical Threshold: {predictor.best_threshold*100:.2f}%")

    # Vital Signs Plot
    st.subheader("📈 Vital Signs Timeline")
    vitals = [col for col in ['HR', 'O2Sat', 'Temp', 'SBP', 'MAP', 'DBP', 'Resp'] if col in patient_data.columns]
    selected_vital = st.selectbox("Select Vital Sign to Plot:", vitals)
    
    fig = px.line(patient_data, x='Hour', y=selected_vital, title=f"{selected_vital} Trend over Time")
    fig.add_vline(x=selected_hour_idx, line_dash="dash", line_color="red", annotation_text="Evaluation Hour")
    st.plotly_chart(fig, use_container_width=True)

except Exception as e:
    st.error(f"An error occurred while running the dashboard: {e}")
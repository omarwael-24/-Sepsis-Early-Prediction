import os
import sys
import joblib
import pandas as pd
import numpy as np
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))
from src.data_preprocessing import DataLoader, SepsisDataProcessor, DropUnwantedColumns
from src.feature_engineering import HourFeature, ExtractingRollingFeatures

class SepsisPredictor:
    def __init__(self, models_dir="models"):
        self.models_dir = models_dir
        self.models = []
        self.best_threshold = 0.5
        self.feature_names = []
        self._load_pipeline()

    def _load_pipeline(self):
        metadata_path = os.path.join(self.models_dir, "pipeline_metadata.pkl")
        metadata = joblib.load(metadata_path)
        self.best_threshold = metadata["best_threshold"]
        self.feature_names = metadata["feature_names"]
        for i in range(1, 6):
            self.models.append(joblib.load(os.path.join(self.models_dir, f"lgbm_fold_{i}.pkl")))

    def predict_action(self, X_input: pd.DataFrame):
        X_aligned = X_input[self.feature_names]
        fold_probs = [model.predict_proba(X_aligned)[:, 1] for model in self.models]
        prob = np.mean(fold_probs, axis=0)
        return prob, (prob >= self.best_threshold).astype(int)

if __name__ == "__main__":
    print("--- Live Patient Inference Test ---")
    raw_df = DataLoader.load_clean()
    
    if isinstance(raw_df.index, pd.MultiIndex):
        raw_df = raw_df.reset_index()
    elif raw_df.index.name is not None:
        raw_df = raw_df.reset_index()

    sepsis_df = raw_df[raw_df['SepsisLabel'] == 1]
    sample_id = sepsis_df['Patient_ID'].iloc[0]
    
    patient_data = raw_df[raw_df['Patient_ID'] == sample_id].copy()
    print(f"Patient ID: {sample_id} ({len(patient_data)} hours)")

    # Run complete preprocessing pipeline
    df = HourFeature.transform_hour_column(patient_data)
    df = SepsisDataProcessor.SetMultIndexGroup(df)
    df = SepsisDataProcessor.impute_missing_values(df)
    df = ExtractingRollingFeatures.extracting_rolling_features(df)
    df = DropUnwantedColumns.drop_unwanted_columns(df)
    
    # Isolate the exact onset of symptoms if available
    danger_hours = df[df['Spesis_onset'] == 1]
    if not danger_hours.empty:
        target_hour = danger_hours.tail(1)
        print("🎯 Testing exactly at the onset of Sepsis symptoms!")
    else:
        target_hour = df.tail(1)

    if 'SepsisLabel' in target_hour.columns: 
        target_hour = target_hour.drop(columns=['SepsisLabel'])

    predictor = SepsisPredictor()
    prob, alarm = predictor.predict_action(target_hour)
    
    print(f"Sepsis Probability: {prob[0]*100:.2f}%")
    print(f"Threshold: {predictor.best_threshold * 100:.2f}%")
    print(f"Alarm: {'🚨 ALERT' if alarm[0] == 1 else '✅ Stable'}")
import sys
import os
from pathlib import Path
import joblib

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.config import Config
from src.data_preprocessing import DataLoader, SepsisDataProcessor, DropUnwantedColumns
from src.feature_engineering import HourFeature, ExtractingRollingFeatures
from src.train import ModelTrainer
from src.metrics import MedicalMetricsEvaluator

def main():
    print("1. Loading Raw Data:")
    df = DataLoader.load_clean()
    print(f"Data loaded successfully with shape: {df.shape}")

    print("\n 2. Extracting Temporal Cyclical Features:")
    df = HourFeature.transform_hour_column(df)

    print("\n 3. Setting Multi-Index & Computing Onset:")
    df = SepsisDataProcessor.SetMultIndexGroup(df)

    print("\n 4. Imputing Missing Values per Patient: ")
    df = SepsisDataProcessor.impute_missing_values(df)

    print("\n 5. Engineering Rolling & Delta Features (6h): ")
    df = ExtractingRollingFeatures.extracting_rolling_features(df)

    print("\n 6. Dropping Unwanted Metadata Columns: ")
    df = DropUnwantedColumns.drop_unwanted_columns(df)
    print(f"Final clean dataset shape for modeling: {df.shape}")

    print("\n 7. Running Stratified Group 5-Fold CV:")
    trainer = ModelTrainer(n_splits=5, random_state=42)
    X, y, folds, trained_models = trainer.run_cross_validation(df)

    print("\n:8. Evaluating Final Medical Metrics:")
    evaluator = MedicalMetricsEvaluator()
    final_results = evaluator.evaluate_cross_validation(
        X=X, 
        y=y, 
        folds=folds, 
        trained_models=trained_models, 
        threshold=0.3
    )

    print("\n FINAL MEDICAL EVALUATION ")
    for metric_name, score in final_results.items():
        print(f"{metric_name.upper().replace('_', ' ')}: {score:.4f}")

    print("\n 9. Saving Trained Models & Metadata:")
    models_dir = "models"
    os.makedirs(models_dir, exist_ok=True)

    for idx, model in enumerate(trained_models):
        model_path = os.path.join(models_dir, f"lgbm_fold_{idx + 1}.pkl")
        joblib.dump(model, model_path)
        print(f"Saved: {model_path}")

    metadata = {
        "best_threshold": final_results['best_threshold_found'],
        "feature_names": list(X.columns)
    }
    metadata_path = os.path.join(models_dir, "pipeline_metadata.pkl")
    joblib.dump(metadata, metadata_path)
    print(f"Saved Metadata: {metadata_path}")


if __name__ == "__main__":
    main()
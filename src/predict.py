import os
import sys
import joblib
import pandas as pd
import numpy as np

class SepsisPredictor:
    def __init__(self, models_dir="models"):
        self.models_dir = models_dir
        self.models = []
        self.best_threshold = 0.5
        self.feature_names = []
        self._load_pipeline()

    def _load_pipeline(self):
        # 1. Load Clinical Metadata
        metadata_path = os.path.join(self.models_dir, "pipeline_metadata.pkl")
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(f"Metadata not found at {metadata_path}. Please run training first.")
        
        metadata = joblib.load(metadata_path)
        self.best_threshold = metadata["best_threshold"]
        self.feature_names = metadata["feature_names"]

        # 2. Load the 5 Trained Fold Models
        for i in range(1, 6):
            model_path = os.path.join(self.models_dir, f"lgbm_fold_{i}.pkl")
            if os.path.exists(model_path):
                self.models.append(joblib.load(model_path))
            else:
                print(f"[Warning] Missing model for fold {i}")
                
        print(f"[Inference Pipeline] Loaded {len(self.models)} models. Optimal Threshold: {self.best_threshold:.4f}")

    def predict_probability(self, X_input: pd.DataFrame) -> float:
        # Align features to ensure exactly the same columns as training
        X_aligned = X_input[self.feature_names]
        
        # Collect predictions from all loaded fold models
        fold_probs = []
        for model in self.models:
            # Predict probability for class 1 (Sepsis)
            prob = model.predict_proba(X_aligned)[:, 1]
            fold_probs.append(prob)
            
        # Compute the ensemble average probability
        avg_prob = np.mean(fold_probs, axis=0)
        return avg_prob

    def predict_action(self, X_input: pd.DataFrame) -> dict:
        # Get ensemble probability
        probabilities = self.predict_probability(X_input)
        
        # Trigger alarm if probability >= optimal clinical threshold
        predictions = (probabilities >= self.best_threshold).astype(int)
        
        return {
            "sepsis_probability": probabilities,
            "trigger_alarm": predictions
        }

# --- تجربة سريعة للتأكد من أن كل شيء يعمل بسلاسة ---
if __name__ == "__main__":
    print("Testing Sepsis Predictor Pipeline...")
    try:
        predictor = SepsisPredictor()
        print("Pipeline works perfectly and is ready to process new patients!")
    except Exception as e:
        print(f"Error initializing predictor: {e}")
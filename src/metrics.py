import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, precision_recall_curve, auc

class MedicalMetricsEvaluator:
    def __init__(self):
        pass

    @staticmethod
    def calculate_metrics(y_true: np.ndarray, y_pred_proba: np.ndarray, threshold: float = 0.5) -> dict:
        y_pred_binary = (y_pred_proba >= threshold).astype(int)
        
        precision = precision_score(y_true, y_pred_binary, zero_division=0)
        recall = recall_score(y_true, y_pred_binary, zero_division=0)
        f1 = f1_score(y_true, y_pred_binary, zero_division=0)
        
        p, r, _ = precision_recall_curve(y_true, y_pred_proba)
        pr_auc = auc(r, p)
        
        return {
            "precision": float(precision),
            "recall": float(recall),
            "f1_score": float(f1),
            "pr_auc": float(pr_auc)
        }

    def evaluate_cross_validation(self, X, y, folds: list, trained_models: list, threshold: float = 0.5) -> dict:
        fold_results = []
        all_y_val = []
        all_y_pred_proba = []
        
        for fold_idx, (_, val_idx) in enumerate(folds):
            X_val = X.iloc[val_idx]
            y_val = y.iloc[val_idx].values
            
            model = trained_models[fold_idx]
            
            y_pred_proba = model.predict_proba(X_val)[:, 1]
            
            all_y_val.append(y_val)
            all_y_pred_proba.append(y_pred_proba)
            
            metrics = self.calculate_metrics(y_val, y_pred_proba, threshold=threshold)
            fold_results.append(metrics)
            
        avg_metrics = {}
        keys = fold_results[0].keys()
        for key in keys:
            avg_metrics[f"avg_{key}"] = float(np.mean([result[key] for result in fold_results]))
            
        concat_y_val = np.concatenate(all_y_val)
        concat_y_pred = np.concatenate(all_y_pred_proba)
        
        best_th = 0.5
        best_f1 = 0.0
        
        for th in np.linspace(0.01, 0.5, 100):
            preds = (concat_y_pred >= th).astype(int)
            f1 = f1_score(concat_y_val, preds, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_th = th
                
        optimal_metrics = self.calculate_metrics(concat_y_val, concat_y_pred, threshold=best_th)
        avg_metrics["best_threshold_found"] = float(best_th)
        avg_metrics["optimized_precision"] = optimal_metrics["precision"]
        avg_metrics["optimized_recall"] = optimal_metrics["recall"]
        avg_metrics["optimized_f1_score"] = optimal_metrics["f1_score"]
        
        return avg_metrics
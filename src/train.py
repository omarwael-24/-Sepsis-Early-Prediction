import pandas as pd
import numpy as np
from lightgbm import LGBMClassifier
from sklearn.model_selection import StratifiedGroupKFold

class GroupDataSplitter:
    def __init__(self, n_splits=5, shuffle=True, random_state=42):
        self.sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)

    def split_data(self, df: pd.DataFrame):
        df = df.dropna(subset=['SepsisLabel'])
        
        y = df['SepsisLabel'].astype(int)
        X = df.drop(columns=['SepsisLabel'])
        
        groups = df.index.get_level_values('Patient_ID')
        
        patient_labels = y.groupby(groups).max()
        y_stratify = groups.map(patient_labels).fillna(0).astype(int)
        
        folds = []
        for train_idx, test_idx in self.sgkf.split(X, y_stratify, groups=groups):
            folds.append((train_idx, test_idx))
            
        return X, y, folds


class ModelTrainer:
    def __init__(self, n_splits=5, random_state=42):
        self.splitter = GroupDataSplitter(n_splits=n_splits, random_state=random_state)
        self.random_state = random_state
        self.n_splits = n_splits

    def run_cross_validation(self, df: pd.DataFrame):
        X, y, folds = self.splitter.split_data(df)
        trained_models = []
        
        balanced_weight = 10.0
        print(f"[Trainer Info] Using Controlled Scale Pos Weight: {balanced_weight}")
        
        for fold_idx, (train_idx, val_idx) in enumerate(folds):
            print(f"--- Training Fold {fold_idx + 1}/{self.n_splits} ---")
            
            X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
            X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]
            
            model = LGBMClassifier(
                n_estimators=150,       
                learning_rate=0.05,      
                max_depth=4,             
                num_leaves=15,          
                min_child_samples=50,    
                scale_pos_weight=balanced_weight, 
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=self.random_state,
                n_jobs=-1,
                verbose=-1
            )
            
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                callbacks=[]
            )
            
            trained_models.append(model)
            
        return X, y, folds, trained_models
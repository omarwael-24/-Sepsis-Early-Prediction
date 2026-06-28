""" ==================== MODEL TRAINING & HYPERPARAMETER TUNING ==================== """
import pandas as pd

import lightgbm as lgb
import numpy as np

from sklearn.model_selection import StratifiedGroupKFold

class GroupDataSplitter:

    def __init__(self, n_split:int =5 ,shuffel:bool=True , randome_state : int =42 ):
        self.n_split=n_split
        self.shuffel=shuffel
        self.randome_state = randome_state

        self.sgkf=StratifiedGroupKFold(n_split=self.n_split, randome_state=self.randome_state ,shuffel=self.shuffel)
        
    
    def splitdata(self,df :pd.DataFrame):

        x= df.drop(columns=['SepsisLabel'])
        y=df['SepsisLabel']

        groups = df.index.get_level_values('Patient_ID')

        patient_labels = y.groupby(groups).max()
        
        y_stratify = groups.map(patient_labels)

        folds=[]

        for train_idx , test_idx in self.sgkf(x,y_stratify,groups=groups):
            folds.append((train_idx, test_idx))
        
        return x, y, folds

class SepsisModel:

    def __init__(self, scale_weight:float = 49.0):
        
        self.scale_weight=scale_weight

        self.params = {
            'objective': 'binary',
            'metric': 'binary_logloss',
            'boosting_type': 'gbdt',
            'scale_pos_weight': self.scale_weight,
            'learning_rate': 0.03,
            'num_leaves': 31,
            'max_depth': 6,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': -1,
            'random_state': 42
        }
        self.model = None

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series, X_val: pd.DataFrame, y_val: pd.Series):
        train_data = lgb.Dataset(X_train, label=y_train)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
        
        self.model = lgb.train(
            params=self.params,
            train_set=train_data,
            num_boost_round=1000,
            valid_sets=[train_data, val_data],
            callbacks=[
                lgb.early_stopping(stopping_rounds=50, verbose=False),
                lgb.log_evaluation(period=0)
            ]
        )
        return self
    
    def predict_proba(self,x:pd.DataFrame) -> np.ndarray:
        
        if self.model == None:
            raise ValueError("Model not trained yet.")
        else:
            return self.model.predict(x,num_iteration=self.model.best_iteration)
        
class Model_trainer:

    def __init__(self):
        self.speliter=GroupDataSplitter()

        self.trained_models=[]

    def run_cross_validation(self, df: pd.DataFrame):

        x , y, fold =self.speliter.splitdata(df)

        for fold_idx , (train_idx , val_idx) in enumerate(fold):
            
            print(f"Training Fold {fold_idx + 1}/{len(fold)}...")

            x_train , y_train = x.iloc(train_idx) , y.iloc(train_idx)

            x_val , y_val = x.iloc(val_idx) , y.iloc(val_idx)

            calculated_weight = (y_train == 0).sum() / (y_train == 1).sum()

            sepsis_model = SepsisModel(scale_pos_weight=calculated_weight)

            sepsis_model.fit(x_train, y_train, x_val, y_val)
            
            self.trained_models.append(sepsis_model)
            
        return x, y, fold, self.trained_models
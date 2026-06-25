""" ==================== MODEL TRAINING & HYPERPARAMETER TUNING ==================== """
import pandas as pd

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




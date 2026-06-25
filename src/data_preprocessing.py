import pandas as pd
import numpy as np 
from src import config


class DataLoader:
    def __init__(self):
        pass

    @staticmethod
    def load_clean() -> pd.DataFrame:
        df = pd.read_csv(config.Config.path_raw_data / 'Dataset.csv')
        return df
    

class DropUnwantedColumns:
    def __init__(self):
        pass

    @staticmethod
    def drop_unwanted_columns(df: pd.DataFrame) -> pd.DataFrame:
        unwanted_columns = ['Unit1', 'Unit2', 'HospAdmTime', 'Unnamed: 0'
                            , 'Hour', 'True_Hour']
        df = df.drop(columns=unwanted_columns, errors='ignore')
        return df
    

class SepsisDataProcessor:

    def __init__(self):
            pass
    
    @staticmethod
    def SetMultIndexGroup(df: pd.DataFrame) -> pd.DataFrame:
        df['Spesis_onset'] = df.groupby('Patient_ID')['SepsisLabel'].diff()
        df = df.set_index(['Patient_ID','ICULOS'])
        return df

    @staticmethod
    def impute_missing_values(df: pd.DataFrame) -> pd.DataFrame:
        features_to_impute = config.Config.VITAL_SIGNS + config.Config.LAB_VALUES
        
        for col in features_to_impute:
            df[f'{col}_is_missing'] = df[col].isna().astype(int)

        df[features_to_impute] = df.groupby(level='Patiend_ID')[features_to_impute].ffill().bfill()

        return df
    
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
        unwanted_columns = ['Unit1', 'Unit2', 'HospAdmTime', 'Unnamed: 0']
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
    
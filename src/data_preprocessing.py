""" ==================== DATA PREPROCESSING & CLEANING ==================== """

import pandas as pd
import numpy as np 
from src import config


class DataLoader:
    def __init__(self):
        pass

    @staticmethod
    def load_clean() -> pd.DataFrame:
        """
        Loads the raw sepsis dataset, drops irrelevant demographic features,
        and returns a cleaned DataFrame.
    
        Returns:
            pd.DataFrame: Cleaned dataset.
        """
        # 1. Load the dataset using path from config
        df = pd.read_csv(config.Config.path_raw_data / 'Dataset.csv')
        return df
    

class DropUnwantedColumns:
    def __init__(self):
        pass

    @staticmethod
    def drop_unwanted_columns() -> pd.DataFrame:
        # Load the data from the first class
        df = DataLoader.load_clean()
        
        # 2. Drop unwanted columns
        unwanted_columns = ['Unit1', 'Unit2', 'HospAdmTime', 'Unnamed: 0']
        df = df.drop(columns=unwanted_columns, errors='ignore')
        
        # Return the modified dataframe (Crucial step that was missing)
        return df

class SepsisDataProcessor:

    def __init__(self):
        pass

    @staticmethod
    def Transform_Hour_Column() -> pd.DataFrame:
        df = DropUnwantedColumns.drop_unwanted_columns()
        df['True_Hour'] = df['Hour'] % 24
        df['Hour_sin'] = np.sin(2 * np.pi * df['True_Hour'] / 24)
        df['Hour_cos'] = np.cos(2 * np.pi * df['True_Hour'] / 24)
        return df
    
    @staticmethod
    def SetMultIndexGroup():
    
        df=SepsisDataProcessor.Transform_Hour_Column()

        # 3. making a Spesis_onset column  
        df['Spesis_onset']=df.groupby('Patient_ID')['SepsisLabel'].diff()
        
        # 4. Set "Patiend_ID" , "ICULOS" as index [Multi_Index]
        df = df.set_index(['Patient_ID','ICULOS'])
        return df

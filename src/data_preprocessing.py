""" ==================== DATA PREPROCESSING & CLEANING ==================== """

import pandas as pd
import numpy as np 
from src import config


def Load_clean()-> pd.DataFrame:
    """
    Loads the raw sepsis dataset, drops irrelevant demographic features,
    and returns a cleaned DataFrame.
    
    Returns:
        pd.DataFrame: Cleaned dataset.
    """

    # 1. Load the dataset using path from config
    df=pd.read_csv(config.Config.path_raw_data/'Dataset.csv')
    
    # 2. Drop unwanted columns
    un_wanted_column=['Unit1', 'Unit2', 'HospAdmTime', 'Unnamed: 0']
    df=df.drop(columns=un_wanted_column)

    # 3. Set "Patiend_ID" , "ICULOS" as index [Multi_Index]
    df = df.set_index(['Patient_ID','ICULOS'])
    
    # 4. making a Spesis_onset column  
    df['Spesis_onset']=df.groupby('Patient_ID')['SepsisLabel'].diff()

    return df

""" ==================== TEMPORAL FEATURE ENGINEERING ==================== """
import pandas as pd 
import numpy as np 
from src import config


class HourFeature:
    def __init__(self):
        pass

    @staticmethod
    def Transform_Hour_Column(df: pd.DataFrame) -> pd.DataFrame:
        df['True_Hour'] = df['Hour'] % 24
        df['Hour_sin'] = np.sin(2 * np.pi * df['True_Hour'] / 24)
        df['Hour_cos'] = np.cos(2 * np.pi * df['True_Hour'] / 24)
        return df
    
class ExtractingRollingFeatures:

    def __init__(self):
        pass

    @staticmethod
    def extracting_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
        Vital_cols=config.Config.VITAL_SIGNS

        groupd=df.groupby(level='Patient_ID')[Vital_cols]

        df_rolling_mean= groupd.rolling(window=6,min_periods=1).mean().reset_index(level=0,drop=True)
        
        df_rolling_std= groupd.rolling(window=6,min_periods=1).std().fillna(0).reset_index(level=0,drop=True)

        for col in Vital_cols:
            df[f'{col}_roll_mean_6h'] = df_rolling_mean[col]
            df[f'{col}_roll_std_6h'] = df_rolling_std[col]

            df[f'{col}_delta_6h'] = df[col] - df[f'{col}_roll_mean_6h']
        
        df['qSOFA_score'] = ((df['Resp'] >= 22).astype(int) + (df['SBP'] <= 100).astype(int))
        df['Long_Stay_Danger'] = (df['ICULOS'] > 48).astype(int)

        return df
    
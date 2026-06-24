""" ==================== TEMPORAL FEATURE ENGINEERING ==================== """
import pandas as pd 
import numpy as np 

class HourFeature:
    def __init__(self):
        pass

    @staticmethod
    def Transform_Hour_Column(df: pd.DataFrame) -> pd.DataFrame:
        df['True_Hour'] = df['Hour'] % 24
        df['Hour_sin'] = np.sin(2 * np.pi * df['True_Hour'] / 24)
        df['Hour_cos'] = np.cos(2 * np.pi * df['True_Hour'] / 24)
        return df
import pandas as pd 
import numpy as np 
from src import config

class HourFeature:
    def __init__(self):
        pass

    @staticmethod
    def transform_hour_column(df: pd.DataFrame) -> pd.DataFrame:
        df['True_Hour'] = df['Hour'] % 24
        df['Hour_sin'] = np.sin(2 * np.pi * df['True_Hour'] / 24)
        df['Hour_cos'] = np.cos(2 * np.pi * df['True_Hour'] / 24)
        return df
    

class ExtractingRollingFeatures:
    def __init__(self):
        pass

    @staticmethod
    def extracting_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
        vital_cols = config.Config.VITAL_SIGNS
        
        # 1. Base Grouping
        grouped = df.groupby(level='Patient_ID')[vital_cols]
        
        df_rolling_mean = grouped.rolling(window=6, min_periods=1).mean().droplevel(0)
        df_rolling_std = grouped.rolling(window=6, min_periods=1).std().fillna(0).droplevel(0)
        
        # Calculate Volatility (Mean Absolute Successive Differences)
        df_diff_abs = df.groupby(level='Patient_ID')[vital_cols].diff().abs()
        df_volatility = df_diff_abs.groupby(level='Patient_ID').rolling(window=6, min_periods=1).mean().fillna(0).droplevel(0)
        
        # Get the first records of each patient as their medical baseline
        df_first = df.groupby(level='Patient_ID')[vital_cols].transform('first')
        
        new_features = {}
        
        # 2. Loop for Rolling, Delta, Volatility, and Baseline Ratios
        for col in vital_cols:
            new_features[f'{col}_roll_mean_6h'] = df_rolling_mean[col]
            new_features[f'{col}_roll_std_6h'] = df_rolling_std[col]
            new_features[f'{col}_delta_6h'] = df[col] - df_rolling_mean[col]
            new_features[f'{col}_volatility_6h'] = df_volatility[col]
            
            # Baseline Ratio
            new_features[f'{col}_baseline_ratio'] = df[col] / (df_first[col] + 1e-5)
            
            # Missingness Density
            if f'{col}_is_missing' in df.columns:
                new_features[f'{col}_missing_density_6h'] = (
                    df.groupby(level='Patient_ID')[f'{col}_is_missing']
                    .rolling(window=6, min_periods=1).mean().droplevel(0)
                )
        
        # 3. Clinical Scores & Flags
        new_features['qSOFA_score'] = ((df['Resp'] >= 22).astype(int) + (df['SBP'] <= 100).astype(int))
        iculos_values = df.index.get_level_values('ICULOS')
        new_features['Long_Stay_Danger'] = (iculos_values > 48).astype(int)
        new_features['Shock_Index'] = df['HR'] / (df['SBP'] + 1e-5)
        
        # SIRS Score
        temp_cond = ((df['Temp'] > 38) | (df['Temp'] < 36)).astype(int)
        hr_cond = (df['HR'] > 90).astype(int)
        resp_cond = (df['Resp'] > 20).astype(int)
        wbc_cond = ((df['WBC'] > 12) | (df['WBC'] < 4)).astype(int)
        new_features['SIRS_score'] = temp_cond + hr_cond + resp_cond + wbc_cond
        
        # 4. Cross-Sign Interactions & MAP Drop Counter
        new_features['Rate_Pressure_Product'] = df['HR'] * df['SBP']
        new_features['Respiratory_Efficiency'] = df['O2Sat'] / (df['Resp'] + 1e-5)
        
        # MAP Drop hours counter (How many hours MAP was below 65 in the last 6h)
        map_under_65 = (df['MAP'] < 65).astype(int)
        new_features['MAP_drop_hours_6h'] = map_under_65.groupby(level='Patient_ID').rolling(window=6, min_periods=1).sum().droplevel(0)
        
        # Cumulative HR Danger placeholder
        new_features['Cumulative_HR_Danger'] = df.groupby(level='Patient_ID')[vital_cols[0]].cumsum()
        
        # Concat everything beautifully
        new_features_df = pd.DataFrame(new_features, index=df.index)
        df = pd.concat([df, new_features_df], axis=1)
        
        return df
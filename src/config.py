""" ==================== CONFIGURATION & HYPERPARAMETERS ==================== """

from pathlib import Path

class Config:
    #sepsis_early folder
    path_root_dir=Path(__file__).resolve().parent.parent

    #src folder
    Path_dir_src= path_root_dir / "src"

    #raw data folder
    path_raw_data= path_root_dir / "data"/"raw"



    # =========================================================================
    # 2. FEATURE GROUP DEFINITIONS
    # =========================================================================
    VITAL_SIGNS = [
        'HR', 'O2Sat', 'Temp', 'SBP', 'MAP', 'DBP', 'Resp', 'EtCO2'
    ]
    
    LAB_VALUES = [
        'BaseExcess', 'HCO3', 'FiO2', 'pH', 'PaCO2', 'SaO2', 'AST', 'BUN', 
        'Alkalinephos', 'Calcium', 'Chloride', 'Creatinine', 'Bilirubin_direct', 
        'Glucose', 'Lactate', 'Magnesium', 'Phosphate', 'Potassium', 
        'Bilirubin_total', 'TroponinI', 'Hct', 'Hgb', 'PTT', 'WBC', 
        'Fibrinogen', 'Platelets'
    ]
    
    DEMOGRAPHICS = [
        'Age', 'Gender', 'Unit1', 'Unit2', 'HospAdmTime', 'ICULOS'
    ]
    
    TARGET = 'SepsisLabel'
    PATIENT_ID = 'Patient_ID'









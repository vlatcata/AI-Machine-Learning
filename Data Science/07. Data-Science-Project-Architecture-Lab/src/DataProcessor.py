import pandas as pd
from sklearn import preprocessing

# The first notebook is Data Exploration, nothing to extract here that is needed.

# Third notebook is Data Analysis so nothing to extract here either.

CATEGORY_MAPPINGS = {
    'Gender': {0: 'Male', 1: 'Female'},
    'Ethnicity': {0: 'Caucasian', 1: 'African American', 2: 'Asian', 3: 'Other'},
    'EducationLevel': {0: 'None', 1: 'High School', 2: "Bachelor's", 3: 'Higher'}
    }

INVERSE_CATEGORY_MAPPINGS = { 
            'Gender': {'Male': 0, 'Female': 1},
            'Ethnicity': {'Caucasian': 0, 'African American': 1, 'Asian': 2, 'Other': 3},
            'EducationLevel': {'None': 0, 'High School': 1, "Bachelor's": 2, 'Higher': 3}
        }

# From notebook 2 I will extract the preprocessing steps into one function.
def preprocess_data(astma: pd.DataFrame, drop_cols=None, category_mappings=None):
    """Data Cleaning and Preprocessing operations: Drop unnecessary columns and map categorical columns from codes to strings.

    Args:
        astma (pd.DataFrame): Input dataframe containing data.

    Returns:
        pd.DataFrame: Preprocessed dataframe.
    """

    astma = astma.copy()
        
    # Initialize constants
    try:
        if drop_cols is None:
            drop_cols = ["DoctorInCharge", "PatientID"]
        if category_mappings is None:
            category_mappings = CATEGORY_MAPPINGS
    except Exception as e:
        print(f"Error initializing constants: {e}")
        raise e

    # Drop specified columns
    astma.drop(columns=drop_cols, inplace=True)

    # Map numeric values to categorical
    try:
        for col, mapping in category_mappings.items():
            if col in astma.columns:
                astma[col] = astma[col].astype('category').map(mapping)
    except Exception as e:
        print(f"Error mapping categorical columns: {e}")
        raise e

    return astma

# From notebook 4 I will extract everything into one function and maybe rewrite some code a little to follow python's best practices.
def manipulate_features(astma: pd.DataFrame, inverse_mappings=None, lifestyle_cols=None, lung_function_cols=None, exposure_cols=None):
    """Feature Manipulation: Encode categorical columns to numeric, engineer features, and scale selected columns.

    Args:
        astma (pd.DataFrame): Input dataframe containing data.

    Returns:
        pd.DataFrame: Processed dataframe with manipulated features.
    """ 
    
    def _engineer_features(astma: pd.DataFrame):
        """ Feature engineering operations """
        try:
            astma['TotalExposure'] = astma[['PollenExposure', 'PollutionExposure', 'DustExposure']].sum(axis=1)
            astma['MedicalComplicationsCount'] = astma[['FamilyHistoryAsthma', 'HistoryOfAllergies', 'Eczema', 'HayFever', 'GastroesophagealReflux']].sum(axis=1)
            astma['SymptomsCount'] = astma[['Wheezing', 'ShortnessOfBreath', 'ChestTightness', 'Coughing', 'NighttimeSymptoms']].sum(axis=1)
        except Exception as e:
            print(f"Error during feature engineering: {e}")
            raise e
            
        return astma

    def _normalize_columns(astma: pd.DataFrame, lifestyle_cols: list, lung_function_cols: list, exposure_cols: list):
        """ Normalization operations """
        if lifestyle_cols is None:
            lifestyle_cols = ['BMI', 'Smoking', 'PhysicalActivity', 'DietQuality', 'SleepQuality']
        if lung_function_cols is None:
            lung_function_cols = ['LungFunctionFEV1', 'LungFunctionFVC', 'TotalExposure']
        if exposure_cols is None:
            exposure_cols = ['PollutionExposure', 'PollenExposure', 'DustExposure']

        scaler = preprocessing.MinMaxScaler()

        # Normalize all columns
        try:
            astma[lifestyle_cols] = scaler.fit_transform(astma[lifestyle_cols])
            astma['LifestyleScore'] = astma[lifestyle_cols].mean(axis=1)

            astma[lung_function_cols] = scaler.fit_transform(astma[lung_function_cols])
            astma['LungFunctionScore'] = astma[lung_function_cols].mean(axis=1)

            astma[exposure_cols] = scaler.fit_transform(astma[exposure_cols])
            astma['EnvironmentalExposure'] = astma[exposure_cols].mean(axis=1)
        except Exception as e:
            print(f"Error during normalization: {e}")
            raise e

        return astma

    astma = astma.copy()
        
    # Inverse map the categorical columns
    if inverse_mappings is None:
        inverse_mappings = INVERSE_CATEGORY_MAPPINGS

    # Encode categorical columns
    try:
        for col, mapping in inverse_mappings.items():
            if col in astma.columns:
                astma[col] = astma[col].map(mapping).astype('Int64')
    except Exception as e:
        print(f"Error encoding categorical columns: {e}")
        raise e

    # Convert all int64 columns to float64
    try:
        int_cols = astma.select_dtypes(include=['int64']).columns
        astma[int_cols] = astma[int_cols].astype('float64')
    except Exception as e:
        print(f"Error converting int64 columns to float64: {e}")
        raise e

    # Feature engineering
    astma = _engineer_features(astma)

    # Normalize columns
    astma = _normalize_columns(astma, lifestyle_cols, lung_function_cols, exposure_cols)

    return astma
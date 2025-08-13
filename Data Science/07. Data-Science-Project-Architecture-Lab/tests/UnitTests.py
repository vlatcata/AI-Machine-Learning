import unittest
import pandas as pd
import numpy as np
from src.DataProcessor import preprocess_data, manipulate_features, CATEGORY_MAPPINGS, INVERSE_CATEGORY_MAPPINGS

class TestDataProcessor(unittest.TestCase):

    def setUp(self):
        # Sample DataFrame for testing
        self.df = pd.read_csv('tests/mock_asthma_disease_data.csv')
        self.df_cleaned = pd.read_csv('tests/mock_asthma_disease_data_cleaned.csv')
        
    # ---------- Preprocess Data Tests ----------

    def test_preprocess_data_drops_columns(self):
        result = preprocess_data(self.df, drop_cols=["DoctorInCharge", "PatientID"])
        
        self.assertNotIn("DoctorInCharge", result.columns)
        self.assertNotIn("PatientID", result.columns)      
        
    def test_preprocess_data_maps_categories(self):
        result = preprocess_data(self.df)
        
        self.assertEqual(result["Gender"].iloc[0], "Male")
        self.assertEqual(result["Ethnicity"].iloc[1], "Asian")
        self.assertEqual(result["EducationLevel"].iloc[0], "None")
        
    def test_process_data_throws_with_bad_mapper(self):
        BAD_CATEGORY_MAPPING = {
        'Gender': {0: {}, 1: {}}
        }

        self.assertRaises(Exception, preprocess_data, self.df, None, BAD_CATEGORY_MAPPING)
        
    # ---------- Manipulate Features Tests ----------

    def test_manipulate_features_normalizes_columns(self):
        result = manipulate_features(self.df_cleaned)

        self.assertEqual(result["BMI"].dtype, np.float64)
        self.assertEqual(result["LungFunctionFEV1"].dtype, np.float64)
        self.assertEqual(result["PollutionExposure"].dtype, np.float64)

    def test_manipulate_features_creates_new_features(self):
        result = manipulate_features(self.df_cleaned)

        self.assertIn("LifestyleScore", result.columns)
        self.assertIn("LungFunctionScore", result.columns)
        self.assertIn("EnvironmentalExposure", result.columns)

    def test_manipulate_features_inverse_maps_categorical_columns(self):
        result = manipulate_features(self.df_cleaned)

        self.assertEqual(result["Gender"].dtype, np.float64)
        self.assertEqual(result["Ethnicity"].dtype, np.float64)
        self.assertEqual(result["EducationLevel"].dtype, np.float64)

    def test_manipulate_features_converts_all_cols_to_float64(self):
        result = manipulate_features(self.df_cleaned)

        for col in result.columns:
            self.assertEqual(result[col].dtype, np.float64)

    def test_manipulate_features_adds_engineered_columns(self):
        result = manipulate_features(self.df_cleaned)

        self.assertIn("TotalExposure", result.columns)
        self.assertIn("MedicalComplicationsCount", result.columns)
        self.assertIn("SymptomsCount", result.columns)
        
    def test_manipulate_features_throws_with_bad_mapper(self):
        BAD_INVERSE_MAPPING = {
            'Gender': {'Male': 'asdasd'}
        }

        self.assertRaises(Exception, manipulate_features, self.df_cleaned, inverse_mappings=BAD_INVERSE_MAPPING)

if __name__ == '__main__':
    unittest.main(verbosity=2) # So more information is displayed on result.
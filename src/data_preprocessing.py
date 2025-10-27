"""
Hospital Readmission Prediction - Data Preprocessing Module

This module handles comprehensive data cleaning, feature engineering, and preparation
for machine learning models to predict hospital readmission risk.

Author: [Your Name]
Date: 2024
"""

import pandas as pd
import numpy as np
import logging
from typing import Tuple, List, Dict
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class HospitalDataPreprocessor:
    """
    Comprehensive data preprocessing pipeline for hospital readmission prediction.
    
    This class handles:
    - Data cleaning and missing value treatment
    - Feature engineering and transformation
    - Categorical encoding
    - Data splitting and scaling
    """
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_names = None
        self.target_column = 'readmitted'
        
    def load_data(self, file_path: str) -> pd.DataFrame:
        """Load the raw diabetic dataset."""
        logger.info(f"Loading data from {file_path}")
        try:
            data = pd.read_csv(file_path)
            logger.info(f"Data loaded successfully. Shape: {data.shape}")
            return data
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise
    
    def clean_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Comprehensive data cleaning pipeline.
        
        Args:
            data: Raw dataset
            
        Returns:
            Cleaned dataset
        """
        logger.info("Starting data cleaning process...")
        df = data.copy()
        
        # Remove irrelevant columns
        columns_to_drop = ['encounter_id', 'patient_nbr', 'weight', 'medical_specialty', 'payer_code']
        df = df.drop(columns=columns_to_drop, errors='ignore')
        
        # Handle missing values in categorical columns
        categorical_columns = df.select_dtypes(include=['object']).columns
        
        for col in categorical_columns:
            if col != 'readmitted':  # Don't modify target variable
                # Replace '?' with mode or 'Unknown'
                mode_value = df[col].mode()[0] if not df[col].mode().empty else 'Unknown'
                df[col] = df[col].replace('?', mode_value)
        
        # Remove patients who expired (discharge_disposition_id = 11)
        df = df[df['discharge_disposition_id'] != 11]
        
        # Remove patients with unknown gender
        df = df[df['gender'] != 'Unknown/Invalid']
        
        logger.info(f"Data cleaning completed. New shape: {df.shape}")
        return df
    
    def engineer_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Create new features for better model performance.
        
        Args:
            data: Cleaned dataset
            
        Returns:
            Dataset with engineered features
        """
        logger.info("Starting feature engineering...")
        df = data.copy()
        
        # Create patient service utilization feature
        df['patient_service_utilization'] = (
            df['number_outpatient'] + 
            df['number_emergency'] + 
            df['number_inpatient']
        )
        
        # Create medication change indicator
        medication_columns = [
            'metformin', 'repaglinide', 'nateglinide', 'chlorpropamide', 
            'glimepiride', 'acetohexamide', 'glipizide', 'glyburide', 
            'tolbutamide', 'pioglitazone', 'rosiglitazone', 'acarbose', 
            'miglitol', 'troglitazone', 'tolazamide', 'insulin', 
            'glyburide-metformin', 'glipizide-metformin', 
            'glimepiride-pioglitazone', 'metformin-rosiglitazone', 
            'metformin-pioglitazone'
        ]
        
        # Count medication changes
        df['medication_changes'] = 0
        for med in medication_columns:
            if med in df.columns:
                df['medication_changes'] += df[med].apply(
                    lambda x: 1 if x in ['Up', 'Down'] else 0
                )
        
        # Count total medications
        df['total_medications'] = 0
        for med in medication_columns:
            if med in df.columns:
                df['total_medications'] += df[med].apply(
                    lambda x: 1 if x in ['Steady', 'Up', 'Down'] else 0
                )
        
        # Create diagnosis categories
        df = self._categorize_diagnoses(df)
        
        # Recode age groups to numeric
        age_mapping = {
            '[0-10)': 5, '[10-20)': 15, '[20-30)': 25, '[30-40)': 35,
            '[40-50)': 45, '[50-60)': 55, '[60-70)': 65, '[70-80)': 75,
            '[80-90)': 85, '[90-100)': 95
        }
        df['age_numeric'] = df['age'].map(age_mapping)
        
        # Create binary features
        df['gender_binary'] = df['gender'].map({'Male': 1, 'Female': 0})
        df['change_binary'] = df['change'].map({'Ch': 1, 'No': 0})
        df['diabetes_med_binary'] = df['diabetesMed'].map({'Yes': 1, 'No': 0})
        
        logger.info("Feature engineering completed")
        return df
    
    def _categorize_diagnoses(self, df: pd.DataFrame) -> pd.DataFrame:
        """Categorize ICD-9 diagnosis codes into meaningful groups."""
        
        def categorize_diag(diag_code):
            if pd.isna(diag_code) or diag_code == '?':
                return 0
            
            try:
                code = float(diag_code)
            except:
                return 0
            
            # Cardiovascular diseases
            if (390 <= code < 460) or (code == 785):
                return 1
            # Respiratory diseases
            elif (460 <= code < 520) or (code == 786):
                return 2
            # Digestive diseases
            elif (520 <= code < 580) or (code == 787):
                return 3
            # Diabetes
            elif code == 250:
                return 4
            # Injury and poisoning
            elif 800 <= code < 1000:
                return 5
            # Musculoskeletal diseases
            elif 710 <= code < 740:
                return 6
            # Genitourinary diseases
            elif (580 <= code < 630) or (code == 788):
                return 7
            # Neoplasms
            elif 140 <= code < 240:
                return 8
            else:
                return 0
        
        # Apply categorization to all diagnosis columns
        for col in ['diag_1', 'diag_2', 'diag_3']:
            if col in df.columns:
                df[f'{col}_category'] = df[col].apply(categorize_diag)
        
        return df
    
    def encode_categorical_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Encode categorical features using appropriate methods.
        
        Args:
            data: Dataset with engineered features
            
        Returns:
            Dataset with encoded features
        """
        logger.info("Starting categorical encoding...")
        df = data.copy()
        
        # First, encode medication columns to numeric
        medication_columns = [
            'metformin', 'repaglinide', 'nateglinide', 'chlorpropamide', 
            'glimepiride', 'acetohexamide', 'glipizide', 'glyburide', 
            'tolbutamide', 'pioglitazone', 'rosiglitazone', 'acarbose', 
            'miglitol', 'troglitazone', 'tolazamide', 'insulin', 
            'glyburide-metformin', 'glipizide-metformin', 
            'glimepiride-pioglitazone', 'metformin-rosiglitazone', 
            'metformin-pioglitazone'
        ]
        
        for col in medication_columns:
            if col in df.columns:
                df[col] = df[col].map({'No': 0, 'Steady': 1, 'Up': 1, 'Down': 1})
        
        # Define categorical columns to encode
        categorical_columns = [
            'race', 'admission_type_id', 'discharge_disposition_id', 
            'admission_source_id', 'max_glu_serum', 'A1Cresult'
        ]
        
        # Create dummy variables for categorical features
        for col in categorical_columns:
            if col in df.columns:
                dummies = pd.get_dummies(df[col], prefix=col, drop_first=True)
                df = pd.concat([df, dummies], axis=1)
                df = df.drop(columns=[col])
        
        # Drop original diagnosis columns (keep categorized versions)
        diagnosis_cols = ['diag_1', 'diag_2', 'diag_3']
        for col in diagnosis_cols:
            if col in df.columns:
                df = df.drop(columns=[col])
        
        # Drop other original categorical columns
        columns_to_drop = ['age', 'gender', 'change', 'diabetesMed']
        for col in columns_to_drop:
            if col in df.columns:
                df = df.drop(columns=[col])
        
        # Ensure all columns are numeric
        for col in df.columns:
            if col != 'readmitted':
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Fill any NaN values with 0
        df = df.fillna(0)
        
        logger.info("Categorical encoding completed")
        return df
    
    def prepare_target_variable(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare the target variable for binary classification.
        
        Args:
            data: Dataset with encoded features
            
        Returns:
            Dataset with binary target variable
        """
        logger.info("Preparing target variable...")
        df = data.copy()
        
        # Convert readmission to binary: 1 if <30 days, 0 otherwise
        df[self.target_column] = df[self.target_column].map({
            '<30': 1,
            '>30': 0,
            'NO': 0
        })
        
        logger.info(f"Target variable distribution: {df[self.target_column].value_counts().to_dict()}")
        return df
    
    def split_and_scale_data(self, data: pd.DataFrame, test_size: float = 0.2, 
                           random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, 
                                                           pd.Series, pd.Series]:
        """
        Split data into train/test sets and scale features.
        
        Args:
            data: Preprocessed dataset
            test_size: Proportion of data for testing
            random_state: Random seed for reproducibility
            
        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        logger.info("Splitting and scaling data...")
        
        # Separate features and target
        X = data.drop(columns=[self.target_column])
        y = data[self.target_column]
        
        # Store feature names
        self.feature_names = X.columns.tolist()
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, 
            stratify=y
        )
        
        # Scale features
        X_train_scaled = pd.DataFrame(
            self.scaler.fit_transform(X_train),
            columns=X_train.columns,
            index=X_train.index
        )
        
        X_test_scaled = pd.DataFrame(
            self.scaler.transform(X_test),
            columns=X_test.columns,
            index=X_test.index
        )
        
        logger.info(f"Data split completed. Train: {X_train_scaled.shape}, Test: {X_test_scaled.shape}")
        
        return X_train_scaled, X_test_scaled, y_train, y_test
    
    def process_pipeline(self, file_path: str) -> Tuple[pd.DataFrame, pd.DataFrame, 
                                                       pd.Series, pd.Series]:
        """
        Complete data processing pipeline.
        
        Args:
            file_path: Path to raw data file
            
        Returns:
            Tuple of processed train/test data
        """
        logger.info("Starting complete data processing pipeline...")
        
        # Load and process data
        data = self.load_data(file_path)
        data = self.clean_data(data)
        data = self.engineer_features(data)
        data = self.encode_categorical_features(data)
        data = self.prepare_target_variable(data)
        
        # Split and scale
        X_train, X_test, y_train, y_test = self.split_and_scale_data(data)
        
        logger.info("Data processing pipeline completed successfully!")
        
        return X_train, X_test, y_train, y_test
    
    def get_feature_names(self) -> List[str]:
        """Get the list of feature names."""
        return self.feature_names
    
    def save_processed_data(self, X_train: pd.DataFrame, X_test: pd.DataFrame,
                          y_train: pd.Series, y_test: pd.Series, 
                          output_dir: str = 'data'):
        """Save processed data to files."""
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        X_train.to_csv(f'{output_dir}/X_train.csv', index=False)
        X_test.to_csv(f'{output_dir}/X_test.csv', index=False)
        y_train.to_csv(f'{output_dir}/y_train.csv', index=False)
        y_test.to_csv(f'{output_dir}/y_test.csv', index=False)
        
        logger.info(f"Processed data saved to {output_dir}/")


def main():
    """Main function to run the preprocessing pipeline."""
    preprocessor = HospitalDataPreprocessor()
    
    # Process the data
    X_train, X_test, y_train, y_test = preprocessor.process_pipeline('data/raw_diabetic_data.csv')
    
    # Save processed data
    preprocessor.save_processed_data(X_train, X_test, y_train, y_test)
    
    print("Data preprocessing completed successfully!")
    print(f"Training set shape: {X_train.shape}")
    print(f"Test set shape: {X_test.shape}")
    print(f"Feature names: {len(preprocessor.get_feature_names())} features")


if __name__ == "__main__":
    main()

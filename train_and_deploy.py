"""
Script to train the model for deployment
Run this once to create the necessary model files
"""

import os
from src.data_preprocessing import HospitalDataPreprocessor

def main():
    print("Training Hospital Readmission Prediction Model...")
    
    # Initialize preprocessor
    preprocessor = HospitalDataPreprocessor()
    
    # Process the pipeline
    print("\n1. Loading and preprocessing data...")
    X_train, X_test, y_train, y_test = preprocessor.process_pipeline('data/raw_diabetic_data.csv')
    
    print("\n2. Training Random Forest model...")
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import classification_report, accuracy_score
    from imblearn.over_sampling import SMOTE
    import joblib
    
    # Apply SMOTE for class balancing
    smote = SMOTE(random_state=42)
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
    
    # Train model
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=20,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train_balanced, y_train_balanced)
    
    print("\n3. Evaluating model...")
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    print(f"Model Accuracy: {accuracy:.4f}")
    
    print("\n4. Saving model and scaler...")
    os.makedirs('models', exist_ok=True)
    joblib.dump(model, 'models/best_model.joblib')
    joblib.dump(preprocessor.scaler, 'models/scaler.joblib')
    
    # Save feature names
    feature_names = preprocessor.get_feature_names()
    import pandas as pd
    pd.DataFrame({'feature': feature_names}).to_csv('data/feature_names.csv', index=False)
    
    print("\nModel training complete! Ready for deployment.")
    print(f"   Model saved: models/best_model.joblib")
    print(f"   Scaler saved: models/scaler.joblib")
    print(f"   Features saved: data/feature_names.csv")

if __name__ == '__main__':
    main()

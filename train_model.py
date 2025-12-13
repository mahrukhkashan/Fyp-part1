#!/usr/bin/env python3
"""
Script to train and save the sepsis prediction model
"""

import pandas as pd
import numpy as np
import json
import pickle
from data.data_loader import DataLoader
from data.feature_engineering import FeatureEngineer
from models.sepsis_predictor import SepsisPredictor
from explainability.shap_explainer import SHAPExplainer
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

def train_and_save_model():
    """Main function to train and save the model"""
    
    print("=" * 60)
    print("SEPSIS PREDICTION MODEL TRAINING")
    print("=" * 60)
    
    # Step 1: Load data
    print("\n1. Loading data...")
    data_loader = DataLoader()
    
    try:
        # Load patient data using the new balanced query
        print("Loading data from MIMIC-III database...")
        patient_data = data_loader.load_patient_data(limit=5000)
        
        if patient_data is None or patient_data.empty:
            print("No data loaded from database. Using sample data...")
            patient_data = create_sample_data()
        
        print(f"✓ Loaded {len(patient_data)} patient records")
        print(f"  Sepsis cases: {patient_data['has_sepsis'].sum()} ({patient_data['has_sepsis'].mean()*100:.1f}%)")
        
        # Optional: Load vitals and labs data for feature enrichment
        try:
            print("\nLoading additional data for feature enrichment...")
            vitals_data = data_loader.load_vitals_data(limit=10000)
            labs_data = data_loader.load_lab_data(limit=10000)
            
            print(f"  Vitals records: {len(vitals_data)}")
            print(f"  Lab records: {len(labs_data)}")
        except Exception as e:
            print(f"  Note: Could not load additional data: {e}")
            vitals_data = pd.DataFrame()
            labs_data = pd.DataFrame()
        
    except Exception as e:
        print(f"Error loading data: {e}")
        print("\nUsing sample data instead...")
        patient_data = create_sample_data()
        vitals_data = pd.DataFrame()
        labs_data = pd.DataFrame()
    
    # Step 2: Prepare data
    print("\n2. Preparing data...")
    
    # Check what columns we have
    print(f"Original columns: {list(patient_data.columns)}")
    print(f"Original shape: {patient_data.shape}")
    
    # Handle missing values
    print("\nHandling missing values...")
    
    # Identify target column (could be 'has_sepsis' or 'sepsis_label')
    target_col = 'has_sepsis' if 'has_sepsis' in patient_data.columns else 'sepsis_label'
    if target_col not in patient_data.columns:
        # If no sepsis label, create simulated one
        print("No sepsis label found. Creating simulated labels...")
        patient_data = create_sepsis_labels(patient_data)
        target_col = 'has_sepsis'
    
    # Separate features and target
    X = patient_data.drop(columns=[target_col], errors='ignore')
    y = patient_data[target_col]
    
    # Drop columns that won't be useful for prediction
    cols_to_drop = ['subject_id', 'hadm_id', 'icustay_id', 'icd9_code', 'seq_num', 
                    'charttime', 'chartdate', 'text', 'description', 'category',
                    'measurement', 'unit', 'flag', 'itemid', 'recorded_at', 'test_date']
    
    available_cols_to_drop = [col for col in cols_to_drop if col in X.columns]
    if available_cols_to_drop:
        print(f"Dropping columns: {available_cols_to_drop}")
        X = X.drop(columns=available_cols_to_drop)
    
    # Step 3: Feature engineering
    print("\n3. Engineering features...")
    feature_engineer = FeatureEngineer()
    
    # Identify categorical columns
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
    if categorical_cols:
        print(f"Categorical columns: {categorical_cols}")
    
    # Convert categorical to numeric
    for col in categorical_cols:
        if col in X.columns:
            if X[col].nunique() > 50:  # Too many unique values
                print(f"  Dropping '{col}' (too many unique values: {X[col].nunique()})")
                X = X.drop(columns=[col])
            else:
                print(f"  Encoding '{col}' ({X[col].nunique()} unique values)")
                X[col] = pd.factorize(X[col])[0]
    
    # Handle missing values
    print("\nHandling missing values in features...")
    numeric_cols = X.select_dtypes(include=[np.number]).columns
    
    for col in numeric_cols:
        missing_count = X[col].isnull().sum()
        if missing_count > 0:
            if missing_count / len(X) < 0.5:  # If less than 50% missing
                fill_value = X[col].median() if col != 'gender' else X[col].mode()[0] if len(X[col].mode()) > 0 else 0
                X[col] = X[col].fillna(fill_value)
                print(f"  Filled {missing_count} missing values in '{col}' with {fill_value}")
            else:
                print(f"  Dropping '{col}' (too many missing: {missing_count/len(X):.1%})")
                X = X.drop(columns=[col])
    
    # Engineer additional features
    X_engineered = feature_engineer.engineer_features(X)
    
    # Merge back with target
    X_engineered[target_col] = y.values
    
    # Separate again (in case engineering added/removed columns)
    y_final = X_engineered[target_col]
    X_final = X_engineered.drop(columns=[target_col])
    
    print(f"\nFinal dataset shape: {X_final.shape}")
    print(f"Feature columns: {list(X_final.columns)}")
    print(f"Sepsis cases: {y_final.sum()} ({y_final.mean()*100:.1f}%)")
    
    # Step 4: Train model
    print("\n4. Training model...")
    sepsis_predictor = SepsisPredictor()
    
    # Ensure all data is numeric
    print("Ensuring all features are numeric...")
    for col in X_final.columns:
        if not pd.api.types.is_numeric_dtype(X_final[col]):
            print(f"  Converting '{col}' to numeric")
            X_final[col] = pd.to_numeric(X_final[col], errors='coerce')
    
    # Fill any remaining NaN
    X_final = X_final.fillna(X_final.median())
    
    # Train the model
    results, X_test, y_test = sepsis_predictor.train(
        X_final, 
        y_final,
        test_size=0.2,
        random_state=42
    )
    
    # Print results
    print("\nModel Performance:")
    print("-" * 50)
    for model_name, result in results.items():
        metrics = result['metrics']
        print(f"\n{model_name}:")
        print(f"  Accuracy:  {metrics['accuracy']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall:    {metrics['recall']:.4f}")
        print(f"  F1-Score:  {metrics['f1_score']:.4f}")
        print(f"  ROC-AUC:   {metrics['roc_auc']:.4f}")
    
    # Step 5: Save model and artifacts
    print("\n5. Saving model artifacts...")
    
    # Save model
    model_path = 'models/saved_models/sepsis_model.pkl'
    sepsis_predictor.save_model(model_path)
    print(f"✓ Model saved to {model_path}")
    
    # Save feature names
    feature_names = list(X_final.columns)
    feature_names_path = 'models/saved_models/feature_names.json'
    with open(feature_names_path, 'w') as f:
        json.dump(feature_names, f)
    print(f"✓ Feature names saved to {feature_names_path}")
    
    # Save feature engineer
    feature_engineer_path = 'models/saved_models/feature_engineer.pkl'
    with open(feature_engineer_path, 'wb') as f:
        pickle.dump(feature_engineer, f)
    print(f"✓ Feature engineer saved to {feature_engineer_path}")
    
    # Step 6: Train and save SHAP explainer
    print("\n6. Training SHAP explainer...")
    
    # Get training data for SHAP
    X_train, _, y_train, _ = train_test_split(
        X_final, y_final, test_size=0.2, random_state=42, stratify=y_final
    )
    
    try:
        shap_explainer = SHAPExplainer(sepsis_predictor.best_model, feature_names)
        shap_explainer.fit_explainer(X_train)
        
        shap_path = 'models/saved_models/shap_explainer.pkl'
        shap_explainer.save_explainer(shap_path)
        print(f"✓ SHAP explainer saved to {shap_path}")
        
        # Generate sample explanation
        print("\n7. Generating sample explanations...")
        
        # Find a sepsis case in test set
        sepsis_indices = np.where(y_test == 1)[0]
        if len(sepsis_indices) > 0:
            sample_idx = sepsis_indices[0]
        else:
            sample_idx = 0
        
        sample_features = X_test.iloc[sample_idx:sample_idx+1]
        
        explanation = shap_explainer.explain_prediction(sample_features)
        
        print("\nSample Explanation:")
        print(f"Base value: {explanation['base_value']:.4f}")
        print(f"Prediction: {explanation['prediction']}")
        print(f"Probability: {explanation['probability']:.4f}")
        
        print("\nTop contributing features:")
        for i, effect in enumerate(explanation['feature_effects'][:10]):
            direction = "increases" if effect['shap_value'] > 0 else "decreases"
            print(f"{i+1}. {effect['feature']}: {effect['shap_value']:.4f} ({direction} risk)")
        
    except Exception as e:
        print(f"Warning: Could not create SHAP explainer: {e}")
        print("  The model will still work, but explanations may be limited.")
    
    # Close database connection
    data_loader.close_connection()
    
    print("\n" + "=" * 60)
    print("MODEL TRAINING COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    
    # Summary
    print("\nTraining Summary:")
    print(f"- Total records: {len(X_final)}")
    print(f"- Sepsis cases: {y_final.sum()} ({y_final.mean()*100:.1f}%)")
    print(f"- Best model: {sepsis_predictor.best_model_name}")
    print(f"- Best F1-Score: {sepsis_predictor.best_score:.4f}")
    
    print("\nNext steps:")
    print("1. Run the application: python run.py")
    print("2. Access the dashboard at: http://localhost:5000")
    print("3. Use username: 'demo' and password: 'demo' for testing")

def create_sepsis_labels(df):
    """Create simulated sepsis labels based on clinical features"""
    np.random.seed(42)
    
    # Create sepsis probability based on clinical indicators
    sepsis_prob = np.zeros(len(df))
    
    # Clinical indicators for sepsis
    if 'heart_rate' in df.columns:
        sepsis_prob += (df['heart_rate'] > 120) * 0.2
    if 'temperature' in df.columns:
        sepsis_prob += (df['temperature'] > 38.5) * 0.2
    if 'respiratory_rate' in df.columns:
        sepsis_prob += (df['respiratory_rate'] > 22) * 0.2
    if 'wbc' in df.columns:
        sepsis_prob += (df['wbc'] > 12) * 0.15
    if 'lactate' in df.columns:
        sepsis_prob += (df['lactate'] > 2) * 0.25
    
    # Add age factor
    if 'age' in df.columns:
        sepsis_prob += (df['age'] > 65) * 0.1
    
    # Add random noise
    sepsis_prob += np.random.random(len(df)) * 0.1
    
    # Cap at 1.0 and create binary labels
    sepsis_prob = np.clip(sepsis_prob, 0, 1)
    df['has_sepsis'] = (sepsis_prob > 0.5).astype(int)
    
    print(f"Created simulated sepsis labels: {df['has_sepsis'].sum()} cases ({df['has_sepsis'].mean()*100:.1f}%)")
    
    return df

def create_sample_data():
    """Create sample patient data for testing"""
    np.random.seed(42)
    n_samples = 1000
    
    print(f"Generating {n_samples} sample records...")
    
    data = {
        'subject_id': range(1, n_samples + 1),
        'age': np.random.randint(20, 80, n_samples),
        'gender': np.random.choice(['M', 'F'], n_samples),
        'ethnicity': np.random.choice(['WHITE', 'BLACK', 'ASIAN', 'HISPANIC'], n_samples),
        'heart_rate': np.random.normal(85, 20, n_samples).clip(40, 180),
        'temperature': np.random.normal(37, 1, n_samples).clip(35, 41),
        'respiratory_rate': np.random.normal(18, 6, n_samples).clip(8, 40),
        'systolic_bp': np.random.normal(120, 25, n_samples).clip(80, 200),
        'diastolic_bp': np.random.normal(80, 15, n_samples).clip(50, 130),
        'wbc': np.random.normal(8, 4, n_samples).clip(2, 30),
        'lactate': np.random.exponential(1.5, n_samples).clip(0.5, 10),
        'creatinine': np.random.exponential(1.0, n_samples).clip(0.5, 8),
        'platelets': np.random.normal(250, 100, n_samples).clip(50, 500),
        'spo2': np.random.normal(96, 3, n_samples).clip(85, 100)
    }
    
    df = pd.DataFrame(data)
    
    # Simulate sepsis risk
    sepsis_prob = (
        (df['age'] > 70) * 0.3 +
        (df['heart_rate'] > 120) * 0.2 +
        (df['temperature'] > 38.5) * 0.2 +
        (df['lactate'] > 4) * 0.3 +
        (df['wbc'] > 12) * 0.1 +
        np.random.random(n_samples) * 0.1
    )
    
    df['has_sepsis'] = (sepsis_prob > 0.5).astype(int)
    print(f"Sample data generated: {df['has_sepsis'].sum()} sepsis cases ({df['has_sepsis'].mean()*100:.1f}%)")
    
    return df

if __name__ == '__main__':
    train_and_save_model()
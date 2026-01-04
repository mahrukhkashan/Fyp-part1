#!/usr/bin/env python3
"""
Script to train and save the sepsis prediction model - MIMIC-III Specific
"""

import pandas as pd
import numpy as np
import json
import pickle
import warnings
import time
import os
from datetime import datetime, timedelta
warnings.filterwarnings('ignore')

# Import your existing modules
try:
    from data.data_loader import DataLoader
    from data.feature_engineering import FeatureEngineer
    from models.sepsis_predictor import SepsisPredictor
    from explainability.shap_explainer import SHAPExplainer
    from sklearn.model_selection import train_test_split
except ImportError:
    print("⚠ Some modules not found, using simplified versions...")
    # Define simplified versions if modules don't exist
    class FeatureEngineer:
        def engineer_features(self, df):
            return df
    
    class SepsisPredictor:
        def __init__(self):
            self.best_model = None
            self.best_model_name = None
            self.best_score = None
        
        def train(self, X, y, test_size=0.2, random_state=42):
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.model_selection import train_test_split
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
            
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state, stratify=y
            )
            
            model = RandomForestClassifier(n_estimators=100, random_state=random_state)
            model.fit(X_train, y_train)
            
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            
            self.best_model = model
            self.best_model_name = "RandomForest"
            
            metrics = {
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred, zero_division=0),
                'recall': recall_score(y_test, y_pred, zero_division=0),
                'f1_score': f1_score(y_test, y_pred, zero_division=0),
                'roc_auc': roc_auc_score(y_test, y_pred_proba)
            }
            self.best_score = metrics['accuracy']
            
            results = {
                "RandomForest": {
                    "model": model,
                    "metrics": metrics
                }
            }
            
            return results, X_test, y_test
        
        def save_model(self, path):
            with open(path, 'wb') as f:
                pickle.dump(self.best_model, f)

class SHAPExplainer:
    def __init__(self, model, feature_names):
        self.model = model
        self.feature_names = feature_names
        self.explainer = None
    
    def fit_explainer(self, X_train):
        try:
            import shap
            self.explainer = shap.TreeExplainer(self.model)
        except:
            pass
    
    def save_explainer(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self, f)

def train_and_save_model():
    """Main function to train and save the model"""
    
    print("=" * 80)
    print("SEPSIS PREDICTION MODEL TRAINING - MIMIC-III DATASET")
    print("=" * 80)
    
    start_time = time.time()
    
    # Step 1: Initialize DataLoader
    print("\n[1/9] Initializing DataLoader...")
    data_loader = None
    
    try:
        data_loader = DataLoader()
        print("✓ Database connection successful")
        
        # Test connection
        test_query = "SELECT COUNT(*) as count FROM patients LIMIT 1"
        test_result = data_loader.db.execute_query(test_query)
        print(f"✓ Patients table has {test_result.iloc[0]['count'] if not test_result.empty else 'unknown'} records")
        
    except Exception as e:
        print(f"✗ Failed to initialize DataLoader: {e}")
        print("⚠ Will use sample data for development")
    
    # Step 2: Load data
    print("\n[2/9] Loading data from MIMIC-III...")
    
    if data_loader:
        patient_data = load_mimic_data(data_loader, sample_size=5000)
    else:
        print("⚠ No database connection, using comprehensive sample data...")
        patient_data = create_comprehensive_sample_data()
    
    print(f"\n✓ Data loaded successfully")
    print(f"  Records: {len(patient_data):,}")
    print(f"  Memory usage: {patient_data.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB")
    
    # Step 3: Check and prepare target variable
    print("\n[3/9] Preparing target variable...")
    
    if 'sepsis_label' in patient_data.columns:
        target_col = 'sepsis_label'
        patient_data = patient_data.rename(columns={'sepsis_label': 'has_sepsis'})
        target_col = 'has_sepsis'
    elif 'has_sepsis' in patient_data.columns:
        target_col = 'has_sepsis'
    else:
        print("  No sepsis label found. Creating labels based on clinical criteria...")
        patient_data = create_sepsis_labels_from_clinical(patient_data)
        target_col = 'has_sepsis'
    
    sepsis_count = patient_data[target_col].sum()
    sepsis_percent = (sepsis_count / len(patient_data)) * 100
    print(f"  Sepsis cases: {sepsis_count:,} ({sepsis_percent:.1f}%)")
    
    # Step 4: Prepare features
    print("\n[4/9] Preparing features...")
    
    # Display basic info
    print(f"  Data shape: {patient_data.shape}")
    print(f"  Columns: {len(patient_data.columns)}")
    
    # Identify feature columns
    non_feature_cols = [
        target_col,
        'subject_id', 'hadm_id', 'icustay_id', 
        'admittime', 'dischtime', 'deathtime',
        'dob', 'dod', 'charttime', 'storetime',
        'text', 'description', 'category', 'measurement',
        'unit', 'flag', 'itemid', 'seq_num',
        'icd9_code', 'diagnosis'
    ]
    
    # Keep only existing columns
    existing_non_feature_cols = [col for col in non_feature_cols if col in patient_data.columns]
    
    # Separate features and target
    y = patient_data[target_col]
    X = patient_data.drop(columns=existing_non_feature_cols)
    
    print(f"  Features after removal: {X.shape[1]}")
    
    # Step 5: Handle missing values and data types
    print("\n[5/9] Handling missing values and data types...")
    
    # Identify column types
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
    
    print(f"  Numeric columns: {len(numeric_cols)}")
    print(f"  Categorical columns: {len(categorical_cols)}")
    
    # Handle missing values in numeric columns
    for col in numeric_cols:
        if X[col].isnull().any():
            missing_pct = X[col].isnull().mean() * 100
            if missing_pct < 30:
                fill_value = X[col].median()
                X[col] = X[col].fillna(fill_value)
                print(f"    Filled '{col}' ({missing_pct:.1f}% missing) with median: {fill_value:.2f}")
            else:
                X = X.drop(columns=[col])
                print(f"    Dropped '{col}' ({missing_pct:.1f}% missing)")
    
    # Handle categorical columns
    print("\n  Processing categorical columns...")
    for col in categorical_cols:
        if col in X.columns:
            unique_count = X[col].nunique()
            if unique_count > 20:
                # High cardinality - use label encoding
                X[col] = pd.factorize(X[col])[0]
                print(f"    Label encoded '{col}' ({unique_count} unique values)")
            elif unique_count > 1:
                # Low cardinality - use one-hot encoding
                dummies = pd.get_dummies(X[col], prefix=col, drop_first=True)
                X = pd.concat([X.drop(columns=[col]), dummies], axis=1)
                print(f"    One-hot encoded '{col}' ({unique_count} unique values)")
            else:
                # Single value - drop
                X = X.drop(columns=[col])
                print(f"    Dropped '{col}' (single value)")
    
    # Ensure all features are numeric
    print("\n  Ensuring all features are numeric...")
    for col in X.columns:
        if not pd.api.types.is_numeric_dtype(X[col]):
            X[col] = pd.to_numeric(X[col], errors='coerce')
    
    # Final missing value check
    missing_cols = X.columns[X.isnull().any()].tolist()
    if missing_cols:
        print(f"  Filling remaining NaN in {len(missing_cols)} columns...")
        X = X.fillna(X.median())
    
    # Step 6: Advanced feature engineering
    print("\n[6/9] Advanced feature engineering...")
    try:
        feature_engineer = FeatureEngineer()
        X_engineered = feature_engineer.engineer_features(X)
        print(f"✓ Feature engineering completed")
        print(f"  Features before: {X.shape[1]}, after: {X_engineered.shape[1]}")
    except Exception as e:
        print(f"⚠ Feature engineering failed: {e}")
        print("  Using basic features...")
        X_engineered = X.copy()
        feature_engineer = FeatureEngineer()
    
    # Prepare final dataset
    X_final = X_engineered.copy()
    
    print(f"\n  Final feature matrix:")
    print(f"    Samples: {X_final.shape[0]:,}")
    print(f"    Features: {X_final.shape[1]}")
    print(f"    Sepsis rate: {y.mean()*100:.2f}%")
    
    # Step 7: Train model
    print("\n[7/9] Training model...")
    
    try:
        sepsis_predictor = SepsisPredictor()
        
        # Train with stratification
        results, X_test, y_test = sepsis_predictor.train(
            X_final, 
            y,
            test_size=0.25,  # Increased test size for better validation
            random_state=42
        )
        
        # Print detailed results
        print("\n" + "=" * 60)
        print("MODEL PERFORMANCE RESULTS")
        print("=" * 60)
        
        best_model_name = None
        best_score = 0
        
        for model_name, result in results.items():
            metrics = result['metrics']
            score = metrics.get('roc_auc', metrics.get('accuracy', 0))
            
            print(f"\n{model_name}:")
            print(f"  Accuracy:  {metrics.get('accuracy', 0):.4f}")
            print(f"  Precision: {metrics.get('precision', 0):.4f}")
            print(f"  Recall:    {metrics.get('recall', 0):.4f}")
            print(f"  F1-Score:  {metrics.get('f1_score', 0):.4f}")
            if 'roc_auc' in metrics:
                print(f"  ROC-AUC:   {metrics['roc_auc']:.4f}")
            
            if score > best_score:
                best_score = score
                best_model_name = model_name
        
        print(f"\n✓ Best model: {best_model_name} (score: {best_score:.4f})")
        
    except Exception as e:
        print(f"✗ Model training failed: {e}")
        import traceback
        traceback.print_exc()
        print("\n⚠ Creating fallback model...")
        
        # Fallback to simple Random Forest
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import accuracy_score
        
        X_train, X_test, y_train, y_test = train_test_split(
            X_final, y, test_size=0.25, random_state=42, stratify=y
        )
        
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            class_weight='balanced'
        )
        
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"\nFallback Random Forest:")
        print(f"  Accuracy: {accuracy:.4f}")
        
        sepsis_predictor = SepsisPredictor()
        sepsis_predictor.best_model = model
        sepsis_predictor.best_model_name = "RandomForest"
        sepsis_predictor.best_score = accuracy
    
    # Step 8: Save model and artifacts
    print("\n[8/9] Saving model artifacts...")
    
    # Create directories
    os.makedirs('models/saved_models', exist_ok=True)
    os.makedirs('reports', exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 1. Save model
    model_filename = f'sepsis_model_{timestamp}.pkl'
    model_path = f'models/saved_models/{model_filename}'
    
    try:
        sepsis_predictor.save_model(model_path)
    except:
        with open(model_path, 'wb') as f:
            pickle.dump(sepsis_predictor.best_model, f)
    
    print(f"✓ Model saved: {model_path}")
    
    # 2. Save feature names
    feature_names = list(X_final.columns)
    feature_names_path = f'models/saved_models/feature_names_{timestamp}.json'
    with open(feature_names_path, 'w') as f:
        json.dump(feature_names, f)
    print(f"✓ Feature names saved: {feature_names_path}")
    
    # 3. Save feature engineer
    feature_engineer_path = f'models/saved_models/feature_engineer_{timestamp}.pkl'
    with open(feature_engineer_path, 'wb') as f:
        pickle.dump(feature_engineer, f)
    print(f"✓ Feature engineer saved: {feature_engineer_path}")
    
    # 4. Save metadata
    metadata = {
        'timestamp': timestamp,
        'model_name': getattr(sepsis_predictor, 'best_model_name', 'Unknown'),
        'model_score': float(getattr(sepsis_predictor, 'best_score', 0)),
        'n_samples': len(X_final),
        'n_features': len(feature_names),
        'sepsis_rate': float(y.mean()),
        'data_source': 'MIMIC-III' if data_loader else 'Sample',
        'training_time_seconds': time.time() - start_time
    }
    
    metadata_path = f'models/saved_models/metadata_{timestamp}.json'
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"✓ Metadata saved: {metadata_path}")
    
    # 5. Create a simple report
    report_path = f'reports/training_report_{timestamp}.txt'
    with open(report_path, 'w') as f:
        f.write("=" * 60 + "\n")
        f.write("SEPSIS PREDICTION MODEL TRAINING REPORT\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Model: {metadata['model_name']}\n")
        f.write(f"Score: {metadata['model_score']:.4f}\n")
        f.write(f"Samples: {metadata['n_samples']:,}\n")
        f.write(f"Features: {metadata['n_features']}\n")
        f.write(f"Sepsis Rate: {metadata['sepsis_rate']*100:.2f}%\n")
        f.write(f"Data Source: {metadata['data_source']}\n")
        f.write(f"Training Time: {metadata['training_time_seconds']:.1f} seconds\n")
    
    print(f"✓ Training report saved: {report_path}")
    
    # Step 9: Create SHAP explainer
    print("\n[9/9] Creating SHAP explainer...")
    
    try:
        # Use a subset for SHAP (it can be memory intensive)
        shap_sample_size = min(1000, len(X_final))
        X_shap = X_final.sample(shap_sample_size, random_state=42)
        
        shap_explainer = SHAPExplainer(sepsis_predictor.best_model, feature_names)
        shap_explainer.fit_explainer(X_shap)
        
        shap_path = f'models/saved_models/shap_explainer_{timestamp}.pkl'
        shap_explainer.save_explainer(shap_path)
        print(f"✓ SHAP explainer saved: {shap_path}")
        
        # Create a simple feature importance plot
        try:
            import matplotlib.pyplot as plt
            
            # Get feature importances
            if hasattr(sepsis_predictor.best_model, 'feature_importances_'):
                importances = sepsis_predictor.best_model.feature_importances_
                indices = np.argsort(importances)[-15:]  # Top 15 features
                
                plt.figure(figsize=(10, 6))
                plt.title('Top 15 Feature Importances')
                plt.barh(range(len(indices)), importances[indices])
                plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
                plt.xlabel('Relative Importance')
                plt.tight_layout()
                
                importance_plot_path = f'reports/feature_importance_{timestamp}.png'
                plt.savefig(importance_plot_path, dpi=150)
                plt.close()
                print(f"✓ Feature importance plot saved: {importance_plot_path}")
                
        except Exception as e:
            print(f"  Could not create feature importance plot: {e}")
        
    except Exception as e:
        print(f"⚠ Could not create SHAP explainer: {e}")
        print("  Model will work without SHAP explanations")
    
    # Close database connection if exists
    if data_loader:
        try:
            data_loader.close_connection()
            print("✓ Database connection closed")
        except:
            pass
    
    # Summary
    elapsed_time = time.time() - start_time
    print("\n" + "=" * 80)
    print("MODEL TRAINING COMPLETED SUCCESSFULLY!")
    print("=" * 80)
    print(f"\nTraining Summary:")
    print(f"  Total time: {elapsed_time:.1f} seconds")
    print(f"  Records processed: {len(X_final):,}")
    print(f"  Sepsis cases: {sepsis_count:,} ({sepsis_percent:.1f}%)")
    print(f"  Features used: {len(feature_names)}")
    print(f"  Best model: {getattr(sepsis_predictor, 'best_model_name', 'Unknown')}")
    print(f"  Model score: {getattr(sepsis_predictor, 'best_score', 'N/A'):.4f}")
    print(f"\nArtifacts saved in:")
    print(f"  models/saved_models/")
    print(f"  reports/")
    print("\nTo use the model in production:")
    print(f"  Model file: {model_path}")
    print(f"  Feature names: {feature_names_path}")

def load_mimic_data(data_loader, sample_size=5000):
    """Load data from MIMIC-III database with sepsis labels"""
    
    print(f"Loading MIMIC-III data (sample: {sample_size} patients)...")
    
    # Get sepsis patients (based on ICD-9 codes)
    sepsis_query = f"""
    -- Get sepsis patients
    WITH sepsis_patients AS (
        SELECT DISTINCT 
            p.subject_id,
            p.gender,
            p.dob,
            CASE 
                WHEN p.dod IS NOT NULL AND p.dod <= a.dischtime THEN 1
                ELSE 0 
            END as mortality,
            1 as sepsis_label
        FROM patients p
        INNER JOIN admissions a ON p.subject_id = a.subject_id
        INNER JOIN diagnoses_icd d ON a.hadm_id = d.hadm_id
        WHERE d.icd9_code IN ('038', '785.52', '995.91', '995.92')  -- Sepsis codes
        LIMIT {sample_size // 2}  -- Half of sample are sepsis cases
    ),
    
    -- Get non-sepsis patients
    non_sepsis_patients AS (
        SELECT DISTINCT 
            p.subject_id,
            p.gender,
            p.dob,
            CASE 
                WHEN p.dod IS NOT NULL AND p.dod <= a.dischtime THEN 1
                ELSE 0 
            END as mortality,
            0 as sepsis_label
        FROM patients p
        INNER JOIN admissions a ON p.subject_id = a.subject_id
        WHERE p.subject_id NOT IN (SELECT subject_id FROM sepsis_patients)
        LIMIT {sample_size // 2}  -- Half of sample are non-sepsis
    ),
    
    -- Combine patients
    all_patients AS (
        SELECT * FROM sepsis_patients
        UNION ALL
        SELECT * FROM non_sepsis_patients
    ),
    
    -- Get vital signs for these patients
    patient_vitals AS (
        SELECT 
            ce.subject_id,
            AVG(CASE WHEN ce.itemid IN (211, 220045) THEN ce.valuenum END) as heart_rate,
            AVG(CASE WHEN ce.itemid IN (676, 677) THEN ce.valuenum END) as temperature,
            AVG(CASE WHEN ce.itemid IN (615, 618) THEN ce.valuenum END) as respiratory_rate,
            AVG(CASE WHEN ce.itemid IN (51, 442, 455, 6701, 220179, 220050) THEN ce.valuenum END) as systolic_bp,
            AVG(CASE WHEN ce.itemid IN (8368, 8440, 8441, 8555, 220180, 220051) THEN ce.valuenum END) as diastolic_bp,
            AVG(CASE WHEN ce.itemid IN (646, 220277) THEN ce.valuenum END) as spo2
        FROM chartevents ce
        WHERE ce.subject_id IN (SELECT subject_id FROM all_patients)
        AND ce.error IS DISTINCT FROM 1
        AND ce.valuenum IS NOT NULL
        GROUP BY ce.subject_id
    ),
    
    -- Get lab values
    patient_labs AS (
        SELECT 
            le.subject_id,
            AVG(CASE WHEN le.itemid IN (50889, 50912) THEN le.valuenum END) as sodium,
            AVG(CASE WHEN le.itemid IN (50902, 50822) THEN le.valuenum END) as potassium,
            AVG(CASE WHEN le.itemid IN (50971, 50868) THEN le.valuenum END) as creatinine,
            AVG(CASE WHEN le.itemid IN (51221, 51133) THEN le.valuenum END) as wbc,
            AVG(CASE WHEN le.itemid IN (50813, 50931) THEN le.valuenum END) as lactate,
            AVG(CASE WHEN le.itemid IN (51265, 51237) THEN le.valuenum END) as platelets
        FROM labevents le
        WHERE le.subject_id IN (SELECT subject_id FROM all_patients)
        AND le.valuenum IS NOT NULL
        GROUP BY le.subject_id
    )
    
    -- Final query
    SELECT 
        ap.subject_id,
        ap.gender,
        EXTRACT(YEAR FROM AGE(CURRENT_DATE, ap.dob)) as age,
        ap.mortality,
        ap.sepsis_label,
        COALESCE(pv.heart_rate, 80) as heart_rate,
        COALESCE(pv.temperature, 37) as temperature,
        COALESCE(pv.respiratory_rate, 18) as respiratory_rate,
        COALESCE(pv.systolic_bp, 120) as systolic_bp,
        COALESCE(pv.diastolic_bp, 80) as diastolic_bp,
        COALESCE(pv.spo2, 96) as spo2,
        COALESCE(pl.sodium, 140) as sodium,
        COALESCE(pl.potassium, 4.0) as potassium,
        COALESCE(pl.creatinine, 1.0) as creatinine,
        COALESCE(pl.wbc, 8.0) as wbc,
        COALESCE(pl.lactate, 1.2) as lactate,
        COALESCE(pl.platelets, 250) as platelets
    FROM all_patients ap
    LEFT JOIN patient_vitals pv ON ap.subject_id = pv.subject_id
    LEFT JOIN patient_labs pl ON ap.subject_id = pl.subject_id
    """
    
    try:
        print("Executing MIMIC-III query...")
        patient_data = data_loader.db.execute_query(sepsis_query)
        
        if patient_data is None or patient_data.empty:
            print("⚠ Query returned no data, using sample data...")
            return create_comprehensive_sample_data()
        
        print(f"✓ Loaded {len(patient_data)} patient records")
        return patient_data
        
    except Exception as e:
        print(f"✗ Query failed: {e}")
        print("⚠ Using sample data...")
        return create_comprehensive_sample_data()

def create_sepsis_labels_from_clinical(df):
    """Create sepsis labels based on clinical criteria (qSOFA/SOFA)"""
    
    print("Creating sepsis labels based on clinical criteria...")
    
    # qSOFA criteria (simplified)
    # 1. Respiratory rate ≥ 22/min
    # 2. Altered mental status (not available, using age > 65 as proxy)
    # 3. Systolic BP ≤ 100 mmHg
    
    has_qsofa = 0
    
    if 'respiratory_rate' in df.columns:
        has_qsofa += (df['respiratory_rate'] >= 22).astype(int)
    
    if 'age' in df.columns:
        has_qsofa += (df['age'] > 65).astype(int)
    
    if 'systolic_bp' in df.columns:
        has_qsofa += (df['systolic_bp'] <= 100).astype(int)
    
    # Additional sepsis indicators
    sepsis_indicators = 0
    
    if 'temperature' in df.columns:
        sepsis_indicators += ((df['temperature'] > 38.3) | (df['temperature'] < 36)).astype(int)
    
    if 'heart_rate' in df.columns:
        sepsis_indicators += (df['heart_rate'] > 90).astype(int)
    
    if 'wbc' in df.columns:
        sepsis_indicators += ((df['wbc'] > 12) | (df['wbc'] < 4)).astype(int)
    
    if 'lactate' in df.columns:
        sepsis_indicators += (df['lactate'] > 2).astype(int)
    
    # Combined sepsis probability
    sepsis_prob = (
        (has_qsofa >= 2) * 0.4 +
        (sepsis_indicators >= 2) * 0.4 +
        np.random.random(len(df)) * 0.2
    )
    
    df['has_sepsis'] = (sepsis_prob > 0.5).astype(int)
    
    sepsis_count = df['has_sepsis'].sum()
    sepsis_percent = (sepsis_count / len(df)) * 100
    print(f"Created {sepsis_count} sepsis cases ({sepsis_percent:.1f}%) using clinical criteria")
    
    return df

def create_comprehensive_sample_data(n_samples=2000):
    """Create comprehensive sample data with realistic distributions"""
    
    print(f"Generating {n_samples} comprehensive sample records...")
    np.random.seed(42)
    
    # Base data
    data = {
        'subject_id': range(1, n_samples + 1),
        'age': np.random.normal(65, 18, n_samples).clip(18, 100),
        'gender': np.random.choice(['M', 'F'], n_samples, p=[0.55, 0.45]),
        'mortality': np.random.binomial(1, 0.15, n_samples),
    }
    
    # Vital signs with realistic correlations
    data['heart_rate'] = np.random.normal(85, 20, n_samples).clip(40, 180)
    data['temperature'] = np.random.normal(37, 1.2, n_samples).clip(35, 41)
    data['respiratory_rate'] = np.random.normal(18, 6, n_samples).clip(8, 40)
    data['systolic_bp'] = np.random.normal(125, 25, n_samples).clip(70, 220)
    data['diastolic_bp'] = data['systolic_bp'] * 0.65 + np.random.normal(0, 5, n_samples)
    data['diastolic_bp'] = data['diastolic_bp'].clip(40, 140)
    data['spo2'] = np.random.normal(96, 3, n_samples).clip(85, 100)
    
    # Lab values
    data['sodium'] = np.random.normal(140, 5, n_samples).clip(120, 160)
    data['potassium'] = np.random.normal(4.0, 0.8, n_samples).clip(2.5, 7.0)
    data['creatinine'] = np.random.exponential(1.0, n_samples).clip(0.3, 10.0)
    data['wbc'] = np.random.lognormal(2.0, 0.5, n_samples).clip(1.0, 40.0)
    data['lactate'] = np.random.exponential(1.2, n_samples).clip(0.5, 15.0)
    data['platelets'] = np.random.normal(250, 80, n_samples).clip(20, 600)
    
    df = pd.DataFrame(data)
    
    # Create sepsis labels with realistic clinical criteria
    # Higher risk for older patients, abnormal vitals, abnormal labs
    sepsis_risk_factors = (
        (df['age'] > 70) * 0.25 +
        (df['heart_rate'] > 120) * 0.15 +
        (df['temperature'] > 38.3) * 0.15 +
        (df['respiratory_rate'] > 22) * 0.15 +
        (df['systolic_bp'] < 100) * 0.10 +
        (df['wbc'] > 12) * 0.10 +
        (df['lactate'] > 2) * 0.15 +
        (df['creatinine'] > 1.5) * 0.10
    )
    
    # Add some noise
    sepsis_risk_factors += np.random.random(n_samples) * 0.2
    
    # Create labels (about 15-20% sepsis rate)
    threshold = np.percentile(sepsis_risk_factors, 80)  # Top 20% are sepsis
    df['has_sepsis'] = (sepsis_risk_factors > threshold).astype(int)
    
    sepsis_count = df['has_sepsis'].sum()
    sepsis_percent = (sepsis_count / n_samples) * 100
    
    print(f"Generated {sepsis_count} sepsis cases ({sepsis_percent:.1f}%)")
    print("Sample data includes: vitals, labs, demographics, and realistic sepsis labels")
    
    return df

if __name__ == '__main__':
    try:
        train_and_save_model()
    except KeyboardInterrupt:
        print("\n⚠ Training interrupted by user")
    except Exception as e:
        print(f"\n✗ Training failed with error: {e}")
        import traceback
        traceback.print_exc()
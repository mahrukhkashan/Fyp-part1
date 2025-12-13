import pandas as pd
import numpy as np
from data.sql_queries import SQLQueries
from config.database_config import DatabaseConnection
import warnings
warnings.filterwarnings('ignore')

class DataLoader:
    """Loads data from MIMIC-III database"""
    
    def __init__(self):
        self.db = DatabaseConnection()
        self.db.connect()
        self.sql = SQLQueries()
    
    def load_patient_data(self, subject_id=None, limit=5000):
        """Load balanced patient data for training"""
        print("Loading patient data...")
        
        try:
            # Use the new balanced dataset query
            query = self.sql.get_sepsis_patients_data(limit)
            df = self.db.execute_query(query)
            
            if df is None or df.empty:
                print("Query returned no data. Using sample data instead...")
                return self._generate_sample_data(min(limit, 1000))
            
            print(f"✓ Successfully loaded {len(df)} records from database")
            print(f"  Sepsis cases: {df['sepsis_label'].sum()} ({df['sepsis_label'].mean()*100:.1f}%)")
            print(f"  Non-sepsis cases: {len(df) - df['sepsis_label'].sum()}")
            
            # Rename target column to match expected format
            df = df.rename(columns={'sepsis_label': 'has_sepsis'})
            
            return df
            
        except Exception as e:
            print(f"Error loading data: {e}")
            print("Using sample data instead...")
            return self._generate_sample_data(min(limit, 1000))
    
    def load_vitals_data(self, subject_ids=None, limit=10000):
        """Load vital signs data from chartevents"""
        print("Loading vital signs data...")
        
        try:
            if subject_ids:
                # If you need this method, implement get_vitals_query_with_subjects
                vitals_query = self.sql.get_vitals_query(limit)
            else:
                vitals_query = self.sql.get_vitals_query(limit)
            
            vitals_df = self.db.execute_query(vitals_query)
            return vitals_df if vitals_df is not None else pd.DataFrame()
        except Exception as e:
            print(f"Error loading vitals: {e}")
            return pd.DataFrame()
    
    def load_lab_data(self, subject_ids=None, limit=10000):
        """Load laboratory results"""
        print("Loading laboratory data...")
        
        try:
            if subject_ids:
                # If you need this method, implement get_labs_query_with_subjects
                labs_query = self.sql.get_labs_query(limit)
            else:
                labs_query = self.sql.get_labs_query(limit)
            
            labs_df = self.db.execute_query(labs_query)
            return labs_df if labs_df is not None else pd.DataFrame()
        except Exception as e:
            print(f"Error loading labs: {e}")
            return pd.DataFrame()
    
    def load_notes_data(self, subject_ids=None, limit=1000):
        """Load clinical notes"""
        print("Loading clinical notes...")
        
        try:
            if subject_ids:
                # If you need this method, implement get_notes_query_with_subjects
                notes_query = self.sql.get_notes_query(limit)
            else:
                notes_query = self.sql.get_notes_query(limit)
            
            notes_df = self.db.execute_query(notes_query)
            return notes_df if notes_df is not None else pd.DataFrame()
        except Exception as e:
            print(f"Error loading notes: {e}")
            return pd.DataFrame()
    
    def get_patient_full_data(self, subject_id):
        """Get complete patient data for prediction"""
        try:
            query = self.sql.get_patient_full_data(subject_id)
            df = self.db.execute_query(query)
            return df if df is not None else pd.DataFrame()
        except Exception as e:
            print(f"Error loading patient data: {e}")
            return pd.DataFrame()
    
    def _generate_sample_data(self, n_samples=1000):
        """Generate sample data if database fails"""
        print(f"Generating {n_samples} sample records...")
        
        np.random.seed(42)
        
        data = {
            'subject_id': range(1, n_samples + 1),
            'hadm_id': range(1001, 1001 + n_samples),
            'age': np.random.normal(60, 20, n_samples).clip(18, 100),
            'gender': np.random.choice(['M', 'F'], n_samples, p=[0.55, 0.45]),
            'ethnicity': np.random.choice(['WHITE', 'BLACK', 'ASIAN', 'HISPANIC'], n_samples),
            'heart_rate': np.random.normal(80, 20, n_samples).clip(40, 180),
            'systolic_bp': np.random.normal(120, 25, n_samples).clip(70, 200),
            'diastolic_bp': np.random.normal(80, 15, n_samples).clip(40, 120),
            'temperature': np.random.normal(37, 1, n_samples).clip(35, 41),
            'respiratory_rate': np.random.normal(18, 6, n_samples).clip(8, 40),
            'spo2': np.random.normal(96, 3, n_samples).clip(70, 100),
            'wbc': np.random.normal(8, 4, n_samples).clip(2, 30),
            'lactate': np.random.exponential(1.5, n_samples).clip(0.5, 10),
            'creatinine': np.random.exponential(1.0, n_samples).clip(0.3, 5),
            'platelets': np.random.normal(250, 100, n_samples).clip(50, 500),
            'bilirubin': np.random.exponential(0.8, n_samples).clip(0.1, 5),
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
    
    def close_connection(self):
        """Close database connection"""
        self.db.close()
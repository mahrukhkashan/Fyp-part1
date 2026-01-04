import pandas as pd
import numpy as np
from data.sql_queries import SQLQueries
from config.database_config import DatabaseConnection
import warnings
warnings.filterwarnings('ignore')

class DataLoader:
    """Loads data from MIMIC-III demo database (first 50 entries per table)"""
    
    def __init__(self, database_name="mimic_demo"):
        self.db = DatabaseConnection()
        self.db.connect()
        self.sql = SQLQueries()
        self.database_name = database_name
        print(f"Connected to {database_name} database")
    
    def load_patient_data(self, subject_id=None, limit=500):
        """Load patient data for training (first 50 entries)"""
        print(f"Loading patient data (first {limit} entries)...")
        
        try:
            # Use optimized query for mimic_demo
            query = self.sql.get_sepsis_training_data_demo(limit)
            print(f"Executing query for {limit} records...")
            
            df = self.db.execute_query(query)
            
            if df is None or df.empty:
                print("Query returned no data. Using sample data...")
                return self._generate_sample_data(min(limit, 500))
            
            print(f"✓ Successfully loaded {len(df)} records")
            
            # Check if we have sepsis label
            if 'sepsis_label' in df.columns:
                sepsis_count = df['sepsis_label'].sum()
                sepsis_percent = (sepsis_count / len(df)) * 100
                print(f"  Sepsis cases: {sepsis_count} ({sepsis_percent:.1f}%)")
                # Rename for consistency
                df = df.rename(columns={'sepsis_label': 'has_sepsis'})
            elif 'has_sepsis' not in df.columns:
                print("Warning: No sepsis label column found")
            
            return df
            
        except Exception as e:
            print(f"Error loading patient data: {e}")
            print("Using sample data...")
            return self._generate_sample_data(min(limit, 50))
    
    def load_vitals_data(self, subject_ids=None, limit=50):
        """Load vital signs data (first 50 entries)"""
        print(f"Loading vital signs data (first {limit} entries)...")
        
        try:
            vitals_query = self.sql.get_vitals_query_demo(limit)
            vitals_df = self.db.execute_query(vitals_query)
            
            if vitals_df is not None and not vitals_df.empty:
                print(f"✓ Loaded {len(vitals_df)} vitals records")
                return vitals_df
            else:
                print("No vitals data found")
                return pd.DataFrame()
                
        except Exception as e:
            print(f"Error loading vitals: {e}")
            return pd.DataFrame()
    
    def load_lab_data(self, subject_ids=None, limit=500):
        """Load laboratory results (first 50 entries)"""
        print(f"Loading laboratory data (first {limit} entries)...")
        
        try:
            labs_query = self.sql.get_labs_query_demo(limit)
            labs_df = self.db.execute_query(labs_query)
            
            if labs_df is not None and not labs_df.empty:
                print(f"✓ Loaded {len(labs_df)} lab records")
                return labs_df
            else:
                print("No lab data found")
                return pd.DataFrame()
                
        except Exception as e:
            print(f"Error loading labs: {e}")
            return pd.DataFrame()
    
    def load_admissions_data(self, limit=50):
        """Load admissions data (first 500 entries)"""
        print(f"Loading admissions data (first {limit} entries)...")
        
        try:
            admissions_query = self.sql.get_admissions_query_demo(limit)
            admissions_df = self.db.execute_query(admissions_query)
            
            if admissions_df is not None and not admissions_df.empty:
                print(f"✓ Loaded {len(admissions_df)} admissions records")
                return admissions_df
            else:
                print("No admissions data found")
                return pd.DataFrame()
                
        except Exception as e:
            print(f"Error loading admissions: {e}")
            return pd.DataFrame()
    
    def load_diagnoses_data(self, limit=500):
        """Load diagnoses data (first 500 entries)"""
        print(f"Loading diagnoses data (first {limit} entries)...")
        
        try:
            diagnoses_query = self.sql.get_diagnoses_query_demo(limit)
            diagnoses_df = self.db.execute_query(diagnoses_query)
            
            if diagnoses_df is not None and not diagnoses_df.empty:
                print(f"✓ Loaded {len(diagnoses_df)} diagnoses records")
                return diagnoses_df
            else:
                print("No diagnoses data found")
                return pd.DataFrame()
                
        except Exception as e:
            print(f"Error loading diagnoses: {e}")
            return pd.DataFrame()
    
    def load_complete_patient_data(self, subject_id):
        """Get complete data for a specific patient"""
        try:
            query = self.sql.get_patient_full_data_demo(subject_id)
            df = self.db.execute_query(query)
            
            if df is not None and not df.empty:
                print(f"✓ Loaded complete data for patient {subject_id}")
                return df
            else:
                print(f"No data found for patient {subject_id}")
                return pd.DataFrame()
                
        except Exception as e:
            print(f"Error loading patient data: {e}")
            return pd.DataFrame()
    
    def test_connection(self):
        """Test database connection"""
        try:
            test_query = "SELECT 1 as test"
            result = self.db.execute_query(test_query)
            if result is not None:
                print("✓ Database connection test passed")
                return True
            else:
                print("✗ Database connection test failed")
                return False
        except Exception as e:
            print(f"✗ Database connection error: {e}")
            return False
    
    def _generate_sample_data(self, n_samples=500):
        """Generate sample data if database fails"""
        print(f"Generating {n_samples} sample records...")
        
        np.random.seed(42)
        
        data = {
            'subject_id': range(1, n_samples + 1),
            'hadm_id': range(1001, 1001 + n_samples),
            'age': np.random.normal(60, 15, n_samples).clip(18, 95),
            'gender': np.random.choice(['M', 'F'], n_samples, p=[0.55, 0.45]),
            'ethnicity': np.random.choice(['WHITE', 'BLACK', 'ASIAN', 'HISPANIC'], n_samples),
            'heart_rate': np.random.normal(80, 15, n_samples).clip(50, 150),
            'systolic_bp': np.random.normal(125, 20, n_samples).clip(90, 180),
            'diastolic_bp': np.random.normal(80, 10, n_samples).clip(60, 120),
            'temperature': np.random.normal(36.8, 0.8, n_samples).clip(35.5, 39.5),
            'respiratory_rate': np.random.normal(18, 4, n_samples).clip(12, 28),
            'spo2': np.random.normal(96, 2, n_samples).clip(92, 100),
            'wbc': np.random.normal(8, 3, n_samples).clip(3, 20),
            'lactate': np.random.exponential(1.2, n_samples).clip(0.5, 5),
            'creatinine': np.random.exponential(0.9, n_samples).clip(0.3, 3),
            'platelets': np.random.normal(250, 80, n_samples).clip(100, 500)
        }
        
        df = pd.DataFrame(data)
        
        # Simulate sepsis risk
        sepsis_prob = (
            (df['age'] > 70) * 0.3 +
            (df['heart_rate'] > 100) * 0.2 +
            (df['temperature'] > 37.8) * 0.2 +
            (df['lactate'] > 2) * 0.25 +
            np.random.random(n_samples) * 0.05
        )
        
        df['has_sepsis'] = (sepsis_prob > 0.5).astype(int)
        
        sepsis_count = df['has_sepsis'].sum()
        sepsis_percent = (sepsis_count / n_samples) * 100
        print(f"Sample data generated: {sepsis_count} sepsis cases ({sepsis_percent:.1f}%)")
        
        return df
    
    def close_connection(self):
        """Close database connection"""
        self.db.close()
        print("Database connection closed")
# test_mimic_tables.py
from config.database_config import DatabaseConnection

db = DatabaseConnection()
if db.connect():
    # Test each table
    tables = [
        'admissions',
        'patients', 
        'diagnoses_icd',
        'chartevents',
        'd_items',
        'labevents',
        'd_labitems'
    ]
    
    for table in tables:
        try:
            result = db.execute_query(f"SELECT COUNT(*) FROM {table}")
            count = result.iloc[0,0] if not result.empty else 0
            print(f"✓ {table}: {count:,} records")
        except Exception as e:
            print(f"✗ {table}: ERROR - {e}")
    
    db.close()
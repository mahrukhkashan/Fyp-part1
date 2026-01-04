class SQLQueries:
    """Contains SQL queries for MIMIC-III demo database (first 5000 entries)"""
    
    @staticmethod
    def get_admissions_query_demo(limit=500):
        """Query for admissions data (first 500 entries)"""
        query = f"""
        SELECT 
            subject_id,
            hadm_id,
            admission_type,
            admission_location,
            discharge_location,
            insurance,
            language,
            religion,
            marital_status,
            ethnicity,
            diagnosis,
            hospital_expire_flag,
            DATE_PART('year', age(admittime, dob)) as admission_age
        FROM (
            SELECT 
                a.*,
                p.dob,
                ROW_NUMBER() OVER (ORDER BY a.subject_id, a.hadm_id) as rn
            FROM admissions a
            JOIN patients p ON a.subject_id = p.subject_id
        ) AS numbered
        WHERE rn <= {limit}
        ORDER BY subject_id, hadm_id
        """
        return query
    
    @staticmethod
    def get_patients_query_demo(limit=500):
        """Query for patient demographics (first 500 entries)"""
        query = f"""
        SELECT 
            subject_id,
            gender,
            dob,
            EXTRACT(YEAR FROM dob) as birth_year,
            CASE 
                WHEN gender = 'M' THEN 1
                WHEN gender = 'F' THEN 0
                ELSE NULL 
            END as gender_numeric,
            DATE_PART('year', age(CURRENT_DATE, dob)) as current_age
        FROM (
            SELECT *, ROW_NUMBER() OVER (ORDER BY subject_id) as rn
            FROM patients
        ) AS numbered
        WHERE rn <= {limit}
        ORDER BY subject_id
        """
        return query
    
    @staticmethod
    def get_icustays_query_demo(limit=500):
        """Query for ICU stays (first 500 entries)"""
        query = f"""
        SELECT 
            subject_id,
            hadm_id,
            icustay_id,
            first_careunit,
            last_careunit,
            first_wardid,
            last_wardid,
            EXTRACT(EPOCH FROM (outtime - intime))/3600 as icu_los_hours
        FROM (
            SELECT *, ROW_NUMBER() OVER (ORDER BY subject_id, hadm_id) as rn
            FROM icustays
            WHERE outtime IS NOT NULL
        ) AS numbered
        WHERE rn <= {limit}
        ORDER BY subject_id, hadm_id
        """
        return query
    
    @staticmethod
    def get_diagnoses_query_demo(limit=500):
        """Query for diagnoses (first 500 entries)"""
        query = f"""
        SELECT 
            subject_id,
            hadm_id,
            icd9_code,
            seq_num
        FROM (
            SELECT *, ROW_NUMBER() OVER (ORDER BY subject_id, hadm_id) as rn
            FROM diagnoses_icd
            WHERE icd9_code IS NOT NULL
        ) AS numbered
        WHERE rn <= {limit}
        ORDER BY subject_id, hadm_id
        """
        return query
    
    @staticmethod
    def get_vitals_query_demo(limit=500):
        """Query for vital signs (first 500 entries)"""
        query = f"""
        SELECT 
            ce.subject_id,
            ce.hadm_id,
            ce.icustay_id,
            ce.itemid,
            di.label as measurement,
            ce.charttime,
            ce.valuenum as value,
            ce.valueuom as unit
        FROM (
            SELECT ce.*, ROW_NUMBER() OVER (ORDER BY ce.subject_id, ce.charttime) as rn
            FROM chartevents ce
            JOIN d_items di ON ce.itemid = di.itemid
            WHERE ce.valuenum IS NOT NULL 
                AND ce.valuenum > 0
                AND di.category IN ('Respiratory', 'Vital Signs', 'Cardiovascular')
        ) AS ce
        WHERE rn <= {limit}
        ORDER BY ce.subject_id, ce.charttime
        """
        return query
    
    @staticmethod
    def get_labs_query_demo(limit=500):
        """Query for laboratory results (first 500 entries)"""
        query = f"""
        SELECT 
            le.subject_id,
            le.hadm_id,
            le.itemid,
            le.charttime,
            le.valuenum as value,
            le.valueuom as unit,
            le.flag
        FROM (
            SELECT le.*, ROW_NUMBER() OVER (ORDER BY le.subject_id, le.charttime) as rn
            FROM labevents le
            JOIN d_labitems dli ON le.itemid = dli.itemid
            WHERE le.valuenum IS NOT NULL
                AND le.valuenum > 0
        ) AS le
        WHERE rn <= {limit}
        ORDER BY le.subject_id, le.charttime
        """
        return query
    
    @staticmethod
    def get_sepsis_training_data_demo(limit=500):
        """Optimized query for sepsis training data from mimic_demo"""
        query = f"""
        WITH patient_sample AS (
            -- Get first 500 patients
            SELECT subject_id, gender, dob
            FROM (
                SELECT *, ROW_NUMBER() OVER (ORDER BY subject_id) as rn
                FROM patients
            ) AS numbered
            WHERE rn <= {limit}
        ),
        admissions_sample AS (
            -- Get admissions for these patients
            SELECT a.*, ps.gender, ps.dob
            FROM admissions a
            JOIN patient_sample ps ON a.subject_id = ps.subject_id
            ORDER BY a.subject_id, a.hadm_id
            LIMIT {limit}
        ),
        sepsis_flags AS (
            -- Check for sepsis diagnoses
            SELECT DISTINCT 
                a.subject_id,
                a.hadm_id,
                CASE 
                    WHEN EXISTS (
                        SELECT 1 FROM diagnoses_icd d 
                        WHERE d.subject_id = a.subject_id 
                        AND d.hadm_id = a.hadm_id
                        AND d.icd9_code IN ('038', '785.52', '995.91', '995.92')
                    ) THEN 1 
                    ELSE 0 
                END as sepsis_label
            FROM admissions_sample a
        ),
        aggregated_vitals AS (
            -- Get aggregated vitals
            SELECT 
                a.subject_id,
                a.hadm_id,
                AVG(CASE WHEN di.label ILIKE '%heart rate%' THEN ce.valuenum END) as heart_rate,
                AVG(CASE WHEN di.label ILIKE '%temperature%' THEN ce.valuenum END) as temperature,
                AVG(CASE WHEN di.label ILIKE '%respiratory rate%' THEN ce.valuenum END) as respiratory_rate,
                AVG(CASE WHEN di.label ILIKE '%blood pressure%' AND di.label ILIKE '%systolic%' THEN ce.valuenum END) as systolic_bp,
                AVG(CASE WHEN di.label ILIKE '%oxygen saturation%' THEN ce.valuenum END) as spo2
            FROM admissions_sample a
            LEFT JOIN chartevents ce ON a.subject_id = ce.subject_id AND a.hadm_id = ce.hadm_id
            LEFT JOIN d_items di ON ce.itemid = di.itemid
            WHERE ce.valuenum IS NOT NULL AND ce.valuenum > 0
            GROUP BY a.subject_id, a.hadm_id
        ),
        aggregated_labs AS (
            -- Get aggregated labs
            SELECT 
                a.subject_id,
                a.hadm_id,
                AVG(CASE WHEN dli.label ILIKE '%white blood cell%' THEN le.valuenum END) as wbc,
                AVG(CASE WHEN dli.label ILIKE '%lactate%' THEN le.valuenum END) as lactate,
                AVG(CASE WHEN dli.label ILIKE '%creatinine%' THEN le.valuenum END) as creatinine
            FROM admissions_sample a
            LEFT JOIN labevents le ON a.subject_id = le.subject_id AND a.hadm_id = le.hadm_id
            LEFT JOIN d_labitems dli ON le.itemid = dli.itemid
            WHERE le.valuenum IS NOT NULL AND le.valuenum > 0
            GROUP BY a.subject_id, a.hadm_id
        )
        SELECT 
            a.subject_id,
            a.hadm_id,
            a.gender,
            DATE_PART('year', age(a.admittime, a.dob)) as age,
            a.ethnicity,
            a.admission_type,
            COALESCE(v.heart_rate, 80) as heart_rate,
            COALESCE(v.temperature, 37) as temperature,
            COALESCE(v.respiratory_rate, 18) as respiratory_rate,
            COALESCE(v.systolic_bp, 120) as systolic_bp,
            COALESCE(v.spo2, 98) as spo2,
            COALESCE(l.wbc, 8) as wbc,
            COALESCE(l.lactate, 1.0) as lactate,
            COALESCE(l.creatinine, 0.9) as creatinine,
            sf.sepsis_label
        FROM admissions_sample a
        LEFT JOIN aggregated_vitals v ON a.subject_id = v.subject_id AND a.hadm_id = v.hadm_id
        LEFT JOIN aggregated_labs l ON a.subject_id = l.subject_id AND a.hadm_id = l.hadm_id
        LEFT JOIN sepsis_flags sf ON a.subject_id = sf.subject_id AND a.hadm_id = sf.hadm_id
        ORDER BY a.subject_id, a.hadm_id
        """
        return query
    
    @staticmethod
    def get_patient_full_data_demo(subject_id):
        """Get complete patient data for prediction (demo version)"""
        query = f"""
        SELECT 
            p.subject_id,
            p.gender,
            DATE_PART('year', age(CURRENT_DATE, p.dob)) as age,
            a.hadm_id,
            a.admission_type,
            a.ethnicity,
            a.diagnosis,
            -- Simple sepsis check
            CASE WHEN EXISTS (
                SELECT 1 FROM diagnoses_icd d 
                WHERE d.subject_id = p.subject_id
                AND d.icd9_code IN ('038', '785.52', '995.91', '995.92')
            ) THEN 1 ELSE 0 END as has_sepsis
        FROM patients p
        LEFT JOIN admissions a ON p.subject_id = a.subject_id
        WHERE p.subject_id = {subject_id}
        ORDER BY a.admittime DESC
        LIMIT 1
        """
        return query
    
    # Keep original methods for backward compatibility
    @staticmethod
    def get_sepsis_patients_data(limit=500):
        """Alias for demo version"""
        return SQLQueries.get_sepsis_training_data_demo(limit)
    
    @staticmethod
    def get_vitals_query(limit=500):
        """Alias for demo version"""
        return SQLQueries.get_vitals_query_demo(limit)
    
    @staticmethod
    def get_labs_query(limit=500):
        """Alias for demo version"""
        return SQLQueries.get_labs_query_demo(limit)
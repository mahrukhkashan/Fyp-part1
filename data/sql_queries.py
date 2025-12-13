class SQLQueries:
    """Contains SQL queries for MIMIC-III database"""
    
    @staticmethod
    def get_admissions_query(limit=None):
        """Query for admissions data"""
        query = """
        SELECT 
            a.subject_id,
            a.hadm_id,
            a.admission_type,
            a.admission_location,
            a.discharge_location,
            a.insurance,
            a.language,
            a.religion,
            a.marital_status,
            a.ethnicity,
            a.diagnosis,
            a.hospital_expire_flag,
            DATE_PART('year', age(a.admittime, p.dob)) as admission_age,
            EXTRACT(EPOCH FROM (a.dischtime - a.admittime))/3600 as length_of_stay_hours
        FROM admissions a
        JOIN patients p ON a.subject_id = p.subject_id
        """
        if limit:
            query += f" LIMIT {limit}"
        return query
    
    @staticmethod
    def get_patients_query(limit=None):
        """Query for patient demographics"""
        query = """
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
            EXTRACT(YEAR FROM age(CURRENT_DATE, dob)) as current_age
        FROM patients
        """
        if limit:
            query += f" LIMIT {limit}"
        return query
    
    @staticmethod
    def get_icustays_query(limit=None):
        """Query for ICU stays"""
        query = """
        SELECT 
            subject_id,
            hadm_id,
            icustay_id,
            first_careunit,
            last_careunit,
            first_wardid,
            last_wardid,
            EXTRACT(EPOCH FROM (outtime - intime))/3600 as icu_los_hours
        FROM icustays
        WHERE outtime IS NOT NULL
        """
        if limit:
            query += f" LIMIT {limit}"
        return query
    
    @staticmethod
    def get_diagnoses_query(limit=None):
        """Query for diagnoses"""
        query = """
        SELECT 
            subject_id,
            hadm_id,
            icd9_code,
            seq_num
        FROM diagnoses_icd
        WHERE icd9_code IS NOT NULL
        """
        if limit:
            query += f" LIMIT {limit}"
        return query
    
    @staticmethod
    def get_vitals_query(limit=None):
        """Query for vital signs"""
        query = """
        SELECT 
            ce.subject_id,
            ce.hadm_id,
            ce.icustay_id,
            ce.itemid,
            di.label as measurement,
            ce.charttime,
            ce.valuenum as value,
            ce.valueuom as unit
        FROM chartevents ce
        JOIN d_items di ON ce.itemid = di.itemid
        WHERE ce.valuenum IS NOT NULL 
            AND ce.valuenum > 0
            AND di.category IN ('Respiratory', 'Vital Signs', 'Cardiovascular')
            AND di.label ILIKE ANY(ARRAY['%heart rate%', '%blood pressure%', '%temperature%', 
                                       '%respiratory rate%', '%oxygen saturation%', '%glucose%'])
        """
        if limit:
            query += f" LIMIT {limit}"
        return query
    
    @staticmethod
    def get_labs_query(limit=None):
        """Query for laboratory results"""
        query = """
        SELECT 
            le.subject_id,
            le.hadm_id,
            le.itemid,
            le.charttime,
            le.valuenum as value,
            le.valueuom as unit,
            le.flag
        FROM labevents le
        JOIN d_labitems dli ON le.itemid = dli.itemid
        WHERE le.valuenum IS NOT NULL
            AND le.valuenum > 0
            AND dli.label ILIKE ANY(ARRAY['%white blood cell%', '%lactate%', '%creatinine%', 
                                         '%platelet%', '%bilirubin%', '%c-reactive protein%',
                                         '%procalcitonin%', '%glucose%', '%hemoglobin%'])
        """
        if limit:
            query += f" LIMIT {limit}"
        return query
    
    @staticmethod
    def get_notes_query(limit=None):
        """Query for clinical notes"""
        query = """
        SELECT 
            subject_id,
            hadm_id,
            chartdate,
            category,
            description,
            text
        FROM noteevents
        WHERE iserror IS NULL 
            OR iserror != '1'
            AND text IS NOT NULL
            AND LENGTH(text) > 100
            AND category IN ('Discharge summary', 'Radiology', 'Nursing', 'Physician')
        """
        if limit:
            query += f" LIMIT {limit}"
        return query
    
    @staticmethod
    def get_patient_full_data(subject_id):
        """Get complete patient data for prediction"""
        query = f"""
        WITH patient_info AS (
            SELECT 
                p.subject_id,
                p.gender,
                DATE_PART('year', age(CURRENT_DATE, p.dob)) as age,
                a.hadm_id,
                a.admission_type,
                a.ethnicity,
                a.diagnosis,
                i.icustay_id,
                i.first_careunit,
                i.last_careunit
            FROM patients p
            LEFT JOIN admissions a ON p.subject_id = a.subject_id
            LEFT JOIN icustays i ON a.hadm_id = i.hadm_id
            WHERE p.subject_id = {subject_id}
            ORDER BY a.admittime DESC
            LIMIT 1
        ),
        latest_vitals AS (
            SELECT 
                ce.subject_id,
                AVG(CASE WHEN di.label ILIKE '%heart rate%' THEN ce.valuenum END) as heart_rate,
                AVG(CASE WHEN di.label ILIKE '%blood pressure%' AND di.label ILIKE '%systolic%' THEN ce.valuenum END) as systolic_bp,
                AVG(CASE WHEN di.label ILIKE '%blood pressure%' AND di.label ILIKE '%diastolic%' THEN ce.valuenum END) as diastolic_bp,
                AVG(CASE WHEN di.label ILIKE '%temperature%' THEN ce.valuenum END) as temperature,
                AVG(CASE WHEN di.label ILIKE '%respiratory rate%' THEN ce.valuenum END) as respiratory_rate,
                AVG(CASE WHEN di.label ILIKE '%oxygen saturation%' THEN ce.valuenum END) as spo2
            FROM chartevents ce
            JOIN d_items di ON ce.itemid = di.itemid
            WHERE ce.subject_id = {subject_id}
                AND ce.valuenum IS NOT NULL
                AND ce.valuenum > 0
                AND di.label ILIKE ANY(ARRAY['%heart rate%', '%blood pressure%', '%temperature%', 
                                           '%respiratory rate%', '%oxygen saturation%'])
            GROUP BY ce.subject_id
        ),
        latest_labs AS (
            SELECT 
                le.subject_id,
                AVG(CASE WHEN dli.label ILIKE '%white blood cell%' THEN le.valuenum END) as wbc,
                AVG(CASE WHEN dli.label ILIKE '%lactate%' THEN le.valuenum END) as lactate,
                AVG(CASE WHEN dli.label ILIKE '%creatinine%' THEN le.valuenum END) as creatinine,
                AVG(CASE WHEN dli.label ILIKE '%platelet%' THEN le.valuenum END) as platelets,
                AVG(CASE WHEN dli.label ILIKE '%bilirubin%' THEN le.valuenum END) as bilirubin,
                AVG(CASE WHEN dli.label ILIKE '%c-reactive protein%' THEN le.valuenum END) as crp,
                AVG(CASE WHEN dli.label ILIKE '%procalcitonin%' THEN le.valuenum END) as procalcitonin
            FROM labevents le
            JOIN d_labitems dli ON le.itemid = dli.itemid
            WHERE le.subject_id = {subject_id}
                AND le.valuenum IS NOT NULL
                AND le.valuenum > 0
                AND dli.label ILIKE ANY(ARRAY['%white blood cell%', '%lactate%', '%creatinine%', 
                                             '%platelet%', '%bilirubin%', '%c-reactive protein%',
                                             '%procalcitonin%'])
            GROUP BY le.subject_id
        ),
        sepsis_diagnosis AS (
            SELECT 
                subject_id,
                hadm_id,
                CASE WHEN EXISTS (
                    SELECT 1 FROM diagnoses_icd d 
                    WHERE d.subject_id = a.subject_id 
                    AND d.hadm_id = a.hadm_id
                    AND d.icd9_code IN ('038', '785.52', '995.91', '995.92')
                ) THEN 1 ELSE 0 END as has_sepsis
            FROM admissions a
            WHERE a.subject_id = {subject_id}
        )
        SELECT 
            pi.*,
            COALESCE(lv.heart_rate, 80) as heart_rate,
            COALESCE(lv.systolic_bp, 120) as systolic_bp,
            COALESCE(lv.diastolic_bp, 80) as diastolic_bp,
            COALESCE(lv.temperature, 37) as temperature,
            COALESCE(lv.respiratory_rate, 18) as respiratory_rate,
            COALESCE(lv.spo2, 98) as spo2,
            COALESCE(ll.wbc, 8) as wbc,
            COALESCE(ll.lactate, 1.0) as lactate,
            COALESCE(ll.creatinine, 0.9) as creatinine,
            COALESCE(ll.platelets, 250) as platelets,
            COALESCE(ll.bilirubin, 0.5) as bilirubin,
            COALESCE(ll.crp, 5) as crp,
            COALESCE(ll.procalcitonin, 0.1) as procalcitonin,
            COALESCE(sd.has_sepsis, 0) as has_sepsis
        FROM patient_info pi
        LEFT JOIN latest_vitals lv ON pi.subject_id = lv.subject_id
        LEFT JOIN latest_labs ll ON pi.subject_id = ll.subject_id
        LEFT JOIN sepsis_diagnosis sd ON pi.subject_id = sd.subject_id AND pi.hadm_id = sd.hadm_id
        """
        return query
    
    @staticmethod
    def get_sepsis_patients_data(limit=5000):
        """Get balanced dataset for training (both sepsis and non-sepsis)"""
        query = f"""
        WITH patient_sepsis AS (
            SELECT DISTINCT
                a.subject_id,
                a.hadm_id,
                CASE WHEN EXISTS (
                    SELECT 1 FROM diagnoses_icd d 
                    WHERE d.subject_id = a.subject_id 
                    AND d.hadm_id = a.hadm_id
                    AND d.icd9_code IN ('038', '785.52', '995.91', '995.92')
                ) THEN 1 ELSE 0 END as sepsis_label
            FROM admissions a
        ),
        sepsis_patients AS (
            SELECT * FROM patient_sepsis WHERE sepsis_label = 1
            LIMIT {limit//2}
        ),
        non_sepsis_patients AS (
            SELECT * FROM patient_sepsis WHERE sepsis_label = 0
            LIMIT {limit//2}
        ),
        combined_patients AS (
            SELECT * FROM sepsis_patients
            UNION ALL
            SELECT * FROM non_sepsis_patients
        ),
        patient_features AS (
            SELECT 
                cp.subject_id,
                cp.hadm_id,
                cp.sepsis_label,
                p.gender,
                DATE_PART('year', age(a.admittime, p.dob)) as age,
                a.ethnicity,
                AVG(CASE WHEN di.label ILIKE '%heart rate%' THEN ce.valuenum END) as heart_rate,
                AVG(CASE WHEN di.label ILIKE '%blood pressure%' AND di.label ILIKE '%systolic%' THEN ce.valuenum END) as systolic_bp,
                AVG(CASE WHEN di.label ILIKE '%blood pressure%' AND di.label ILIKE '%diastolic%' THEN ce.valuenum END) as diastolic_bp,
                AVG(CASE WHEN di.label ILIKE '%temperature%' THEN ce.valuenum END) as temperature,
                AVG(CASE WHEN di.label ILIKE '%respiratory rate%' THEN ce.valuenum END) as respiratory_rate,
                AVG(CASE WHEN di.label ILIKE '%oxygen saturation%' THEN ce.valuenum END) as spo2,
                AVG(CASE WHEN dli.label ILIKE '%white blood cell%' THEN le.valuenum END) as wbc,
                AVG(CASE WHEN dli.label ILIKE '%lactate%' THEN le.valuenum END) as lactate,
                AVG(CASE WHEN dli.label ILIKE '%creatinine%' THEN le.valuenum END) as creatinine,
                AVG(CASE WHEN dli.label ILIKE '%platelet%' THEN le.valuenum END) as platelets,
                AVG(CASE WHEN dli.label ILIKE '%bilirubin%' THEN le.valuenum END) as bilirubin
            FROM combined_patients cp
            LEFT JOIN patients p ON cp.subject_id = p.subject_id
            LEFT JOIN admissions a ON cp.subject_id = a.subject_id AND cp.hadm_id = a.hadm_id
            LEFT JOIN chartevents ce ON cp.subject_id = ce.subject_id AND cp.hadm_id = ce.hadm_id
            LEFT JOIN d_items di ON ce.itemid = di.itemid
            LEFT JOIN labevents le ON cp.subject_id = le.subject_id AND cp.hadm_id = le.hadm_id
            LEFT JOIN d_labitems dli ON le.itemid = dli.itemid
            GROUP BY cp.subject_id, cp.hadm_id, cp.sepsis_label, p.gender, p.dob, a.admittime, a.ethnicity
        )
        SELECT * FROM patient_features
        WHERE heart_rate IS NOT NULL 
            AND systolic_bp IS NOT NULL 
            AND temperature IS NOT NULL
        """
        return query
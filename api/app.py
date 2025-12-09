from flask import Flask, render_template, request, jsonify, session, redirect, url_for
from flask_cors import CORS
import pandas as pd
import numpy as np
import json
from datetime import datetime
import os
from werkzeug.security import generate_password_hash, check_password_hash

from config.config import Config
from data.data_loader import DataLoader
from data.feature_engineering import FeatureEngineer
from models.sepsis_predictor import SepsisPredictor
from explainability.shap_explainer import SHAPExplainer
from chatbot.nlp_processor import NLPProcessor
from chatbot.response_generator import ResponseGenerator
from utils.helpers import Helpers
from api.auth import auth_bp
from api.routes import api_bp

# Initialize Flask app
app = Flask(__name__)
app.config.from_object(Config)
CORS(app)

# Register blueprints
app.register_blueprint(auth_bp)
app.register_blueprint(api_bp)

# Initialize components
data_loader = DataLoader()
feature_engineer = FeatureEngineer()
sepsis_predictor = SepsisPredictor()
nlp_processor = NLPProcessor()
response_generator = ResponseGenerator()
helpers = Helpers()

# Global variables
model_trained = False
feature_names = []
shap_explainer = None

@app.route('/')
def index():
    """Home page"""
    return render_template('index.html')

@app.route('/dashboard')
def dashboard():
    """Dashboard based on user role"""
    if 'user_id' not in session:
        return redirect(url_for('auth.login'))
    
    user_role = session.get('role', 'patient')
    
    if user_role == 'clinician':
        return render_template('clinician_dashboard.html')
    else:
        return render_template('patient_dashboard.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Make sepsis prediction"""
    try:
        if 'user_id' not in session:
            return jsonify({'error': 'Not authenticated'}), 401
        
        # Get patient data from request
        data = request.json
        
        # Load model if not already loaded
        global model_trained, feature_names
        if not model_trained:
            sepsis_predictor.load_model('models/saved_models/sepsis_model.pkl')
            model_trained = True
        
        # Prepare patient features
        patient_features = helpers.prepare_patient_features(data)
        
        # Make prediction
        result = sepsis_predictor.predict_single(
            patient_features, 
            feature_names,
            threshold=0.5
        )
        
        # Generate explanation if SHAP explainer is available
        explanation = None
        if shap_explainer:
            patient_df = pd.DataFrame([patient_features], columns=feature_names)
            explanation = shap_explainer.explain_prediction(patient_df)
        
        response = {
            'success': True,
            'prediction': result['prediction'],
            'probability': result['probability'],
            'risk_level': result['risk_level'],
            'explanation': explanation,
            'timestamp': datetime.now().isoformat()
        }
        
        return jsonify(response)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/chat', methods=['POST'])
def chat():
    """Handle chatbot conversations"""
    try:
        if 'user_id' not in session:
            return jsonify({'error': 'Not authenticated'}), 401
        
        data = request.json
        user_message = data.get('message', '')
        
        if not user_message:
            return jsonify({'error': 'No message provided'}), 400
        
        # Process user message
        intent = nlp_processor.extract_intent(user_message)
        entities = nlp_processor.extract_entities(user_message)
        
        # Get patient context if available
        patient_context = session.get('patient_context', {})
        
        # Generate response
        response = response_generator.generate_response(
            intent=intent,
            entities=entities,
            context=patient_context,
            original_message=user_message
        )
        
        # Store conversation in session
        if 'conversation' not in session:
            session['conversation'] = []
        
        session['conversation'].append({
            'user': user_message,
            'bot': response['response'],
            'timestamp': datetime.now().isoformat()
        })
        
        return jsonify(response)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/explain', methods=['POST'])
def explain():
    """Generate detailed explanation for a prediction"""
    try:
        if 'user_id' not in session:
            return jsonify({'error': 'Not authenticated'}), 401
        
        data = request.json
        prediction_id = data.get('prediction_id')
        
        if not prediction_id:
            return jsonify({'error': 'Prediction ID required'}), 400
        
        # Load SHAP explainer if not loaded
        global shap_explainer
        if not shap_explainer:
            shap_explainer = SHAPExplainer(sepsis_predictor.best_model, feature_names)
            shap_explainer.load_explainer('models/saved_models/shap_explainer.pkl')
        
        # Get prediction data (in a real app, this would come from database)
        # For now, return template explanation
        explanation = {
            'top_factors': [
                {'feature': 'lactate', 'impact': 'High', 'direction': 'increases'},
                {'feature': 'heart_rate', 'impact': 'Medium', 'direction': 'increases'},
                {'feature': 'temperature', 'impact': 'Medium', 'direction': 'increases'},
                {'feature': 'wbc', 'impact': 'Low', 'direction': 'increases'},
                {'feature': 'age', 'impact': 'Low', 'direction': 'increases'}
            ],
            'clinical_interpretation': 'The patient shows elevated lactate levels and tachycardia, which are strong indicators of possible sepsis.',
            'recommendations': [
                'Monitor vital signs closely',
                'Consider blood cultures',
                'Administer broad-spectrum antibiotics if infection confirmed',
                'Check for source of infection'
            ]
        }
        
        return jsonify({
            'success': True,
            'explanation': explanation
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/train_model', methods=['POST'])
def train_model_endpoint():
    """Train the sepsis prediction model (admin only)"""
    try:
        if 'user_id' not in session or session.get('role') != 'admin':
            return jsonify({'error': 'Unauthorized'}), 403
        
        print("Starting model training...")
        
        # Load data
        patient_data = data_loader.load_patient_data(limit=5000)
        vitals_data = data_loader.load_vitals_data(limit=10000)
        labs_data = data_loader.load_lab_data(limit=10000)
        
        # Merge data
        merged_data = helpers.merge_patient_data(patient_data, vitals_data, labs_data)
        
        # Engineer features
        engineered_data = feature_engineer.engineer_features(merged_data)
        
        # Prepare for training
        X = engineered_data.drop(['has_sepsis', 'subject_id', 'hadm_id'], axis=1, errors='ignore')
        y = engineered_data['has_sepsis']
        
        global feature_names
        feature_names = list(X.columns)
        
        # Train model
        results, X_test, y_test = sepsis_predictor.train(X, y)
        
        # Save model
        sepsis_predictor.save_model()
        
        # Train SHAP explainer
        X_train, _ = sepsis_predictor.models['random_forest'].train_test_split(X, y, test_size=0.2)
        shap_explainer = SHAPExplainer(sepsis_predictor.best_model, feature_names)
        shap_explainer.fit_explainer(X_train)
        shap_explainer.save_explainer()
        
        global model_trained
        model_trained = True
        
        # Prepare response
        best_model_metrics = results[sepsis_predictor.best_model_name]['metrics']
        
        response = {
            'success': True,
            'message': 'Model trained successfully',
            'best_model': sepsis_predictor.best_model_name,
            'metrics': {
                'accuracy': float(best_model_metrics['accuracy']),
                'precision': float(best_model_metrics['precision']),
                'recall': float(best_model_metrics['recall']),
                'f1_score': float(best_model_metrics['f1_score']),
                'roc_auc': float(best_model_metrics['roc_auc'])
            },
            'feature_count': len(feature_names),
            'top_features': sepsis_predictor.feature_importance.head(10).to_dict('records')
        }
        
        return jsonify(response)
        
    except Exception as e:
        print(f"Training error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/patient/<int:patient_id>')
def get_patient(patient_id):
    """Get patient information"""
    try:
        if 'user_id' not in session:
            return jsonify({'error': 'Not authenticated'}), 401
        
        # Query database for patient
        query = data_loader.sql.get_patient_full_data(patient_id)
        patient_data = data_loader.db.execute_query(query)
        
        if patient_data.empty:
            return jsonify({'error': 'Patient not found'}), 404
        
        # Convert to dictionary
        patient_info = patient_data.iloc[0].to_dict()
        
        # Get recent predictions for this patient
        predictions = []  # This would come from database in real app
        
        response = {
            'patient_info': patient_info,
            'predictions': predictions,
            'has_sepsis_history': bool(patient_info.get('has_sepsis', 0))
        }
        
        return jsonify(response)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Create necessary directories
    os.makedirs('models/saved_models', exist_ok=True)
    os.makedirs('uploads', exist_ok=True)
    
    # Load intent classifier for chatbot
    nlp_processor.load_intent_classifier()
    
    # Try to load pre-trained model
    try:
        sepsis_predictor.load_model('models/saved_models/sepsis_model.pkl')
        model_trained = True
        
        # Load feature names from training
        with open('models/saved_models/feature_names.json', 'r') as f:
            feature_names = json.load(f)
        
        # Load SHAP explainer
        shap_explainer = SHAPExplainer(sepsis_predictor.best_model, feature_names)
        shap_explainer.load_explainer('models/saved_models/shap_explainer.pkl')
        
        print("Pre-trained model loaded successfully")
    except:
        print("No pre-trained model found. Please train the model first.")
    
    app.run(debug=True, port=5000)
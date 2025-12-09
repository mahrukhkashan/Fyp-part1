from flask import Blueprint, request, jsonify, session
from werkzeug.security import generate_password_hash, check_password_hash
from utils.helpers import Helpers
from utils.logger import get_logger
import json

auth_bp = Blueprint('auth', __name__)
logger = get_logger()

# Sample users database (in production, use real database)
users_db = {
    'demo': {
        'id': 1,
        'username': 'demo',
        'email': 'demo@hospital.com',
        'password_hash': Helpers.hash_password('demo123'),
        'role': 'clinician',
        'full_name': 'Demo Doctor',
        'specialization': 'Critical Care',
        'hospital_id': 'HOS001'
    },
    'patient': {
        'id': 2,
        'username': 'patient',
        'email': 'patient@example.com',
        'password_hash': Helpers.hash_password('patient123'),
        'role': 'patient',
        'full_name': 'John Patient',
        'date_of_birth': '1980-05-15',
        'gender': 'Male',
        'blood_type': 'O+'
    },
    'admin': {
        'id': 3,
        'username': 'admin',
        'email': 'admin@hospital.com',
        'password_hash': Helpers.hash_password('admin123'),
        'role': 'admin',
        'full_name': 'System Administrator',
        'hospital_id': 'ADM001'
    }
}

@auth_bp.route('/login', methods=['POST'])
def login():
    """User login endpoint"""
    try:
        data = request.json
        
        if not data or 'username' not in data or 'password' not in data:
            return jsonify({'error': 'Username and password required'}), 400
        
        username = data['username']
        password = data['password']
        
        # Check if user exists
        if username not in users_db:
            logger.error(f"Login failed: User '{username}' not found")
            return jsonify({'error': 'Invalid credentials'}), 401
        
        user = users_db[username]
        
        # Verify password
        if not check_password_hash(user['password_hash'], password):
            logger.error(f"Login failed: Invalid password for user '{username}'")
            return jsonify({'error': 'Invalid credentials'}), 401
        
        # Create session
        session['user_id'] = user['id']
        session['username'] = user['username']
        session['role'] = user['role']
        session['full_name'] = user.get('full_name', '')
        
        # Log successful login
        logger.info(f"User '{username}' logged in successfully")
        
        return jsonify({
            'success': True,
            'message': 'Login successful',
            'user': {
                'id': user['id'],
                'username': user['username'],
                'role': user['role'],
                'full_name': user.get('full_name', ''),
                'specialization': user.get('specialization', ''),
                'hospital_id': user.get('hospital_id', '')
            }
        })
        
    except Exception as e:
        logger.error(f"Login error: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

@auth_bp.route('/register', methods=['POST'])
def register():
    """User registration endpoint"""
    try:
        data = request.json
        
        # Validate required fields
        required_fields = ['username', 'email', 'password', 'role', 'full_name']
        for field in required_fields:
            if field not in data or not data[field]:
                return jsonify({'error': f'Missing required field: {field}'}), 400
        
        username = data['username']
        email = data['email']
        
        # Check if user already exists
        if username in users_db:
            return jsonify({'error': 'Username already exists'}), 409
        
        # Check if email already exists
        for user in users_db.values():
            if user['email'] == email:
                return jsonify({'error': 'Email already registered'}), 409
        
        # Validate role
        valid_roles = ['clinician', 'patient', 'nurse', 'researcher']
        if data['role'] not in valid_roles:
            return jsonify({'error': 'Invalid role'}), 400
        
        # Create new user
        new_user_id = max([u['id'] for u in users_db.values()]) + 1
        
        new_user = {
            'id': new_user_id,
            'username': username,
            'email': email,
            'password_hash': generate_password_hash(data['password']),
            'role': data['role'],
            'full_name': data['full_name'],
            'created_at': Helpers.get_current_timestamp()
        }
        
        # Add additional fields based on role
        if data['role'] == 'clinician':
            new_user['specialization'] = data.get('specialization', '')
            new_user['hospital_id'] = data.get('hospital_id', '')
        elif data['role'] == 'patient':
            new_user['date_of_birth'] = data.get('date_of_birth', '')
            new_user['gender'] = data.get('gender', '')
            new_user['blood_type'] = data.get('blood_type', '')
        
        # Add to database (in production, save to real database)
        users_db[username] = new_user
        
        logger.info(f"New user registered: {username} ({data['role']})")
        
        return jsonify({
            'success': True,
            'message': 'Registration successful',
            'user': {
                'id': new_user_id,
                'username': username,
                'role': data['role'],
                'full_name': data['full_name']
            }
        })
        
    except Exception as e:
        logger.error(f"Registration error: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

@auth_bp.route('/logout', methods=['POST'])
def logout():
    """User logout endpoint"""
    username = session.get('username', 'Unknown')
    
    # Clear session
    session.clear()
    
    logger.info(f"User '{username}' logged out")
    
    return jsonify({
        'success': True,
        'message': 'Logout successful'
    })

@auth_bp.route('/profile', methods=['GET'])
def get_profile():
    """Get user profile"""
    if 'user_id' not in session:
        return jsonify({'error': 'Not authenticated'}), 401
    
    user_id = session['user_id']
    
    # Find user (in production, query database)
    user = None
    for u in users_db.values():
        if u['id'] == user_id:
            user = u
            break
    
    if not user:
        return jsonify({'error': 'User not found'}), 404
    
    # Don't return password hash
    user_data = {k: v for k, v in user.items() if k != 'password_hash'}
    
    return jsonify({
        'success': True,
        'profile': user_data
    })

@auth_bp.route('/profile/update', methods=['PUT'])
def update_profile():
    """Update user profile"""
    if 'user_id' not in session:
        return jsonify({'error': 'Not authenticated'}), 401
    
    user_id = session['user_id']
    data = request.json
    
    # Find user
    user = None
    username = None
    for uname, u in users_db.items():
        if u['id'] == user_id:
            user = u
            username = uname
            break
    
    if not user:
        return jsonify({'error': 'User not found'}), 404
    
    # Update allowed fields
    allowed_fields = ['full_name', 'email', 'specialization', 'hospital_id']
    for field in allowed_fields:
        if field in data:
            user[field] = data[field]
    
    # Update password if provided
    if 'password' in data and data['password']:
        user['password_hash'] = generate_password_hash(data['password'])
    
    logger.info(f"User '{username}' updated profile")
    
    return jsonify({
        'success': True,
        'message': 'Profile updated successfully',
        'profile': {k: v for k, v in user.items() if k != 'password_hash'}
    })

@auth_bp.route('/check_auth', methods=['GET'])
def check_auth():
    """Check if user is authenticated"""
    if 'user_id' in session:
        return jsonify({
            'authenticated': True,
            'user': {
                'id': session.get('user_id'),
                'username': session.get('username'),
                'role': session.get('role'),
                'full_name': session.get('full_name')
            }
        })
    else:
        return jsonify({'authenticated': False})
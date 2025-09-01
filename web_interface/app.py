"""
Flask Web Interface for Autonomous ML Agent
Provides a complete web application with HTML templates and real-time updates
"""

from flask import Flask, render_template, request, jsonify, send_file, session
from flask_socketio import SocketIO, emit, join_room, leave_room
import asyncio
import json
import tempfile
import os
import pandas as pd
import numpy as np
from datetime import datetime
import uuid
import threading
import time

# Import the autonomous ML agent
import sys
sys.path.append('..')
from autonomous_ml.core import AutonomousMLAgent
from autonomous_ml.config import Config

# Initialize Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here'
socketio = SocketIO(app, cors_allowed_origins="*")

# Global agent instance
agent = AutonomousMLAgent()

# Global experiment state
experiments = {}
experiment_lock = threading.Lock()

class ExperimentManager:
    """Manages experiment state and real-time updates"""
    
    def __init__(self):
        self.experiments = {}
        self.active_experiments = set()
    
    def create_experiment(self, config):
        """Create a new experiment"""
        experiment_id = str(uuid.uuid4())
        experiment = {
            'id': experiment_id,
            'config': config,
            'status': 'created',
            'progress': 0,
            'message': 'Experiment created',
            'created_at': datetime.now().isoformat(),
            'results': None,
            'updates': []
        }
        
        with experiment_lock:
            self.experiments[experiment_id] = experiment
        
        return experiment_id
    
    def update_experiment(self, experiment_id, **kwargs):
        """Update experiment state"""
        with experiment_lock:
            if experiment_id in self.experiments:
                self.experiments[experiment_id].update(kwargs)
                self.experiments[experiment_id]['updated_at'] = datetime.now().isoformat()
    
    def get_experiment(self, experiment_id):
        """Get experiment by ID"""
        with experiment_lock:
            return self.experiments.get(experiment_id)
    
    def get_all_experiments(self):
        """Get all experiments"""
        with experiment_lock:
            return dict(self.experiments)
    
    def add_update(self, experiment_id, update):
        """Add real-time update to experiment"""
        with experiment_lock:
            if experiment_id in self.experiments:
                self.experiments[experiment_id]['updates'].append(update)
                # Keep only last 50 updates
                if len(self.experiments[experiment_id]['updates']) > 50:
                    self.experiments[experiment_id]['updates'] = self.experiments[experiment_id]['updates'][-50:]

experiment_manager = ExperimentManager()

@app.route('/')
def index():
    """Main dashboard page"""
    return render_template('index.html')

@app.route('/dashboard')
def dashboard():
    """Dashboard page"""
    return render_template('dashboard.html')

@app.route('/experiments')
def experiments_page():
    """Experiments page"""
    return render_template('experiments.html')

@app.route('/results/<experiment_id>')
def results_page(experiment_id):
    """Results page for specific experiment"""
    experiment = experiment_manager.get_experiment(experiment_id)
    if not experiment:
        return "Experiment not found", 404
    return render_template('results.html', experiment=experiment)

# API Routes
@app.route('/api/upload', methods=['POST'])
def upload_file():
    """Upload dataset file"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not file.filename.endswith('.csv'):
            return jsonify({'error': 'Only CSV files are supported'}), 400
        
        # Save uploaded file
        filename = f"{uuid.uuid4()}_{file.filename}"
        filepath = os.path.join('uploads', filename)
        os.makedirs('uploads', exist_ok=True)
        file.save(filepath)
        
        # Analyze the data
        df = pd.read_csv(filepath)
        
        data_info = {
            'filename': file.filename,
            'filepath': filepath,
            'shape': df.shape,
            'columns': df.columns.tolist(),
            'dtypes': df.dtypes.astype(str).to_dict(),
            'missing_values': df.isnull().sum().to_dict(),
            'numeric_columns': df.select_dtypes(include=[np.number]).columns.tolist(),
            'categorical_columns': df.select_dtypes(include=['object']).columns.tolist(),
            'sample_data': df.head(5).to_dict('records')
        }
        
        return jsonify({
            'status': 'success',
            'message': 'File uploaded successfully',
            'data_info': data_info
        })
        
    except Exception as e:
        return jsonify({'error': f'Error uploading file: {str(e)}'}), 500

@app.route('/api/start_experiment', methods=['POST'])
def start_experiment():
    """Start a new ML experiment"""
    try:
        config = request.json
        
        # Validate required fields
        required_fields = ['data_path', 'target_column']
        for field in required_fields:
            if field not in config:
                return jsonify({'error': f'Missing required field: {field}'}), 400
        
        # Create experiment
        experiment_id = experiment_manager.create_experiment(config)
        
        # Start experiment in background
        def run_experiment():
            asyncio.run(run_experiment_async(experiment_id, config))
        
        thread = threading.Thread(target=run_experiment)
        thread.daemon = True
        thread.start()
        
        return jsonify({
            'status': 'success',
            'message': 'Experiment started successfully',
            'experiment_id': experiment_id
        })
        
    except Exception as e:
        return jsonify({'error': f'Error starting experiment: {str(e)}'}), 500

@app.route('/api/experiment/<experiment_id>')
def get_experiment(experiment_id):
    """Get experiment status and details"""
    experiment = experiment_manager.get_experiment(experiment_id)
    if not experiment:
        return jsonify({'error': 'Experiment not found'}), 404
    
    return jsonify(experiment)

@app.route('/api/experiments')
def get_all_experiments():
    """Get all experiments"""
    experiments = experiment_manager.get_all_experiments()
    return jsonify(experiments)

@app.route('/api/results/<experiment_id>')
def get_results(experiment_id):
    """Get experiment results"""
    experiment = experiment_manager.get_experiment(experiment_id)
    if not experiment:
        return jsonify({'error': 'Experiment not found'}), 404
    
    if experiment['status'] != 'completed':
        return jsonify({'error': 'Experiment not completed yet'}), 400
    
    return jsonify(experiment['results'])

@app.route('/api/download/<experiment_id>')
def download_results(experiment_id):
    """Download experiment results"""
    experiment = experiment_manager.get_experiment(experiment_id)
    if not experiment:
        return "Experiment not found", 404
    
    if experiment['status'] != 'completed':
        return "Experiment not completed yet", 400
    
    # Create temporary file
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as tmp_file:
        json.dump(experiment['results'], tmp_file, indent=2, default=str)
        tmp_path = tmp_file.name
    
    return send_file(
        tmp_path,
        as_attachment=True,
        download_name=f"ml_results_{experiment_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
        mimetype='application/json'
    )

# WebSocket Events
@socketio.on('connect')
def handle_connect():
    """Handle client connection"""
    print(f'Client connected: {request.sid}')
    emit('connected', {'message': 'Connected to Autonomous ML Agent'})

@socketio.on('disconnect')
def handle_disconnect():
    """Handle client disconnection"""
    print(f'Client disconnected: {request.sid}')

@socketio.on('join_experiment')
def handle_join_experiment(data):
    """Join an experiment room for real-time updates"""
    experiment_id = data.get('experiment_id')
    if experiment_id:
        join_room(f'experiment_{experiment_id}')
        emit('joined_experiment', {'experiment_id': experiment_id})

@socketio.on('leave_experiment')
def handle_leave_experiment(data):
    """Leave an experiment room"""
    experiment_id = data.get('experiment_id')
    if experiment_id:
        leave_room(f'experiment_{experiment_id}')
        emit('left_experiment', {'experiment_id': experiment_id})

@socketio.on('get_experiment_status')
def handle_get_status(data):
    """Get current experiment status"""
    experiment_id = data.get('experiment_id')
    if experiment_id:
        experiment = experiment_manager.get_experiment(experiment_id)
        if experiment:
            emit('experiment_status', experiment)
        else:
            emit('error', {'message': 'Experiment not found'})

async def run_experiment_async(experiment_id, config):
    """Run experiment asynchronously with real-time updates"""
    try:
        # Update experiment status
        experiment_manager.update_experiment(
            experiment_id,
            status='running',
            progress=0,
            message='Initializing experiment...'
        )
        
        # Emit status update
        socketio.emit('experiment_update', {
            'experiment_id': experiment_id,
            'status': 'running',
            'progress': 0,
            'message': 'Initializing experiment...'
        }, room=f'experiment_{experiment_id}')
        
        # Progress callback
        async def progress_callback(message, progress):
            experiment_manager.update_experiment(
                experiment_id,
                progress=progress,
                message=message
            )
            
            # Add to updates
            update = {
                'timestamp': datetime.now().isoformat(),
                'message': message,
                'progress': progress
            }
            experiment_manager.add_update(experiment_id, update)
            
            # Emit real-time update
            socketio.emit('experiment_progress', {
                'experiment_id': experiment_id,
                'message': message,
                'progress': progress,
                'timestamp': datetime.now().isoformat()
            }, room=f'experiment_{experiment_id}')
        
        # Run the experiment
        results = await agent.run_pipeline_async(
            data_path=config['data_path'],
            target_column=config['target_column'],
            feature_columns=config.get('feature_columns'),
            time_budget=config.get('time_budget', 900),
            target_metric=config.get('target_metric', 'accuracy'),
            llm_provider=config.get('llm_provider', 'openai'),
            progress_callback=progress_callback
        )
        
        # Update experiment with results
        experiment_manager.update_experiment(
            experiment_id,
            status='completed',
            progress=100,
            message='Experiment completed successfully!',
            results=results,
            completed_at=datetime.now().isoformat()
        )
        
        # Emit completion
        socketio.emit('experiment_completed', {
            'experiment_id': experiment_id,
            'results': results,
            'message': 'Experiment completed successfully!'
        }, room=f'experiment_{experiment_id}')
        
    except Exception as e:
        # Update experiment with error
        experiment_manager.update_experiment(
            experiment_id,
            status='failed',
            message=f'Experiment failed: {str(e)}',
            error=str(e),
            failed_at=datetime.now().isoformat()
        )
        
        # Emit error
        socketio.emit('experiment_failed', {
            'experiment_id': experiment_id,
            'error': str(e),
            'message': f'Experiment failed: {str(e)}'
        }, room=f'experiment_{experiment_id}')

if __name__ == '__main__':
    # Create uploads directory
    os.makedirs('uploads', exist_ok=True)
    
    # Run the Flask-SocketIO app
    socketio.run(app, debug=True, host='0.0.0.0', port=5000)

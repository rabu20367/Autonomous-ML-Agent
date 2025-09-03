"""
FastAPI Backend for Autonomous ML Agent
Provides real-time WebSocket communication and REST API endpoints
"""

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, UploadFile, File, Depends, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import asyncio
import json
import tempfile
import os
from typing import Dict, List, Any, Optional
from datetime import datetime
from pathlib import Path
import uvicorn
import pandas as pd
import numpy as np

try:
    from .core import AutonomousMLAgent
    from .config import Config
except ImportError:
    # Handle direct execution
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from autonomous_ml.core import AutonomousMLAgent
    from autonomous_ml.config import Config

# Initialize FastAPI app
app = FastAPI(
    title="Autonomous ML Agent API",
    description="Real-time ML training and monitoring API",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# WebSocket connection manager
class ConnectionManager:
    """Manages WebSocket connections for real-time updates"""
    
    def __init__(self):
        self.active_connections: List[WebSocket] = []
        self.connection_data: Dict[WebSocket, Dict] = {}
    
    async def connect(self, websocket: WebSocket, client_id: str = None):
        """Accept a new WebSocket connection"""
        await websocket.accept()
        self.active_connections.append(websocket)
        self.connection_data[websocket] = {
            'client_id': client_id or f"client_{len(self.active_connections)}",
            'connected_at': datetime.now(),
            'last_activity': datetime.now()
        }
        print(f"Client {self.connection_data[websocket]['client_id']} connected")
    
    def disconnect(self, websocket: WebSocket):
        """Remove a WebSocket connection"""
        if websocket in self.active_connections:
            client_id = self.connection_data.get(websocket, {}).get('client_id', 'unknown')
            self.active_connections.remove(websocket)
            if websocket in self.connection_data:
                del self.connection_data[websocket]
            print(f"Client {client_id} disconnected")
    
    async def send_personal_message(self, message: str, websocket: WebSocket):
        """Send a message to a specific WebSocket connection"""
        try:
            await websocket.send_text(message)
            if websocket in self.connection_data:
                self.connection_data[websocket]['last_activity'] = datetime.now()
        except Exception as e:
            print(f"Error sending personal message: {e}")
            self.disconnect(websocket)
    
    async def broadcast(self, message: str, exclude: WebSocket = None):
        """Broadcast a message to all connected clients"""
        disconnected = []
        for connection in self.active_connections:
            if connection != exclude:
                try:
                    await connection.send_text(message)
                    if connection in self.connection_data:
                        self.connection_data[connection]['last_activity'] = datetime.now()
                except Exception as e:
                    print(f"Error broadcasting to client: {e}")
                    disconnected.append(connection)
        
        # Remove disconnected clients
        for connection in disconnected:
            self.disconnect(connection)
    
    def get_connection_info(self) -> Dict[str, Any]:
        """Get information about active connections"""
        return {
            'total_connections': len(self.active_connections),
            'connections': [
                {
                    'client_id': data['client_id'],
                    'connected_at': data['connected_at'].isoformat(),
                    'last_activity': data['last_activity'].isoformat()
                }
                for data in self.connection_data.values()
            ]
        }

# Global instances
manager = ConnectionManager()
agent = AutonomousMLAgent()
security = HTTPBearer()

# Global experiment state
current_experiment = None
experiment_lock = asyncio.Lock()

# Authentication
def verify_api_key(api_key: str = Header(None)):
    """Verify API key for protected endpoints"""
    expected_key = os.getenv('API_KEY', 'default-dev-key')
    if not api_key or api_key != expected_key:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")
    return api_key

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Autonomous ML Agent API",
        "version": "1.0.0",
        "status": "running",
        "websocket_connections": manager.get_connection_info(),
        "endpoints": {
            "websocket": "/ws",
            "start_experiment": "/start_experiment",
            "experiment_status": "/experiment_status",
            "results": "/results",
            "upload": "/upload",
            "health": "/health"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "agent_status": agent.experiment_status,
        "active_connections": len(manager.active_connections)
    }

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    """Upload a dataset file"""
    try:
        # Validate file type and name
        if not file.filename or not file.filename.endswith('.csv'):
            raise HTTPException(status_code=400, detail="Only CSV files are supported")
        
        # Sanitize filename to prevent path traversal
        import re
        safe_filename = re.sub(r'[^a-zA-Z0-9._-]', '_', file.filename)
        if not safe_filename:
            safe_filename = f"upload_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        
        # Create secure upload directory
        upload_dir = Path(tempfile.gettempdir()) / "autonomous_ml_uploads"
        upload_dir.mkdir(exist_ok=True, mode=0o700)
        
        # Save uploaded file with secure path
        tmp_path = upload_dir / safe_filename
        content = await file.read()
        
        # Validate file size (max 100MB)
        if len(content) > 100 * 1024 * 1024:
            raise HTTPException(status_code=400, detail="File too large. Maximum size is 100MB")
        
        with open(tmp_path, 'wb') as f:
            f.write(content)
        
        # Load and analyze the data
        df = pd.read_csv(tmp_path)
        
        # Basic data analysis
        data_info = {
            "filename": file.filename,
            "shape": df.shape,
            "columns": df.columns.tolist(),
            "dtypes": df.dtypes.to_dict(),
            "missing_values": df.isnull().sum().to_dict(),
            "numeric_columns": df.select_dtypes(include=[np.number]).columns.tolist(),
            "categorical_columns": df.select_dtypes(include=['object']).columns.tolist(),
            "file_id": safe_filename  # Use safe filename instead of full path
        }
        
        return JSONResponse(content={
            "status": "success",
            "message": "File uploaded successfully",
            "data_info": data_info
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error uploading file: {str(e)}")

@app.post("/start_experiment")
async def start_experiment(experiment_config: Dict[str, Any]):
    """Start a new ML experiment"""
    global current_experiment
    
    async with experiment_lock:
        if current_experiment and current_experiment.get('status') == 'running':
            raise HTTPException(status_code=409, detail="An experiment is already running")
        
        try:
            # Validate configuration
            required_fields = ['data_path', 'target_column']
            for field in required_fields:
                if field not in experiment_config:
                    raise HTTPException(status_code=400, detail=f"Missing required field: {field}")
            
            # Start experiment in background
            current_experiment = {
                'status': 'starting',
                'config': experiment_config,
                'started_at': datetime.now().isoformat(),
                'progress': 0,
                'message': 'Initializing experiment...'
            }
            
            # Create background task
            asyncio.create_task(run_experiment_async(experiment_config))
            
            return JSONResponse(content={
                "status": "started",
                "message": "Experiment started successfully",
                "experiment_id": current_experiment.get('id', 'unknown')
            })
            
        except Exception as e:
            current_experiment = None
            raise HTTPException(status_code=500, detail=f"Error starting experiment: {str(e)}")

@app.get("/experiment_status")
async def get_experiment_status():
    """Get current experiment status"""
    global current_experiment
    
    if not current_experiment:
        return JSONResponse(content={
            "status": "no_experiment",
            "message": "No experiment is currently running"
        })
    
    return JSONResponse(content={
        "status": current_experiment.get('status', 'unknown'),
        "progress": current_experiment.get('progress', 0),
        "message": current_experiment.get('message', ''),
        "started_at": current_experiment.get('started_at'),
        "agent_status": agent.experiment_status,
        "real_time_updates": agent.real_time_updates[-10:]  # Last 10 updates
    })

@app.get("/results")
async def get_results():
    """Get experiment results"""
    global current_experiment
    
    if not current_experiment or current_experiment.get('status') != 'completed':
        raise HTTPException(status_code=404, detail="No completed experiment results available")
    
    if not agent.results:
        raise HTTPException(status_code=404, detail="No results available from agent")
    
    return JSONResponse(content=agent.results)

@app.get("/results/download")
async def download_results():
    """Download experiment results as JSON file"""
    global current_experiment
    
    if not current_experiment or current_experiment.get('status') != 'completed':
        raise HTTPException(status_code=404, detail="No completed experiment results available")
    
    if not agent.results:
        raise HTTPException(status_code=404, detail="No results available from agent")
    
    # Create temporary file
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as tmp_file:
        json.dump(agent.results, tmp_file, indent=2, default=str)
        tmp_path = tmp_file.name
    
    return FileResponse(
        path=tmp_path,
        filename=f"ml_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
        media_type="application/json"
    )

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket, client_id: str = None):
    """WebSocket endpoint for real-time updates"""
    await manager.connect(websocket, client_id)
    
    try:
        # Send initial connection confirmation
        await manager.send_personal_message(
            json.dumps({
                "type": "connection",
                "message": "Connected to Autonomous ML Agent",
                "client_id": manager.connection_data[websocket]['client_id'],
                "timestamp": datetime.now().isoformat()
            }),
            websocket
        )
        
        # Keep connection alive and handle incoming messages
        while True:
            try:
                # Wait for messages from client
                data = await websocket.receive_text()
                message = json.loads(data)
                
                # Handle different message types
                if message.get('type') == 'ping':
                    await manager.send_personal_message(
                        json.dumps({
                            "type": "pong",
                            "timestamp": datetime.now().isoformat()
                        }),
                        websocket
                    )
                elif message.get('type') == 'get_status':
                    await manager.send_personal_message(
                        json.dumps({
                            "type": "status",
                            "agent_status": agent.experiment_status,
                            "progress": agent.experiment_progress,
                            "updates": agent.real_time_updates[-5:],
                            "timestamp": datetime.now().isoformat()
                        }),
                        websocket
                    )
                
            except WebSocketDisconnect:
                break
            except Exception as e:
                print(f"Error handling WebSocket message: {e}")
                await manager.send_personal_message(
                    json.dumps({
                        "type": "error",
                        "message": f"Error processing message: {str(e)}",
                        "timestamp": datetime.now().isoformat()
                    }),
                    websocket
                )
    
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        print(f"WebSocket error: {e}")
        manager.disconnect(websocket)

async def run_experiment_async(config: Dict[str, Any]):
    """Run experiment asynchronously with real-time updates"""
    global current_experiment
    
    try:
        # Update experiment status
        current_experiment['status'] = 'running'
        current_experiment['progress'] = 0
        current_experiment['message'] = 'Starting experiment...'
        
        # Progress callback for WebSocket updates
        async def progress_callback(message: str, progress: float):
            current_experiment['message'] = message
            current_experiment['progress'] = progress
            
            # Broadcast update to all connected clients
            update_message = {
                "type": "progress",
                "message": message,
                "progress": progress,
                "timestamp": datetime.now().isoformat()
            }
            await manager.broadcast(json.dumps(update_message))
        
        # Run experiment
        results = await agent.run_pipeline_async(
            data_path=config["data_path"],
            target_column=config["target_column"],
            feature_columns=config.get("feature_columns"),
            time_budget=config.get("time_budget", 900),
            target_metric=config.get("target_metric", "accuracy"),
            llm_provider=config.get("llm_provider", "openai"),
            progress_callback=progress_callback
        )
        
        # Update experiment status
        current_experiment['status'] = 'completed'
        current_experiment['progress'] = 100
        current_experiment['message'] = 'Experiment completed successfully!'
        current_experiment['completed_at'] = datetime.now().isoformat()
        current_experiment['results'] = results
        
        # Send completion update
        completion_message = {
            "type": "completed",
            "message": "Experiment completed successfully!",
            "results": results,
            "timestamp": datetime.now().isoformat()
        }
        await manager.broadcast(json.dumps(completion_message))
        
    except Exception as e:
        # Update experiment status
        current_experiment['status'] = 'failed'
        current_experiment['message'] = f'Experiment failed: {str(e)}'
        current_experiment['error'] = str(e)
        current_experiment['failed_at'] = datetime.now().isoformat()
        
        # Send error update
        error_message = {
            "type": "error",
            "message": f"Experiment failed: {str(e)}",
            "timestamp": datetime.now().isoformat()
        }
        await manager.broadcast(json.dumps(error_message))

@app.get("/connections")
async def get_connections():
    """Get information about active WebSocket connections"""
    return JSONResponse(content=manager.get_connection_info())

@app.post("/deploy")
async def deploy_model(api_key: str = Depends(verify_api_key)):
    """Generate deployment package for the best model (requires API key)"""
    try:
        if not agent.results or not agent.results.get('best_model'):
            raise HTTPException(status_code=404, detail="No trained model available. Run experiment first.")
        
        # Generate deployment package
        deployment_path = agent.generate_deployment_package()
        
        # Log deployment request
        print(f"Deployment package generated by user with API key: {api_key[:8]}...")
        
        return JSONResponse(content={
            "status": "success",
            "message": "Deployment package generated successfully",
            "deployment_path": deployment_path,
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating deployment package: {str(e)}")

@app.post("/broadcast")
async def broadcast_message(message: Dict[str, Any]):
    """Broadcast a message to all connected clients"""
    try:
        await manager.broadcast(json.dumps({
            "type": "broadcast",
            "message": message.get("message", ""),
            "data": message.get("data", {}),
            "timestamp": datetime.now().isoformat()
        }))
        
        return JSONResponse(content={
            "status": "success",
            "message": "Message broadcasted successfully",
            "recipients": len(manager.active_connections)
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error broadcasting message: {str(e)}")

# Error handlers
@app.exception_handler(404)
async def not_found_handler(request, exc):
    return JSONResponse(
        status_code=404,
        content={"error": "Not found", "message": "The requested resource was not found"}
    )

@app.exception_handler(500)
async def internal_error_handler(request, exc):
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "message": "An unexpected error occurred"}
    )

if __name__ == "__main__":
    uvicorn.run(
        "autonomous_ml.web_api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )

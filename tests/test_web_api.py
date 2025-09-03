"""
Test web API functionality
"""

import pytest
import tempfile
import os
from pathlib import Path
from fastapi.testclient import TestClient
from autonomous_ml.web_api import app


class TestWebAPI:
    """Test web API endpoints"""
    
    def setup_method(self):
        """Set up test client"""
        self.client = TestClient(app)
    
    def test_root_endpoint(self):
        """Test root endpoint"""
        response = self.client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert "version" in data
        assert "endpoints" in data
    
    def test_health_check(self):
        """Test health check endpoint"""
        response = self.client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "timestamp" in data
    
    def test_upload_csv_file(self):
        """Test CSV file upload"""
        # Create a temporary CSV file
        csv_content = "feature1,feature2,target\n1,2,0\n3,4,1\n5,6,0"
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as tmp_file:
            tmp_file.write(csv_content)
            tmp_file.flush()
            
            with open(tmp_file.name, 'rb') as f:
                response = self.client.post("/upload", files={"file": ("test.csv", f, "text/csv")})
        
        # Clean up
        os.unlink(tmp_file.name)
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert "data_info" in data
        assert data["data_info"]["shape"] == [3, 3]
        assert "file_id" in data["data_info"]
    
    def test_upload_invalid_file(self):
        """Test upload of invalid file type"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as tmp_file:
            tmp_file.write("not a csv file")
            tmp_file.flush()
            
            with open(tmp_file.name, 'rb') as f:
                response = self.client.post("/upload", files={"file": ("test.txt", f, "text/plain")})
        
        # Clean up
        os.unlink(tmp_file.name)
        
        assert response.status_code == 400
        data = response.json()
        assert "Only CSV files are supported" in data["detail"]
    
    def test_upload_large_file(self):
        """Test upload of file that's too large"""
        # Create a large CSV file (simulate > 100MB)
        large_content = "feature1,feature2,target\n" + "1,2,0\n" * 1000000
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as tmp_file:
            tmp_file.write(large_content)
            tmp_file.flush()
            
            with open(tmp_file.name, 'rb') as f:
                response = self.client.post("/upload", files={"file": ("large.csv", f, "text/csv")})
        
        # Clean up
        os.unlink(tmp_file.name)
        
        # Should fail due to file size
        assert response.status_code == 400
        data = response.json()
        assert "File too large" in data["detail"]
    
    def test_deploy_endpoint_without_auth(self):
        """Test deploy endpoint without authentication"""
        response = self.client.post("/deploy")
        assert response.status_code == 401
    
    def test_deploy_endpoint_with_auth(self):
        """Test deploy endpoint with authentication"""
        # Set API key in environment
        os.environ['API_KEY'] = 'test-key'
        
        headers = {"api-key": "test-key"}
        response = self.client.post("/deploy", headers=headers)
        
        # Should fail because no model is trained, but auth should pass
        assert response.status_code == 404
        data = response.json()
        assert "No trained model available" in data["detail"]
    
    def test_connections_endpoint(self):
        """Test connections endpoint"""
        response = self.client.get("/connections")
        assert response.status_code == 200
        data = response.json()
        assert "total_connections" in data
        assert "connections" in data
    
    def test_broadcast_endpoint(self):
        """Test broadcast endpoint"""
        message = {"message": "Test broadcast", "data": {"test": True}}
        response = self.client.post("/broadcast", json=message)
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"

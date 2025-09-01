"""
Experience Memory - Vector-based storage and retrieval of ML experiment experiences
"""

import json
import sqlite3
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
import chromadb
from sentence_transformers import SentenceTransformer
import hashlib
from pathlib import Path

class ExperienceMemory:
    """Vector-based memory system for ML experiment experiences"""
    
    def __init__(self, db_path: str = "experiences.db", collection_name: str = "ml_experiences"):
        self.db_path = db_path
        self.collection_name = collection_name
        
        # Initialize ChromaDB
        self.chroma_client = chromadb.PersistentClient(path="./chroma_db")
        self.collection = self.chroma_client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        
        # Initialize embedding model
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Initialize SQLite for metadata
        self._init_sqlite_db()
    
    def _init_sqlite_db(self):
        """Initialize SQLite database for structured metadata"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS experiments (
                id TEXT PRIMARY KEY,
                dataset_hash TEXT,
                experiment_plan TEXT,
                results TEXT,
                timestamp TEXT,
                success BOOLEAN,
                performance_score REAL
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def _create_experience_embedding(self, experiment_plan: Dict[str, Any], 
                                   results: Dict[str, Any]) -> List[float]:
        """Create rich embedding for experiment experience"""
        
        # Create text representation
        text_parts = []
        
        # Model and strategy
        text_parts.append(f"Model: {experiment_plan.get('model_selection', {}).get('primary_model', 'unknown')}")
        text_parts.append(f"Preprocessing: {experiment_plan.get('preprocessing_strategy', {})}")
        
        # Results
        if results.get('success', False):
            metrics = results.get('performance_metrics', {})
            text_parts.append(f"Success: accuracy={metrics.get('test_accuracy', 0):.3f}")
        else:
            text_parts.append("Failed experiment")
        
        # Create embedding
        text = " ".join(text_parts)
        embedding = self.embedding_model.encode(text)
        
        return embedding.tolist()
    
    def store_experiment(self, experiment_plan: Dict[str, Any], results: Dict[str, Any]):
        """Store experiment experience"""
        
        # Create unique ID
        exp_id = experiment_plan.get('experiment_id', f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        
        # Create embedding
        embedding = self._create_experience_embedding(experiment_plan, results)
        
        # Store in ChromaDB
        self.collection.add(
            embeddings=[embedding],
            documents=[json.dumps({
                'experiment_plan': experiment_plan,
                'results': results
            })],
            metadatas=[{
                'experiment_id': exp_id,
                'success': results.get('success', False),
                'performance_score': results.get('performance_metrics', {}).get('test_accuracy', 0.0),
                'timestamp': datetime.now().isoformat()
            }],
            ids=[exp_id]
        )
        
        # Store in SQLite
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT OR REPLACE INTO experiments 
            (id, dataset_hash, experiment_plan, results, timestamp, success, performance_score)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (
            exp_id,
            hashlib.md5(json.dumps(experiment_plan, sort_keys=True).encode()).hexdigest(),
            json.dumps(experiment_plan),
            json.dumps(results),
            datetime.now().isoformat(),
            results.get('success', False),
            results.get('performance_metrics', {}).get('test_accuracy', 0.0)
        ))
        
        conn.commit()
        conn.close()
    
    def find_similar_experiences(self, dataset_profile: Dict[str, Any], 
                               n_results: int = 5) -> List[Dict[str, Any]]:
        """Find similar past experiences"""
        
        # Create query embedding based on dataset characteristics
        query_text = f"Dataset: {dataset_profile.get('shape', 'unknown')} samples, {dataset_profile.get('target_type', 'unknown')} target"
        query_embedding = self.embedding_model.encode(query_text).tolist()
        
        # Search in ChromaDB
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where={"success": True}  # Only successful experiments
        )
        
        # Format results
        similar_experiences = []
        if results['documents'] and results['documents'][0]:
            for i, doc in enumerate(results['documents'][0]):
                experience_data = json.loads(doc)
                metadata = results['metadatas'][0][i]
                
                similar_experiences.append({
                    'experiment_plan': experience_data['experiment_plan'],
                    'results': experience_data['results'],
                    'similarity_score': 1 - results['distances'][0][i],
                    'metadata': metadata
                })
        
        return similar_experiences
    
    def get_global_insights(self) -> Dict[str, Any]:
        """Get global insights from all stored experiences"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Overall statistics
        cursor.execute('''
            SELECT COUNT(*), AVG(performance_score), COUNT(DISTINCT dataset_hash)
            FROM experiments 
            WHERE success = 1
        ''')
        
        overall_stats = cursor.fetchone()
        
        # Best performing models
        cursor.execute('''
            SELECT json_extract(experiment_plan, '$.model_selection.primary_model') as model,
                   AVG(performance_score) as avg_score,
                   COUNT(*) as count
            FROM experiments 
            WHERE success = 1
            GROUP BY model
            ORDER BY avg_score DESC
            LIMIT 10
        ''')
        
        model_performance = cursor.fetchall()
        
        conn.close()
        
        return {
            'total_experiments': overall_stats[0] or 0,
            'average_performance': overall_stats[1] or 0.0,
            'unique_datasets': overall_stats[2] or 0,
            'top_models': [{'model': row[0], 'avg_score': row[1], 'count': row[2]} for row in model_performance]
        }

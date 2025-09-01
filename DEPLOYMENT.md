# Deployment Guide

## Quick Start

1. **Install Dependencies**
   ```bash
   pip install -e .
   ```

2. **Set Environment Variables**
   ```bash
   cp env.example .env
   # Edit .env with your API keys
   ```

3. **Run Basic Example**
   ```bash
   python examples/basic_usage.py
   ```

## Production Deployment

### 1. Environment Setup
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate     # Windows

# Install package
pip install -e .
```

### 2. Configuration
```bash
# Set LLM API key
export LLM_API_KEY=your_api_key_here

# Or use .env file
echo "LLM_API_KEY=your_api_key_here" > .env
```

### 3. CLI Usage
```bash
# Train models
autonomous-ml train --data-path data.csv --target-column target

# Analyze dataset
autonomous-ml analyze --data-path data.csv --target-column target

# Generate deployment package
autonomous-ml deploy --results-dir results/
```

### 4. Python API Usage
```python
from autonomous_ml import AutonomousMLAgent
from autonomous_ml.config import Config

# Initialize
config = Config()
agent = AutonomousMLAgent(config)

# Run pipeline
results = agent.run_pipeline('data.csv', 'target')
```

## Docker Deployment

### 1. Build Image
```bash
docker build -t autonomous-ml-agent .
```

### 2. Run Container
```bash
docker run -e LLM_API_KEY=your_key autonomous-ml-agent
```

## Cloud Deployment

### AWS Lambda
- Package as Lambda layer
- Use serverless framework
- Configure API Gateway

### Google Cloud Run
- Build container image
- Deploy to Cloud Run
- Configure environment variables

### Azure Container Instances
- Build and push to ACR
- Deploy container instance
- Configure secrets

## Monitoring

### Health Checks
- `/health` endpoint for service health
- Metrics endpoint for performance monitoring
- Log aggregation for debugging

### Performance Monitoring
- Track experiment success rates
- Monitor API response times
- Alert on failures

## Security

### API Key Management
- Use environment variables
- Rotate keys regularly
- Use secret management services

### Data Privacy
- Encrypt data at rest
- Use secure connections
- Implement access controls

## Scaling

### Horizontal Scaling
- Use load balancers
- Implement auto-scaling
- Cache frequently used data

### Vertical Scaling
- Monitor resource usage
- Optimize memory usage
- Tune hyperparameters

## Troubleshooting

### Common Issues
1. **API Key Errors**: Check environment variables
2. **Memory Issues**: Reduce dataset size or increase memory
3. **Timeout Errors**: Increase timeout settings
4. **Import Errors**: Check Python path and dependencies

### Debug Mode
```bash
export LOG_LEVEL=DEBUG
autonomous-ml train --data-path data.csv --target-column target
```

### Logs
- Check `logs/` directory for detailed logs
- Use `tail -f logs/autonomous_ml.log` for real-time monitoring

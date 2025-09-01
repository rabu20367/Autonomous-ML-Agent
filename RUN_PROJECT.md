# 🚀 How to Run the Autonomous ML Agent Project

## 🎯 Complete Browser-Based ML Platform

Your Autonomous ML Agent is now a **complete browser-based platform** where you can interact with every feature through a beautiful web interface!

## 🌐 What You Can Do in the Browser

### ✅ **Fully Interactive Features:**
- **📊 Upload Datasets:** Drag & drop CSV files directly in the browser
- **🎯 Configure Experiments:** Set target metrics, time budgets, model selection
- **🧪 Run ML Training:** Start experiments with one click
- **📈 Real-time Monitoring:** Watch experiments progress live
- **🏆 View Results:** Interactive leaderboards and performance charts
- **💡 Get AI Insights:** Natural language explanations of results
- **📥 Download Models:** Export trained models and deployment packages
- **🔄 Manage Experiments:** Start, pause, stop, and reset experiments

## 🚀 Quick Start (3 Steps)

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Generate Example Datasets
```bash
python examples/datasets/generate_all_datasets.py
```

### Step 3: Start Web Interface
```bash
# Option A: Streamlit Dashboard (Recommended)
python -m streamlit run autonomous_ml/web_dashboard.py

# Option B: Flask Web App
cd web_interface && python app.py

# Option C: FastAPI Backend
python -m uvicorn autonomous_ml.web_api:app --reload
```

### Step 4: Open Your Browser
- **Streamlit:** http://localhost:8501
- **Flask:** http://localhost:5000
- **FastAPI:** http://localhost:8000

## 🎮 Interactive Web Features

### 1. **Data Upload & Analysis**
- Drag and drop CSV files
- Automatic data profiling
- Interactive data preview
- Missing value detection
- Feature type analysis

### 2. **Experiment Configuration**
- LLM provider selection (OpenAI, Anthropic, Google)
- Time budget adjustment (5-60 minutes)
- Target metric selection (accuracy, precision, recall, F1, ROC AUC)
- Model selection (enable/disable specific models)
- Feature engineering options

### 3. **Real-time Experiment Monitoring**
- Live progress updates
- Performance trend visualization
- Experiment status tracking
- Real-time notifications
- WebSocket-based updates

### 4. **Results & Analysis**
- Interactive model leaderboard
- Performance comparison charts
- Feature importance visualization
- AI-generated insights
- Download options for models and results

### 5. **Model Deployment**
- Generate FastAPI services
- Create Docker configurations
- Download deployment packages
- Production-ready code generation

## 🎯 Example Workflow

1. **Open Browser:** Go to http://localhost:8501
2. **Upload Dataset:** Drag & drop `examples/datasets/iris.csv`
3. **Configure:** Select target column, set time budget to 15 minutes
4. **Start Training:** Click "Start Training" button
5. **Monitor Progress:** Watch real-time updates in the monitoring section
6. **View Results:** Explore the leaderboard and performance charts
7. **Download Model:** Export the best model for production use

## 🔧 Advanced Options

### Multiple Web Interfaces

**Streamlit Dashboard (Primary):**
```bash
python -m streamlit run autonomous_ml/web_dashboard.py --server.port 8501
```
- Complete ML pipeline management
- Real-time monitoring
- Interactive visualizations
- Best for end-to-end workflow

**Flask Web Application:**
```bash
cd web_interface && python app.py
```
- Full web app with HTML templates
- WebSocket support
- Custom web development
- Best for custom interfaces

**FastAPI Backend:**
```bash
python -m uvicorn autonomous_ml.web_api:app --reload
```
- REST API endpoints
- WebSocket support
- API documentation at /docs
- Best for API integration

### Automated Startup Scripts

**Windows:**
```cmd
start_web_interface.bat
```

**Linux/Mac:**
```bash
./start_web_interface.sh
```

**Python (Cross-platform):**
```bash
python start_web_interface.py streamlit
```

## 📊 Example Datasets

The project includes three ready-to-use datasets:

1. **Iris Dataset** (`examples/datasets/iris.csv`)
   - Classification problem
   - 150 samples, 4 features
   - Target: Species classification

2. **Titanic Dataset** (`examples/datasets/titanic.csv`)
   - Classification problem
   - 1000 samples, 12 features
   - Target: Survival prediction

3. **Housing Dataset** (`examples/datasets/housing.csv`)
   - Regression problem
   - 1000 samples, 13 features
   - Target: House price prediction

## 🎨 Web Interface Features

### Modern UI/UX
- **Responsive Design:** Works on desktop, tablet, and mobile
- **Real-time Updates:** Live progress tracking and notifications
- **Interactive Charts:** Plotly-based visualizations
- **Drag & Drop:** Intuitive file upload
- **Progress Bars:** Visual experiment progress
- **Status Indicators:** Connection and experiment status

### Advanced Functionality
- **WebSocket Communication:** Real-time bidirectional updates
- **File Validation:** Automatic CSV format checking
- **Error Handling:** Comprehensive error messages and recovery
- **Session Management:** Persistent experiment state
- **Download Options:** Multiple export formats

## 🔍 Troubleshooting

### Common Issues

1. **Port Already in Use:**
```bash
# Find and kill process using port 8501
lsof -ti:8501 | xargs kill -9  # Linux/Mac
netstat -ano | findstr :8501   # Windows
```

2. **Missing Dependencies:**
```bash
pip install -r requirements.txt
```

3. **File Upload Issues:**
- Ensure file is CSV format
- Check file size (max 100MB)
- Verify file encoding (UTF-8)

4. **WebSocket Connection Issues:**
- Check firewall settings
- Ensure ports are open
- Try refreshing the browser

### Performance Tips

1. **Large Datasets:**
- Use data sampling for initial experiments
- Increase time budget for comprehensive analysis
- Consider feature selection for high-dimensional data

2. **Slow Experiments:**
- Reduce number of experiments
- Use faster models (Logistic Regression, kNN)
- Decrease cross-validation folds

## 🎯 What Makes This Special

### 🌟 **Unique Features:**
- **Self-Learning:** Learns from past experiments
- **Real-time Monitoring:** Watch experiments progress live
- **AI-Powered Insights:** Natural language explanations
- **Production-Ready:** Generates deployable models
- **Multi-Interface:** Streamlit, Flask, and FastAPI options
- **Interactive:** Every feature accessible through the browser

### 🚀 **Innovation:**
- **Structured Planning:** LLM generates JSON plans, not raw code
- **Experience Memory:** Vector database for meta-learning
- **Adaptive Strategy:** Uses past experiences to improve
- **Real-time Updates:** WebSocket-based live monitoring
- **Comprehensive UI:** Complete browser-based workflow

## 🎉 Ready to Start?

1. **Install:** `pip install -r requirements.txt`
2. **Generate Data:** `python examples/datasets/generate_all_datasets.py`
3. **Start Web:** `python -m streamlit run autonomous_ml/web_dashboard.py`
4. **Open Browser:** http://localhost:8501
5. **Upload Dataset:** Drag & drop a CSV file
6. **Start Training:** Click "Start Training"
7. **Watch Magic:** See AI train models in real-time!

---

**🎯 Your Autonomous ML Agent is now a complete browser-based platform where you can interact with every feature through a beautiful, modern web interface!**

**🌐 Everything is accessible through the browser - no coding required!**

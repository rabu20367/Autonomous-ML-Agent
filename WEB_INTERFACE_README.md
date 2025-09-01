# 🌐 Autonomous ML Agent - Web Interface

Complete browser-based interface for the Autonomous ML Agent with real-time monitoring, interactive visualizations, and comprehensive ML pipeline management.

## 🚀 Quick Start

### Option 1: Automated Setup (Recommended)

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

### Option 2: Manual Setup

1. **Install Dependencies:**
```bash
pip install -r requirements.txt
```

2. **Generate Example Datasets:**
```bash
python examples/datasets/generate_all_datasets.py
```

3. **Start Web Interface:**
```bash
# Streamlit Dashboard (Recommended)
python -m streamlit run autonomous_ml/web_dashboard.py

# Flask Web App
cd web_interface && python app.py

# FastAPI Backend
python -m uvicorn autonomous_ml.web_api:app --reload
```

## 🎯 Available Interfaces

### 1. Streamlit Dashboard (Primary)
- **URL:** http://localhost:8501
- **Features:** Complete ML pipeline management, real-time monitoring, interactive visualizations
- **Best for:** End-to-end ML workflow

### 2. Flask Web Application
- **URL:** http://localhost:5000
- **Features:** Full web app with HTML templates, WebSocket support
- **Best for:** Custom web development

### 3. FastAPI Backend
- **URL:** http://localhost:8000
- **API Docs:** http://localhost:8000/docs
- **Features:** REST API, WebSocket support, real-time updates
- **Best for:** API integration, mobile apps

## 🎮 Features

### 📊 Data Management
- **Drag & Drop Upload:** Upload CSV files directly through the browser
- **Data Preview:** Interactive data exploration and visualization
- **Data Quality Check:** Automatic missing value detection and analysis
- **Feature Selection:** Choose target and feature columns with visual feedback

### 🧪 Experiment Control
- **Real-time Monitoring:** Watch experiments progress in real-time
- **Progress Tracking:** Visual progress bars and status updates
- **Experiment Management:** Start, pause, stop, and reset experiments
- **Configuration:** Adjust time budgets, target metrics, and model selection

### 📈 Results & Analysis
- **Interactive Leaderboard:** Compare model performance with sorting and filtering
- **Performance Charts:** Visualize model performance trends and comparisons
- **Feature Importance:** Understand which features matter most
- **AI Insights:** Natural language explanations of results and recommendations

### 📥 Download Options
- **Model Export:** Download trained models for deployment
- **Results Export:** Export experiment results as JSON
- **Deployment Package:** Generate production-ready FastAPI services

## 🎨 Interface Screenshots

### Main Dashboard
- Clean, modern interface with intuitive navigation
- Real-time experiment monitoring
- Interactive data visualization
- Comprehensive results analysis

### Data Upload
- Drag-and-drop file upload
- Automatic data profiling
- Feature type detection
- Missing value analysis

### Experiment Monitoring
- Live progress updates
- Performance trend visualization
- Real-time status notifications
- Experiment history tracking

## 🔧 Configuration

### LLM Providers
- **OpenAI:** GPT-3.5, GPT-4
- **Anthropic:** Claude
- **Google:** Gemini

### Target Metrics
- **Accuracy:** Overall correctness
- **Precision:** True positive rate
- **Recall:** Sensitivity
- **F1-Score:** Harmonic mean of precision and recall
- **ROC AUC:** Area under the ROC curve

### Time Budgets
- **Quick:** 5-10 minutes (3-5 experiments)
- **Standard:** 15-30 minutes (8-12 experiments)
- **Comprehensive:** 45-60 minutes (15-20 experiments)

## 📊 Example Datasets

The project includes three example datasets for testing:

### 1. Iris Dataset
- **Type:** Classification
- **Samples:** 150
- **Features:** 4 (sepal length, sepal width, petal length, petal width)
- **Target:** Species (setosa, versicolor, virginica)

### 2. Titanic Dataset
- **Type:** Classification
- **Samples:** 1000
- **Features:** 12 (passenger info, ticket details, etc.)
- **Target:** Survival (0/1)

### 3. Housing Dataset
- **Type:** Regression
- **Samples:** 1000
- **Features:** 13 (crime rate, rooms, age, etc.)
- **Target:** House price

## 🚀 Advanced Usage

### Custom Model Selection
```python
# Enable/disable specific models in the web interface
models = {
    "Logistic Regression": True,
    "Random Forest": True,
    "Gradient Boosting": True,
    "k-Nearest Neighbors": True,
    "Multi-layer Perceptron": True,
    "Support Vector Machine": True
}
```

### Feature Engineering Options
- **Feature Selection:** Automatic feature importance-based selection
- **Polynomial Features:** Generate polynomial feature combinations
- **Interaction Features:** Create feature interaction terms

### Real-time Updates
The web interface provides real-time updates through WebSocket connections:
- Experiment progress
- Model training status
- Performance metrics
- Error notifications

## 🔍 Troubleshooting

### Common Issues

1. **Port Already in Use:**
```bash
# Kill process using port 8501
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
- Ensure ports 5000, 8000, 8501 are open
- Try refreshing the browser

### Performance Optimization

1. **Large Datasets:**
- Use data sampling for initial experiments
- Increase time budget for comprehensive analysis
- Consider feature selection for high-dimensional data

2. **Slow Experiments:**
- Reduce number of experiments
- Use faster models (Logistic Regression, kNN)
- Decrease cross-validation folds

## 📱 Mobile Support

The web interface is fully responsive and works on:
- **Desktop:** Full feature set
- **Tablet:** Optimized layout
- **Mobile:** Core functionality

## 🔒 Security

- **File Upload:** Automatic file type validation
- **API Keys:** Secure environment variable storage
- **CORS:** Configurable cross-origin resource sharing
- **Input Validation:** Comprehensive input sanitization

## 🎯 Best Practices

1. **Start Small:** Begin with example datasets
2. **Monitor Progress:** Watch real-time updates
3. **Analyze Results:** Review insights and recommendations
4. **Export Models:** Download for production use
5. **Iterate:** Run multiple experiments with different configurations

## 🆘 Support

- **Documentation:** Check the main README.md
- **Issues:** Report bugs on GitHub
- **Examples:** Use the provided example datasets
- **Community:** Join discussions on GitHub Discussions

## 🎉 What's Next?

1. **Upload a Dataset:** Use the example datasets or your own CSV files
2. **Configure Experiment:** Set target metric, time budget, and model selection
3. **Start Training:** Watch the AI train multiple models in real-time
4. **Analyze Results:** Explore the leaderboard and insights
5. **Download Models:** Export the best model for production use

---

**Ready to start?** Run `python start_web_interface.py streamlit` and open http://localhost:8501 in your browser! 🚀

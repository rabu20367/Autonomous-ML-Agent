/**
 * Autonomous ML Agent - Main JavaScript
 * Handles all interactive functionality and real-time updates
 */

class AutonomousMLAgent {
    constructor() {
        this.socket = null;
        this.currentExperiment = null;
        this.uploadedData = null;
        this.realTimeUpdates = [];
        this.initializeEventListeners();
        this.connectWebSocket();
        this.initializeUI();
    }

    initializeEventListeners() {
        // File upload
        const uploadArea = document.getElementById('upload-area');
        const fileInput = document.getElementById('file-input');
        
        uploadArea.addEventListener('click', () => fileInput.click());
        uploadArea.addEventListener('dragover', this.handleDragOver.bind(this));
        uploadArea.addEventListener('dragleave', this.handleDragLeave.bind(this));
        uploadArea.addEventListener('drop', this.handleDrop.bind(this));
        fileInput.addEventListener('change', this.handleFileSelect.bind(this));

        // Configuration updates
        document.getElementById('time-budget').addEventListener('input', this.updateTimeBudget.bind(this));

        // Experiment controls
        document.getElementById('start-btn').addEventListener('click', this.startExperiment.bind(this));
        document.getElementById('pause-btn').addEventListener('click', this.pauseExperiment.bind(this));
        document.getElementById('stop-btn').addEventListener('click', this.stopExperiment.bind(this));
        document.getElementById('reset-btn').addEventListener('click', this.resetExperiment.bind(this));

        // Download buttons
        document.getElementById('download-model').addEventListener('click', this.downloadModel.bind(this));
        document.getElementById('download-results').addEventListener('click', this.downloadResults.bind(this));
        document.getElementById('download-deployment').addEventListener('click', this.downloadDeployment.bind(this));

        // Target column selection
        document.getElementById('target-column').addEventListener('change', this.updateFeatureColumns.bind(this));
    }

    initializeUI() {
        // Initialize time budget display
        this.updateTimeBudget();
        
        // Set initial connection status
        this.updateConnectionStatus('Connecting...');
    }

    connectWebSocket() {
        this.socket = io();
        
        this.socket.on('connect', () => {
            console.log('Connected to server');
            this.updateConnectionStatus('Connected');
        });
        
        this.socket.on('disconnect', () => {
            console.log('Disconnected from server');
            this.updateConnectionStatus('Disconnected');
        });
        
        this.socket.on('connected', (data) => {
            console.log('Server connection confirmed:', data.message);
        });
        
        this.socket.on('experiment_progress', (data) => {
            this.handleExperimentProgress(data);
        });
        
        this.socket.on('experiment_completed', (data) => {
            this.handleExperimentCompleted(data);
        });
        
        this.socket.on('experiment_failed', (data) => {
            this.handleExperimentFailed(data);
        });
        
        this.socket.on('experiment_update', (data) => {
            this.handleExperimentUpdate(data);
        });
    }

    updateConnectionStatus(status) {
        const statusElement = document.getElementById('connection-status');
        statusElement.textContent = status;
        statusElement.className = `status-value ${status.toLowerCase()}`;
    }

    updateTimeBudget() {
        const slider = document.getElementById('time-budget');
        const display = document.getElementById('time-budget-value');
        display.textContent = slider.value;
    }

    handleDragOver(e) {
        e.preventDefault();
        e.currentTarget.classList.add('drag-over');
    }

    handleDragLeave(e) {
        e.preventDefault();
        e.currentTarget.classList.remove('drag-over');
    }

    handleDrop(e) {
        e.preventDefault();
        e.currentTarget.classList.remove('drag-over');
        const files = e.dataTransfer.files;
        if (files.length > 0) {
            this.processFile(files[0]);
        }
    }

    handleFileSelect(e) {
        const file = e.target.files[0];
        if (file) {
            this.processFile(file);
        }
    }

    async processFile(file) {
        if (!file.name.endsWith('.csv')) {
            this.showNotification('Please upload a CSV file', 'error');
            return;
        }

        this.showLoading('Uploading and analyzing file...');

        try {
            const formData = new FormData();
            formData.append('file', file);

            const response = await fetch('/api/upload', {
                method: 'POST',
                body: formData
            });

            if (response.ok) {
                const result = await response.json();
                this.uploadedData = result.data_info;
                this.displayDataPreview(result.data_info);
                this.enableStartButton();
                this.showNotification('File uploaded successfully!', 'success');
            } else {
                const error = await response.json();
                throw new Error(error.error || 'Upload failed');
            }
        } catch (error) {
            console.error('Error uploading file:', error);
            this.showNotification(`Error uploading file: ${error.message}`, 'error');
        } finally {
            this.hideLoading();
        }
    }

    displayDataPreview(dataInfo) {
        const preview = document.getElementById('data-preview');
        const table = document.getElementById('data-table');
        
        // Update data info
        document.getElementById('data-rows').textContent = dataInfo.shape[0];
        document.getElementById('data-columns').textContent = dataInfo.shape[1];
        document.getElementById('data-missing').textContent = Object.values(dataInfo.missing_values).reduce((a, b) => a + b, 0);
        
        // Create table
        let tableHTML = '<table><thead><tr>';
        dataInfo.columns.forEach(col => {
            tableHTML += `<th>${col}</th>`;
        });
        tableHTML += '</tr></thead><tbody>';
        
        dataInfo.sample_data.forEach(row => {
            tableHTML += '<tr>';
            dataInfo.columns.forEach(col => {
                tableHTML += `<td>${row[col] || '-'}</td>`;
            });
            tableHTML += '</tr>';
        });
        tableHTML += '</tbody></table>';
        
        table.innerHTML = tableHTML;
        preview.style.display = 'block';
        
        // Update target configuration
        this.updateTargetConfiguration(dataInfo);
    }

    updateTargetConfiguration(dataInfo) {
        const targetSelect = document.getElementById('target-column');
        const featureSelect = document.getElementById('feature-columns');
        
        // Clear existing options
        targetSelect.innerHTML = '<option value="">Select target column...</option>';
        featureSelect.innerHTML = '';
        
        // Add column options
        dataInfo.columns.forEach(col => {
            const targetOption = document.createElement('option');
            targetOption.value = col;
            targetOption.textContent = col;
            targetSelect.appendChild(targetOption);
            
            const featureOption = document.createElement('option');
            featureOption.value = col;
            featureOption.textContent = col;
            featureSelect.appendChild(featureOption);
        });
        
        // Show target configuration
        document.getElementById('target-config').style.display = 'block';
    }

    updateFeatureColumns() {
        const targetColumn = document.getElementById('target-column').value;
        const featureSelect = document.getElementById('feature-columns');
        
        // Clear existing selections
        Array.from(featureSelect.options).forEach(option => {
            option.selected = option.value !== targetColumn;
        });
    }

    enableStartButton() {
        document.getElementById('start-btn').disabled = false;
    }

    async startExperiment() {
        if (!this.uploadedData) {
            this.showNotification('Please upload a dataset first', 'error');
            return;
        }

        const targetColumn = document.getElementById('target-column').value;
        if (!targetColumn) {
            this.showNotification('Please select a target column', 'error');
            return;
        }

        const featureColumns = Array.from(document.getElementById('feature-columns').selectedOptions)
            .map(option => option.value);

        const config = {
            data_path: this.uploadedData.filepath,
            target_column: targetColumn,
            feature_columns: featureColumns,
            time_budget: parseInt(document.getElementById('time-budget').value) * 60,
            target_metric: document.getElementById('target-metric').value,
            llm_provider: document.getElementById('llm-provider').value
        };

        try {
            this.showLoading('Starting experiment...');
            
            const response = await fetch('/api/start_experiment', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(config)
            });

            if (response.ok) {
                const result = await response.json();
                this.currentExperiment = result.experiment_id;
                
                // Join experiment room for real-time updates
                this.socket.emit('join_experiment', { experiment_id: this.currentExperiment });
                
                this.updateExperimentControls(true);
                this.showNotification('Experiment started successfully!', 'success');
            } else {
                const error = await response.json();
                throw new Error(error.error || 'Failed to start experiment');
            }
        } catch (error) {
            console.error('Error starting experiment:', error);
            this.showNotification(`Error starting experiment: ${error.message}`, 'error');
        } finally {
            this.hideLoading();
        }
    }

    pauseExperiment() {
        // Implementation for pausing experiment
        this.showNotification('Experiment paused', 'info');
    }

    stopExperiment() {
        // Implementation for stopping experiment
        this.showNotification('Experiment stopped', 'warning');
        this.updateExperimentControls(false);
    }

    resetExperiment() {
        this.currentExperiment = null;
        this.realTimeUpdates = [];
        this.updateExperimentControls(false);
        this.updateProgress(0, 'Ready to start');
        this.clearResults();
        this.showNotification('Experiment state reset', 'info');
    }

    updateExperimentControls(running) {
        document.getElementById('start-btn').disabled = running;
        document.getElementById('pause-btn').disabled = !running;
        document.getElementById('stop-btn').disabled = !running;
        document.getElementById('reset-btn').disabled = running;
    }

    handleExperimentProgress(data) {
        this.updateProgress(data.progress, data.message);
        this.addRealTimeUpdate(data);
        this.updateLastUpdate();
    }

    handleExperimentCompleted(data) {
        this.updateProgress(100, 'Experiment completed successfully!');
        this.updateExperimentControls(false);
        this.displayResults(data.results);
        this.showNotification('Experiment completed successfully!', 'success');
    }

    handleExperimentFailed(data) {
        this.updateProgress(0, `Experiment failed: ${data.error}`);
        this.updateExperimentControls(false);
        this.showNotification(`Experiment failed: ${data.error}`, 'error');
    }

    handleExperimentUpdate(data) {
        this.addRealTimeUpdate(data);
    }

    updateProgress(progress, message) {
        const progressFill = document.getElementById('progress-fill');
        const progressText = document.getElementById('progress-text');
        
        progressFill.style.width = `${progress}%`;
        progressText.textContent = message;
    }

    addRealTimeUpdate(update) {
        this.realTimeUpdates.push(update);
        this.updateRealTimeDisplay();
        this.updatePerformanceChart();
    }

    updateRealTimeDisplay() {
        const updatesList = document.getElementById('recent-updates');
        const recentUpdates = this.realTimeUpdates.slice(-5);
        
        updatesList.innerHTML = recentUpdates.map(update => {
            const timestamp = new Date(update.timestamp).toLocaleTimeString();
            const statusClass = update.message.includes('completed') ? 'success' : 
                              update.message.includes('failed') ? 'error' : '';
            
            return `
                <div class="update-item ${statusClass}">
                    <strong>${timestamp}</strong><br>
                    ${update.message}
                </div>
            `;
        }).join('');
    }

    updatePerformanceChart() {
        const scores = this.realTimeUpdates
            .filter(update => update.progress && update.progress > 0)
            .map(update => update.progress);
        
        if (scores.length > 1) {
            const trace = {
                x: Array.from({length: scores.length}, (_, i) => i + 1),
                y: scores,
                type: 'scatter',
                mode: 'lines+markers',
                name: 'Progress',
                line: { color: '#667eea' },
                marker: { color: '#764ba2' }
            };
            
            const layout = {
                title: 'Experiment Progress',
                xaxis: { title: 'Update' },
                yaxis: { title: 'Progress (%)' },
                margin: { t: 40, l: 40, r: 40, b: 40 }
            };
            
            Plotly.newPlot('performance-chart', [trace], layout, {responsive: true});
        }
    }

    displayResults(results) {
        // Show results section
        document.getElementById('results-section').style.display = 'block';
        
        // Update summary metrics
        this.updateSummaryMetrics(results);
        
        // Display leaderboard
        this.displayLeaderboard(results.leaderboard);
        
        // Display performance charts
        this.displayPerformanceCharts(results.leaderboard);
        
        // Display best model details
        this.displayBestModelDetails(results.best_model);
        
        // Display AI insights
        this.displayAIInsights(results.insights);
    }

    updateSummaryMetrics(results) {
        const experiments = results.experiment_history || [];
        const successful = experiments.filter(exp => exp.success).length;
        const bestScore = Math.max(...experiments.map(exp => exp.performance_metrics?.test_accuracy || 0));
        const totalTime = experiments.reduce((sum, exp) => sum + (exp.execution_time || 0), 0);
        
        document.getElementById('total-experiments').textContent = experiments.length;
        document.getElementById('best-score').textContent = bestScore.toFixed(4);
        document.getElementById('successful-experiments').textContent = successful;
        document.getElementById('total-time').textContent = `${totalTime.toFixed(2)}s`;
    }

    displayLeaderboard(leaderboard) {
        const table = document.getElementById('leaderboard-table');
        
        if (!leaderboard || leaderboard.length === 0) {
            table.innerHTML = '<p>No results available</p>';
            return;
        }
        
        let tableHTML = '<table><thead><tr><th>Rank</th><th>Model</th><th>Accuracy</th><th>Precision</th><th>Recall</th><th>F1</th><th>Time</th></tr></thead><tbody>';
        
        leaderboard.forEach((model, index) => {
            tableHTML += `
                <tr>
                    <td>${index + 1}</td>
                    <td>${model.model}</td>
                    <td>${model.test_accuracy?.toFixed(4) || '-'}</td>
                    <td>${model.test_precision?.toFixed(4) || '-'}</td>
                    <td>${model.test_recall?.toFixed(4) || '-'}</td>
                    <td>${model.test_f1?.toFixed(4) || '-'}</td>
                    <td>${model.execution_time?.toFixed(2) || '-'}s</td>
                </tr>
            `;
        });
        
        tableHTML += '</tbody></table>';
        table.innerHTML = tableHTML;
    }

    displayPerformanceCharts(leaderboard) {
        if (!leaderboard || leaderboard.length === 0) return;
        
        // Performance comparison chart
        const models = leaderboard.map(m => m.model);
        const accuracies = leaderboard.map(m => m.test_accuracy || 0);
        
        const trace1 = {
            x: models,
            y: accuracies,
            type: 'bar',
            name: 'Accuracy',
            marker: { color: '#667eea' }
        };
        
        const layout1 = {
            title: 'Model Performance Comparison',
            xaxis: { title: 'Model' },
            yaxis: { title: 'Accuracy' },
            margin: { t: 40, l: 40, r: 40, b: 40 }
        };
        
        Plotly.newPlot('performance-comparison-chart', [trace1], layout1, {responsive: true});
        
        // Time vs Performance chart
        const times = leaderboard.map(m => m.execution_time || 0);
        
        const trace2 = {
            x: times,
            y: accuracies,
            mode: 'markers',
            type: 'scatter',
            name: 'Models',
            text: models,
            marker: { 
                size: 10,
                color: accuracies,
                colorscale: 'Viridis',
                showscale: true
            }
        };
        
        const layout2 = {
            title: 'Training Time vs Performance',
            xaxis: { title: 'Training Time (s)' },
            yaxis: { title: 'Accuracy' },
            margin: { t: 40, l: 40, r: 40, b: 40 }
        };
        
        Plotly.newPlot('time-performance-chart', [trace2], layout2, {responsive: true});
    }

    displayBestModelDetails(bestModel) {
        const details = document.getElementById('best-model-details');
        
        if (!bestModel || !bestModel.result) {
            details.innerHTML = '<p>No best model details available</p>';
            return;
        }
        
        const model = bestModel.result;
        const performance = bestModel.performance || {};
        
        details.innerHTML = `
            <div class="model-details">
                <h4>Model Information</h4>
                <p><strong>Model:</strong> ${model.plan?.model_selection?.primary_model || 'Unknown'}</p>
                <p><strong>Experiment ID:</strong> ${model.experiment_id || 'Unknown'}</p>
                <p><strong>Success:</strong> ${model.success ? 'Yes' : 'No'}</p>
                <p><strong>Execution Time:</strong> ${model.execution_time?.toFixed(2) || 'Unknown'} seconds</p>
                
                <h4>Performance Metrics</h4>
                ${Object.entries(performance).map(([metric, value]) => 
                    `<p><strong>${metric}:</strong> ${value?.toFixed(4) || 'N/A'}</p>`
                ).join('')}
            </div>
        `;
    }

    displayAIInsights(insights) {
        const insightsDiv = document.getElementById('ai-insights');
        
        if (!insights) {
            insightsDiv.innerHTML = '<p>No insights available</p>';
            return;
        }
        
        let html = '';
        
        if (insights.performance_summary) {
            html += '<h4>Performance Summary</h4>';
            html += `<p><strong>Best Accuracy:</strong> ${insights.performance_summary.best_accuracy?.toFixed(4) || 'N/A'}</p>`;
            html += `<p><strong>Model Diversity:</strong> ${insights.performance_summary.model_diversity || 'N/A'}</p>`;
            html += `<p><strong>Average Accuracy:</strong> ${insights.performance_summary.average_accuracy?.toFixed(4) || 'N/A'}</p>`;
        }
        
        if (insights.recommendations) {
            html += '<h4>Recommendations</h4><ul>';
            insights.recommendations.forEach(rec => {
                html += `<li>${rec}</li>`;
            });
            html += '</ul>';
        }
        
        if (insights.technical_insights) {
            html += '<h4>Technical Insights</h4><ul>';
            insights.technical_insights.forEach(insight => {
                html += `<li>${insight}</li>`;
            });
            html += '</ul>';
        }
        
        insightsDiv.innerHTML = html || '<p>No insights available</p>';
    }

    clearResults() {
        document.getElementById('results-section').style.display = 'none';
        document.getElementById('recent-updates').innerHTML = '';
        document.getElementById('performance-chart').innerHTML = '';
    }

    async downloadModel() {
        if (!this.currentExperiment) {
            this.showNotification('No experiment results to download', 'error');
            return;
        }
        
        try {
            const response = await fetch(`/api/download/${this.currentExperiment}`);
            if (response.ok) {
                const blob = await response.blob();
                const url = window.URL.createObjectURL(blob);
                const a = document.createElement('a');
                a.href = url;
                a.download = `ml_model_${this.currentExperiment}.json`;
                document.body.appendChild(a);
                a.click();
                window.URL.revokeObjectURL(url);
                document.body.removeChild(a);
                this.showNotification('Model downloaded successfully!', 'success');
            } else {
                throw new Error('Download failed');
            }
        } catch (error) {
            this.showNotification(`Error downloading model: ${error.message}`, 'error');
        }
    }

    async downloadResults() {
        if (!this.currentExperiment) {
            this.showNotification('No experiment results to download', 'error');
            return;
        }
        
        try {
            const response = await fetch(`/api/download/${this.currentExperiment}`);
            if (response.ok) {
                const blob = await response.blob();
                const url = window.URL.createObjectURL(blob);
                const a = document.createElement('a');
                a.href = url;
                a.download = `results_${this.currentExperiment}.json`;
                document.body.appendChild(a);
                a.click();
                window.URL.revokeObjectURL(url);
                document.body.removeChild(a);
                this.showNotification('Results downloaded successfully!', 'success');
            } else {
                throw new Error('Download failed');
            }
        } catch (error) {
            this.showNotification(`Error downloading results: ${error.message}`, 'error');
        }
    }

    downloadDeployment() {
        this.showNotification('Deployment package download not yet implemented', 'info');
    }

    showLoading(message) {
        document.getElementById('loading-message').textContent = message;
        document.getElementById('loading-overlay').style.display = 'flex';
    }

    hideLoading() {
        document.getElementById('loading-overlay').style.display = 'none';
    }

    showNotification(message, type = 'info') {
        // Create notification element
        const notification = document.createElement('div');
        notification.className = `notification notification-${type}`;
        notification.textContent = message;
        
        // Style the notification
        notification.style.cssText = `
            position: fixed;
            top: 20px;
            right: 20px;
            padding: 15px 20px;
            border-radius: 8px;
            color: white;
            font-weight: 600;
            z-index: 1001;
            animation: slideIn 0.3s ease-out;
            max-width: 300px;
        `;
        
        // Set background color based on type
        const colors = {
            success: '#28a745',
            error: '#dc3545',
            warning: '#ffc107',
            info: '#17a2b8'
        };
        notification.style.backgroundColor = colors[type] || colors.info;
        
        // Add to page
        document.body.appendChild(notification);
        
        // Remove after 5 seconds
        setTimeout(() => {
            notification.style.animation = 'slideOut 0.3s ease-in';
            setTimeout(() => {
                if (notification.parentNode) {
                    notification.parentNode.removeChild(notification);
                }
            }, 300);
        }, 5000);
    }

    updateLastUpdate() {
        const now = new Date();
        document.getElementById('last-update').textContent = now.toLocaleTimeString();
    }
}

// Add CSS for notifications
const style = document.createElement('style');
style.textContent = `
    @keyframes slideIn {
        from { transform: translateX(100%); opacity: 0; }
        to { transform: translateX(0); opacity: 1; }
    }
    
    @keyframes slideOut {
        from { transform: translateX(0); opacity: 1; }
        to { transform: translateX(100%); opacity: 0; }
    }
`;
document.head.appendChild(style);

// Initialize the application when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    window.mlAgent = new AutonomousMLAgent();
});

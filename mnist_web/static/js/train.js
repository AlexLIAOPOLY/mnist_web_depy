// Training page script for MNIST Web App
document.addEventListener('DOMContentLoaded', function() {
    console.log('MNIST Web App Training Page initialized');

    // Elements
    const trainForm = document.getElementById('training-form');
    const startButton = document.getElementById('start-training');
    const stopButton = document.getElementById('stop-training');
    const resetButton = document.getElementById('reset-model');
    const trainingStatusValue = document.getElementById('training-status-value');
    const currentEpoch = document.getElementById('current-epoch');
    const bestAccuracy = document.getElementById('best-accuracy');
    const currentLoss = document.getElementById('current-loss');
    const progressBar = document.getElementById('progress-bar');
    const progressPercentage = document.getElementById('progress-percentage');
    const accuracyChart = document.getElementById('accuracy-chart');
    const lossChart = document.getElementById('loss-chart');
    const samplePredictionsContainer = document.getElementById('sample-predictions-container');
    
    let trainingInterval;
    let charts = {};
    
    // Initialize charts if they exist
    if (accuracyChart && lossChart) {
        initializeCharts();
    }
    
    // Start training button
    if (startButton) {
        startButton.addEventListener('click', function() {
            startTraining();
        });
    }
    
    // Stop training button
    if (stopButton) {
        stopButton.addEventListener('click', function() {
            stopTraining();
        });
    }
    
    // Reset model button
    if (resetButton) {
        resetButton.addEventListener('click', function() {
            resetTraining();
        });
    }
    
    // Listen for language changes
    document.addEventListener('languageChanged', function(e) {
        updateChartLabels();
    });
    
    function startTraining() {
        if (startButton) startButton.disabled = true;
        if (stopButton) stopButton.disabled = false;
        if (resetButton) resetButton.disabled = true;
        
        // Update status text
        if (trainingStatusValue) {
            trainingStatusValue.textContent = window.MNIST.i18n.getTranslation('training');
        }
        
        // Get form data
        const formData = {
            epochs: parseInt(document.getElementById('epochs').value) || 10,
            batch_size: parseInt(document.getElementById('batch-size').value) || 32,
            learning_rate: parseFloat(document.getElementById('learning-rate').value) || 0.001,
            hidden_size: parseInt(document.getElementById('hidden-layer-size').value) || 128,
            activation: document.getElementById('activation').value || 'relu'
        };
        
        // Send request to start training
        fetch('/api/train', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(formData),
        })
        .then(response => response.json())
        .then(data => {
            if (data.status === 'success') {
                // Start polling for training status
                trainingInterval = setInterval(updateTrainingStatus, 1000);
            } else {
                alert('Error: ' + data.message);
                resetTrainingControls();
            }
        })
        .catch(error => {
            console.error('Error:', error);
            alert('An error occurred while starting training');
            resetTrainingControls();
        });
    }
    
    function stopTraining() {
        fetch('/api/stop_training', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
        })
        .then(response => response.json())
        .then(data => {
            if (data.status === 'success') {
                console.log('Training stopped successfully');
                if (trainingStatusValue) {
                    trainingStatusValue.textContent = window.MNIST.i18n.getTranslation('stopped');
                }
                resetTrainingControls();
            } else {
                console.error('Error stopping training:', data.message);
            }
        })
        .catch(error => {
            console.error('Error:', error);
        });
    }
    
    function resetTraining() {
        // Reset UI elements
        if (trainingStatusValue) trainingStatusValue.textContent = window.MNIST.i18n.getTranslation('not_started');
        if (currentEpoch) currentEpoch.textContent = '0/0';
        if (bestAccuracy) bestAccuracy.textContent = '0%';
        if (currentLoss) currentLoss.textContent = '-';
        if (progressBar) progressBar.style.width = '0%';
        if (progressPercentage) progressPercentage.textContent = '0%';
        
        // Reset charts
        if (charts.accuracy) {
            charts.accuracy.data.labels = [];
            charts.accuracy.data.datasets[0].data = [];
            charts.accuracy.data.datasets[1].data = [];
            charts.accuracy.update();
        }
        
        if (charts.loss) {
            charts.loss.data.labels = [];
            charts.loss.data.datasets[0].data = [];
            charts.loss.data.datasets[1].data = [];
            charts.loss.update();
        }
        
        // Clear sample predictions
        if (samplePredictionsContainer) {
            samplePredictionsContainer.innerHTML = '';
        }
    }
    
    function resetTrainingControls() {
        if (startButton) startButton.disabled = false;
        if (stopButton) stopButton.disabled = true;
        if (resetButton) resetButton.disabled = false;
        if (trainingInterval) clearInterval(trainingInterval);
    }
    
    function updateTrainingStatus() {
        fetch('/api/training_status')
        .then(response => response.json())
        .then(data => {
            // Update status text
            if (trainingStatusValue) {
                if (!data.is_training) {
                    trainingStatusValue.textContent = window.MNIST.i18n.getTranslation('completed');
                } else {
                    trainingStatusValue.textContent = window.MNIST.i18n.getTranslation('training');
                }
            }
            
            // Update current epoch
            if (currentEpoch && data.log && data.log.length > 0) {
                const currentEpochNum = data.log[data.log.length - 1].epoch;
                const totalEpochs = data.total_epochs || currentEpochNum;
                currentEpoch.textContent = `${currentEpochNum}/${totalEpochs}`;
            }
            
            // Update best accuracy
            if (bestAccuracy && data.accuracy !== undefined) {
                bestAccuracy.textContent = `${(data.accuracy * 100).toFixed(2)}%`;
            }
            
            // Update current loss
            if (currentLoss && data.log && data.log.length > 0) {
                const currentLossValue = data.log[data.log.length - 1].loss;
                currentLoss.textContent = currentLossValue.toFixed(4);
            }
            
            // Update progress bar
            if (progressBar && progressPercentage) {
                const progress = data.progress * 100;
                progressBar.style.width = `${progress}%`;
                progressPercentage.textContent = `${Math.round(progress)}%`;
            }
            
            // Update charts
            if (data.history && data.history.accuracy && data.history.accuracy.length > 0) {
                updateCharts(data.history);
            }
            
            // Update sample predictions if there's a model being trained
            if (data.log && data.log.length > 0) {
                updateSamplePredictions();
            }
            
            // Check if training is complete
            if (!data.is_training) {
                resetTrainingControls();
            }
        })
        .catch(error => {
            console.error('Error:', error);
        });
    }
    
    function initializeCharts() {
        const ctx1 = accuracyChart.getContext('2d');
        const ctx2 = lossChart.getContext('2d');
        
        charts.accuracy = new Chart(ctx1, {
            type: 'line',
            data: {
                labels: [],
                datasets: [
                    {
                        label: window.MNIST.i18n.getTranslation('train_accuracy') || 'Training Accuracy',
                        backgroundColor: 'rgba(75, 192, 192, 0.2)',
                        borderColor: 'rgba(75, 192, 192, 1)',
                        data: [],
                        fill: false,
                    },
                    {
                        label: window.MNIST.i18n.getTranslation('val_accuracy') || 'Validation Accuracy',
                        backgroundColor: 'rgba(54, 162, 235, 0.2)',
                        borderColor: 'rgba(54, 162, 235, 1)',
                        data: [],
                        fill: false,
                    }
                ]
            },
            options: {
                responsive: true,
                title: {
                    display: true,
                    text: window.MNIST.i18n.getTranslation('accuracy_chart') || 'Accuracy over Epochs'
                },
                scales: {
                    xAxes: [{
                        display: true,
                        scaleLabel: {
                            display: true,
                            labelString: window.MNIST.i18n.getTranslation('epoch') || 'Epoch'
                        }
                    }],
                    yAxes: [{
                        display: true,
                        scaleLabel: {
                            display: true,
                            labelString: window.MNIST.i18n.getTranslation('accuracy') || 'Accuracy'
                        },
                        ticks: {
                            min: 0,
                            max: 1,
                            callback: function(value) {
                                return (value * 100) + '%';
                            }
                        }
                    }]
                }
            }
        });
        
        charts.loss = new Chart(ctx2, {
            type: 'line',
            data: {
                labels: [],
                datasets: [
                    {
                        label: window.MNIST.i18n.getTranslation('train_loss') || 'Training Loss',
                        backgroundColor: 'rgba(255, 99, 132, 0.2)',
                        borderColor: 'rgba(255, 99, 132, 1)',
                        data: [],
                        fill: false,
                    },
                    {
                        label: window.MNIST.i18n.getTranslation('val_loss') || 'Validation Loss',
                        backgroundColor: 'rgba(153, 102, 255, 0.2)',
                        borderColor: 'rgba(153, 102, 255, 1)',
                        data: [],
                        fill: false,
                    }
                ]
            },
            options: {
                responsive: true,
                title: {
                    display: true,
                    text: window.MNIST.i18n.getTranslation('loss_chart') || 'Loss over Epochs'
                },
                scales: {
                    xAxes: [{
                        display: true,
                        scaleLabel: {
                            display: true,
                            labelString: window.MNIST.i18n.getTranslation('epoch') || 'Epoch'
                        }
                    }],
                    yAxes: [{
                        display: true,
                        scaleLabel: {
                            display: true,
                            labelString: window.MNIST.i18n.getTranslation('loss') || 'Loss'
                        }
                    }]
                }
            }
        });
    }
    
    function updateChartLabels() {
        if (charts.accuracy) {
            charts.accuracy.data.datasets[0].label = window.MNIST.i18n.getTranslation('train_accuracy') || 'Training Accuracy';
            charts.accuracy.data.datasets[1].label = window.MNIST.i18n.getTranslation('val_accuracy') || 'Validation Accuracy';
            charts.accuracy.options.title.text = window.MNIST.i18n.getTranslation('accuracy_chart') || 'Accuracy over Epochs';
            charts.accuracy.options.scales.xAxes[0].scaleLabel.labelString = window.MNIST.i18n.getTranslation('epoch') || 'Epoch';
            charts.accuracy.options.scales.yAxes[0].scaleLabel.labelString = window.MNIST.i18n.getTranslation('accuracy') || 'Accuracy';
            charts.accuracy.update();
        }
        
        if (charts.loss) {
            charts.loss.data.datasets[0].label = window.MNIST.i18n.getTranslation('train_loss') || 'Training Loss';
            charts.loss.data.datasets[1].label = window.MNIST.i18n.getTranslation('val_loss') || 'Validation Loss';
            charts.loss.options.title.text = window.MNIST.i18n.getTranslation('loss_chart') || 'Loss over Epochs';
            charts.loss.options.scales.xAxes[0].scaleLabel.labelString = window.MNIST.i18n.getTranslation('epoch') || 'Epoch';
            charts.loss.options.scales.yAxes[0].scaleLabel.labelString = window.MNIST.i18n.getTranslation('loss') || 'Loss';
            charts.loss.update();
        }
    }
    
    function updateCharts(history) {
        const epochs = Array.from({length: history.accuracy.length}, (_, i) => i + 1);
        
        if (charts.accuracy) {
            charts.accuracy.data.labels = epochs;
            charts.accuracy.data.datasets[0].data = history.accuracy;
            charts.accuracy.data.datasets[1].data = history.val_accuracy;
            charts.accuracy.update();
        }
        
        if (charts.loss) {
            charts.loss.data.labels = epochs;
            charts.loss.data.datasets[0].data = history.loss;
            charts.loss.data.datasets[1].data = history.val_loss;
            charts.loss.update();
        }
    }
    
    /**
     * 获取并显示模型在随机测试样本上的预测结果
     */
    function updateSamplePredictions() {
        if (!samplePredictionsContainer) return;
        
        // 获取预测样本数据
        fetch('/api/dataset_samples', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                dataset: 'test',
                count: 6,  // 显示6个样本
                with_predictions: true
            })
        })
        .then(response => response.json())
        .then(data => {
            if (data.status === 'success' && data.samples && data.samples.length > 0) {
                // 清空容器
                samplePredictionsContainer.innerHTML = '';
                
                // 添加每个样本预测
                data.samples.forEach(sample => {
                    const sampleElement = document.createElement('div');
                    sampleElement.className = 'sample-prediction-item';
                    
                    // 创建预测结果的HTML
                    const isCorrect = sample.true_label === sample.predicted_label;
                    const resultClass = isCorrect ? 'prediction-correct' : 'prediction-incorrect';
                    
                    // 确保预测标签和置信度都有定义
                    const predictedLabel = sample.predicted_label !== undefined ? sample.predicted_label : 'N/A';
                    const confidence = sample.confidence !== undefined ? (sample.confidence * 100).toFixed(1) : 'N/A';
                    
                    sampleElement.innerHTML = `
                        <div class="sample-image">
                            <img src="data:image/png;base64,${sample.image}" alt="Digit ${sample.true_label}">
                        </div>
                        <div class="sample-info">
                            <div class="true-label">True: ${sample.true_label}</div>
                            <div class="predicted-label ${resultClass}">Predicted: ${predictedLabel}</div>
                            <div class="confidence">Confidence: ${confidence}%</div>
                        </div>
                    `;
                    
                    samplePredictionsContainer.appendChild(sampleElement);
                });
            } else if (samplePredictionsContainer.innerHTML === '') {
                // 如果没有数据且容器为空，显示一条消息
                samplePredictionsContainer.innerHTML = '<p class="sample-message">' + 
                    window.MNIST.i18n.getTranslation('waiting_for_model') || 
                    'Waiting for model training to progress...' + '</p>';
            }
        })
        .catch(error => {
            console.error('Error fetching sample predictions:', error);
            if (samplePredictionsContainer.innerHTML === '') {
                samplePredictionsContainer.innerHTML = '<p class="sample-message">Error fetching predictions</p>';
            }
        });
    }
}); 
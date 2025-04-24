// Draw and Test Predictions script for MNIST Web App
document.addEventListener('DOMContentLoaded', function() {
    console.log('MNIST Web App Draw/Test Page initialized');

    // Canvas elements
    const canvas = document.getElementById('drawing-canvas');
    const clearButton = document.getElementById('clear-canvas');
    const predictButton = document.getElementById('predict-button');
    const saveButton = document.getElementById('save-image');
    const predictedDigit = document.getElementById('predicted-digit');
    const confidenceText = document.getElementById('confidence-text');
    const recentPredictionsContainer = document.getElementById('recent-predictions-container');
    const brushSizeInput = document.getElementById('brush-size');
    const modelSelector = document.getElementById('model-selector');
    
    let ctx;
    let isDrawing = false;
    let brushSize = 15; // 默认笔刷大小
    let savedModels = []; // 保存所有加载的模型
    let selectedModelId = 'latest'; // 默认选择最新模型
    
    // 设置所有文本为中文
    function setChineseText() {
        // 确保所有提示和按钮文本都是中文
        if (confidenceText && confidenceText.textContent === 'Draw and click "Predict" to see results') {
            confidenceText.textContent = '绘制并点击"预测"查看结果';
        }
        
        // 清空按钮
        if (clearButton && clearButton.textContent === 'Clear Canvas') {
            clearButton.textContent = '清空画布';
        }
        
        // 其他可能需要翻译的文本
        document.querySelectorAll('[data-i18n]').forEach(el => {
            const key = el.getAttribute('data-i18n');
            // 这里可以根据需要添加更多翻译
        });
    }
    
    // 初始化完成后调用
    function init() {
        // 初始化Canvas
        if (canvas) {
            ctx = canvas.getContext('2d');
            initializeCanvas();
        }
        
        // 加载所有可用模型
        if (modelSelector) {
            loadAvailableModels();
            
            // 监听模型选择变化
            modelSelector.addEventListener('change', function() {
                selectedModelId = this.value;
                updateModelInfo(selectedModelId);
            });
        }
        
        // 设置按钮事件监听
        setupButtonListeners();
        
        // 设置中文文本
        setChineseText();
    }
    
    // 设置按钮事件监听
    function setupButtonListeners() {
        // Clear canvas button
        if (clearButton) {
            clearButton.addEventListener('click', () => {
                // 添加点击效果
                clearButton.classList.add('animate-pulse');
                setTimeout(() => clearButton.classList.remove('animate-pulse'), 500);
                
                clearCanvas();
            });
        }
        
        // Predict button
        if (predictButton) {
            predictButton.addEventListener('click', () => {
                // 添加点击效果
                predictButton.classList.add('animate-pulse');
                setTimeout(() => predictButton.classList.remove('animate-pulse'), 500);
                
                predictDigit();
            });
        }
        
        // Save button
        if (saveButton) {
            saveButton.addEventListener('click', () => {
                // 添加点击效果
                saveButton.classList.add('animate-pulse');
                setTimeout(() => saveButton.classList.remove('animate-pulse'), 500);
                
                saveImage();
            });
        }
        
        // 画笔大小设置
        if (brushSizeInput) {
            brushSizeInput.addEventListener('input', function() {
                brushSize = this.value;
                ctx.lineWidth = brushSize;
            });
        }
    }
    
    // Initialize canvas if it exists
    if (canvas) {
        ctx = canvas.getContext('2d');
        initializeCanvas();
    }
    
    // 加载所有可用模型
    if (modelSelector) {
        loadAvailableModels();
        
        // 监听模型选择变化
        modelSelector.addEventListener('change', function() {
            selectedModelId = this.value;
            updateModelInfo(selectedModelId);
        });
    }
    
    // 加载所有可用的模型
    function loadAvailableModels() {
        console.log('开始加载模型列表...');
        fetch('/api/models')
        .then(response => {
            console.log('API响应状态码:', response.status);
            return response.json();
        })
        .then(data => {
            console.log('API返回数据:', data);
            if (data.status === 'success' && data.models) {
                console.log(`找到${data.models.length}个模型`);
                savedModels = data.models;
                
                // 先清空除了默认选项外的所有选项
                console.log('当前选择器选项数量:', modelSelector.options.length);
                while (modelSelector.options.length > 2) {
                    modelSelector.remove(2);
                }
                
                // 添加所有训练过的模型
                savedModels.forEach(model => {
                    console.log('添加模型到选择器:', model.id, model.name);
                    const option = document.createElement('option');
                    option.value = model.id;
                    option.textContent = `${model.name} (${(model.performance.accuracy * 100).toFixed(2)}%)`;
                    modelSelector.appendChild(option);
                });
                
                // 更新当前选中模型的信息
                console.log('更新模型信息:', modelSelector.value);
                updateModelInfo(modelSelector.value);
            } else {
                console.error('加载模型失败:', data.message || '未知错误');
            }
        })
        .catch(error => {
            console.error('请求模型列表错误:', error);
        });
    }
    
    // 更新所选模型的信息
    function updateModelInfo(modelId) {
        const modelType = document.getElementById('model-type');
        const modelAccuracy = document.getElementById('model-accuracy');
        const modelTrained = document.getElementById('model-trained');
        
        if (modelId === 'latest') {
            // 获取最新模型信息
            fetch('/api/models/latest')
            .then(response => response.json())
            .then(data => {
                if (data.status === 'success' && data.model) {
                    updateModelDisplay(data.model);
                }
            })
            .catch(error => {
                console.error('Error loading latest model:', error);
            });
        } else if (modelId === 'best') {
            // 获取最佳精度模型信息
            fetch('/api/models/best')
            .then(response => response.json())
            .then(data => {
                if (data.status === 'success' && data.model) {
                    updateModelDisplay(data.model);
                }
            })
            .catch(error => {
                console.error('Error loading best model:', error);
            });
        } else {
            // 查找特定模型
            const selectedModel = savedModels.find(m => m.id === modelId);
            if (selectedModel) {
                updateModelDisplay(selectedModel);
            }
        }
    }
    
    // 更新模型信息显示
    function updateModelDisplay(model) {
        console.log('更新模型显示信息:', model);
        const modelType = document.getElementById('model-type');
        const modelAccuracy = document.getElementById('model-accuracy');
        const modelTrained = document.getElementById('model-trained');
        
        if (!model) {
            console.error('无效的模型数据');
            return;
        }
        
        if (modelType) {
            // 使用i18n获取当前语言的模型类型名称
            const modelTypeName = MNIST.i18n.getTranslation('simple_neural_network') || 'Simple Neural Network';
            modelType.textContent = modelTypeName;
            if (model.params && model.params.hidden_size) {
                modelType.textContent += ` (Hidden Size: ${model.params.hidden_size})`;
            }
        }
        
        if (modelAccuracy && model.performance && model.performance.accuracy !== undefined) {
            modelAccuracy.textContent = `${(model.performance.accuracy * 100).toFixed(2)}%`;
        } else if (modelAccuracy) {
            modelAccuracy.textContent = 'Unknown';
        }
        
        if (modelTrained && model.created_at) {
            try {
                const date = new Date(model.created_at);
                modelTrained.textContent = date.toLocaleString();
            } catch (e) {
                console.error('日期解析错误:', e);
                modelTrained.textContent = model.created_at || 'Unknown';
            }
        } else if (modelTrained) {
            modelTrained.textContent = 'Unknown';
        }
    }
    
    function initializeCanvas() {
        // Set canvas background to white
        ctx.fillStyle = 'white';
        ctx.fillRect(0, 0, canvas.width, canvas.height);
        
        // Set drawing properties
        ctx.lineWidth = brushSize;
        ctx.lineCap = 'round';
        ctx.lineJoin = 'round';
        ctx.strokeStyle = 'black';
        
        // Add event listeners for drawing
        canvas.addEventListener('mousedown', startDrawing);
        canvas.addEventListener('mousemove', draw);
        canvas.addEventListener('mouseup', stopDrawing);
        canvas.addEventListener('mouseout', stopDrawing);
        
        // Touch support
        canvas.addEventListener('touchstart', startDrawingTouch);
        canvas.addEventListener('touchmove', drawTouch);
        canvas.addEventListener('touchend', stopDrawing);
        
        // 添加鼠标进入效果
        canvas.addEventListener('mouseenter', () => {
            canvas.classList.add('hover-active');
        });
        
        canvas.addEventListener('mouseleave', () => {
            canvas.classList.remove('hover-active');
        });
    }
    
    // 获取相对于canvas的精确坐标
    function getMousePos(canvas, evt) {
        const rect = canvas.getBoundingClientRect();
        const scaleX = canvas.width / rect.width;
        const scaleY = canvas.height / rect.height;
        return {
            x: (evt.clientX - rect.left) * scaleX,
            y: (evt.clientY - rect.top) * scaleY
        };
    }
    
    function startDrawing(e) {
        isDrawing = true;
        const pos = getMousePos(canvas, e);
        ctx.beginPath();
        ctx.moveTo(pos.x, pos.y);
        
        // 添加轻微的绘画视觉反馈
        canvas.style.cursor = 'crosshair';
    }
    
    function draw(e) {
        if (!isDrawing) return;
        const pos = getMousePos(canvas, e);
        ctx.lineTo(pos.x, pos.y);
        ctx.stroke();
        
        // 添加轻微的绘画动画效果
        const point = document.createElement('div');
        point.classList.add('drawing-point');
        point.style.left = e.pageX + 'px';
        point.style.top = e.pageY + 'px';
        point.style.width = brushSize/2 + 'px';
        point.style.height = brushSize/2 + 'px';
        document.body.appendChild(point);
        
        // 动画结束后移除
        setTimeout(() => {
            point.remove();
        }, 300);
    }
    
    function startDrawingTouch(e) {
        e.preventDefault();
        if (e.touches.length !== 1) return;
        
        const touch = e.touches[0];
        const rect = canvas.getBoundingClientRect();
        const scaleX = canvas.width / rect.width;
        const scaleY = canvas.height / rect.height;
        const offsetX = (touch.clientX - rect.left) * scaleX;
        const offsetY = (touch.clientY - rect.top) * scaleY;
        
        isDrawing = true;
        ctx.beginPath();
        ctx.moveTo(offsetX, offsetY);
    }
    
    function drawTouch(e) {
        e.preventDefault();
        if (!isDrawing || e.touches.length !== 1) return;
        
        const touch = e.touches[0];
        const rect = canvas.getBoundingClientRect();
        const scaleX = canvas.width / rect.width;
        const scaleY = canvas.height / rect.height;
        const offsetX = (touch.clientX - rect.left) * scaleX;
        const offsetY = (touch.clientY - rect.top) * scaleY;
        
        ctx.lineTo(offsetX, offsetY);
        ctx.stroke();
    }
    
    function stopDrawing() {
        isDrawing = false;
        canvas.style.cursor = 'default';
    }
    
    function clearCanvas() {
        // 添加清除动画
        canvas.classList.add('animate-fadeIn');
        setTimeout(() => canvas.classList.remove('animate-fadeIn'), 500);
        
        ctx.fillStyle = 'white';
        ctx.fillRect(0, 0, canvas.width, canvas.height);
        
        // 重置预测显示
        if (predictedDigit) {
            predictedDigit.textContent = '?';
            predictedDigit.classList.remove('predicted-correct', 'predicted-wrong');
        }
        
        if (confidenceText) {
            confidenceText.textContent = '绘制并点击"预测"查看结果';
        }
        
        // 重置概率条形图
        const probBars = document.querySelectorAll('.probability-bar');
        probBars.forEach(bar => {
            bar.style.width = '0%';
        });
        
        const probValues = document.querySelectorAll('.probability-value');
        probValues.forEach(value => {
            value.textContent = '0%';
        });
    }
    
    function predictDigit() {
        // 添加加载动画
        predictedDigit.innerHTML = '<div class="loading-spinner"></div>';
        
        // 获取图像数据
        const imageData = canvas.toDataURL('image/png');
        
        // 发送到后端进行实际预测，使用选定的模型
        fetch('/api/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                image: imageData,
                model_id: selectedModelId
            })
        })
        .then(response => response.json())
        .then(data => {
            if (data.status === 'success') {
                // 更新预测结果
                updatePrediction(data.prediction, data.probabilities);
                
                // 添加到最近预测
                addRecentPrediction(data.prediction, imageData);
            } else {
                // 显示错误信息
                predictedDigit.textContent = '!';
                predictedDigit.classList.add('predicted-wrong');
                confidenceText.textContent = `错误: ${data.message || '预测失败'}`;
                console.error('预测错误:', data.message);
            }
        })
        .catch(error => {
            // 处理错误
            console.error('预测请求错误:', error);
            predictedDigit.textContent = '!';
            predictedDigit.classList.add('predicted-wrong');
            confidenceText.textContent = '连接错误，请重试';
        });
    }
    
    function updatePrediction(digit, probabilities) {
        if (predictedDigit) {
            // 添加数字出现的动画
            predictedDigit.classList.add('animate-bounce');
            predictedDigit.textContent = digit;
            
            // 一段时间后移除动画类
            setTimeout(() => {
                predictedDigit.classList.remove('animate-bounce');
            }, 1000);
            
            // 根据预测置信度决定样式
            const maxProb = Math.max(...probabilities);
            if (maxProb > 0.7) {
                predictedDigit.classList.add('predicted-correct');
                predictedDigit.classList.remove('predicted-wrong');
            } else {
                predictedDigit.classList.add('predicted-wrong');
                predictedDigit.classList.remove('predicted-correct');
            }
        }
        
        if (confidenceText) {
            const maxProb = Math.max(...probabilities);
            confidenceText.textContent = `置信度: ${(maxProb * 100).toFixed(2)}%`;
        }
        
        // 更新概率条形图
        const probBars = document.querySelectorAll('.probability-bar');
        const probValues = document.querySelectorAll('.probability-value');
        
        probabilities.forEach((prob, index) => {
            if (index < probBars.length) {
                const width = (prob * 100).toFixed(2) + '%';
                
                // 添加条形图增长动画
                probBars[index].style.transition = 'width 1s ease-out';
                probBars[index].style.width = width;
                
                // 设置条形图颜色
                if (index === digit) {
                    probBars[index].style.backgroundColor = 'var(--primary-color)';
                } else {
                    probBars[index].style.backgroundColor = 'var(--secondary-color)';
                }
                
                probValues[index].textContent = (prob * 100).toFixed(2) + '%';
            }
        });
    }
    
    function saveImage() {
        const a = document.createElement('a');
        a.href = canvas.toDataURL('image/png');
        a.download = `mnist-digit-${new Date().getTime()}.png`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        
        // 添加保存成功的提示
        const notification = document.createElement('div');
        notification.classList.add('save-notification');
        notification.innerHTML = '<span>✓</span> 图像保存成功!';
        document.body.appendChild(notification);
        
        // 淡出并移除
        setTimeout(() => {
            notification.classList.add('fade-out');
            setTimeout(() => {
                notification.remove();
            }, 500);
        }, 2000);
    }
    
    function addRecentPrediction(digit, imageData) {
        if (!recentPredictionsContainer) return;
        
        // 移除空消息
        const emptyMessage = recentPredictionsContainer.querySelector('.empty-message');
        if (emptyMessage) {
            emptyMessage.remove();
        }
        
        // 创建预测项
        const predictionItem = document.createElement('div');
        predictionItem.classList.add('recent-prediction-item', 'animate-fadeIn');
        
        predictionItem.innerHTML = `
            <div class="prediction-image">
                <img src="${imageData}" alt="Drawn digit ${digit}">
            </div>
            <div class="prediction-info">
                <div class="prediction-digit">${digit}</div>
                <div class="prediction-time">${new Date().toLocaleTimeString()}</div>
            </div>
        `;
        
        // 添加到容器的开始位置
        if (recentPredictionsContainer.firstChild) {
            recentPredictionsContainer.insertBefore(predictionItem, recentPredictionsContainer.firstChild);
        } else {
            recentPredictionsContainer.appendChild(predictionItem);
        }
        
        // 限制显示10个最近的预测
        const items = recentPredictionsContainer.querySelectorAll('.recent-prediction-item');
        if (items.length > 10) {
            items[items.length - 1].remove();
        }
    }
    
    // 最后调用init函数初始化页面
    init();
}); 
/**
 * models.js - Handles model comparison and visualization
 * Part of the MNIST Web Application
 */

// 检查Chart.js加载情况
if (typeof Chart === 'undefined') {
    console.error('Chart.js库未加载，某些功能可能不可用');
}

// 注册Matrix图表类型（如果Chart.js存在）
if (typeof Chart !== 'undefined') {
    // 简单的matrix图表控制器
    Chart.controllers.matrix = Chart.controllers.scatter.extend({
        draw: function() {
            var meta = this.getMeta();
            var dataset = this.getDataset();
            var data = meta.data || [];
            
            // 确保我们有正确的维度
            if (!dataset.width || !dataset.height) {
                console.warn('Matrix图表需要width和height属性');
                return;
            }
            
            var cellWidth = this.chart.width / dataset.width;
            var cellHeight = this.chart.height / dataset.height;
            
            // 绘制每个单元格
            for (var i = 0; i < data.length; i++) {
                var x = i % dataset.width;
                var y = Math.floor(i / dataset.width);
                
                var value = dataset.data[i] || 0;
                var color = dataset.backgroundColor;
                
                // 如果backgroundColor是函数，调用它
                if (typeof color === 'function') {
                    color = color({
                        dataIndex: i,
                        dataset: dataset,
                        datasetIndex: meta.index,
                        value: value
                    });
                }
                
                // 绘制矩形
                var ctx = this.chart.ctx;
                ctx.save();
                ctx.fillStyle = color;
                ctx.fillRect(x * cellWidth, y * cellHeight, cellWidth, cellHeight);
                
                // 对角线特殊显示
                if (x === y) {
                    ctx.strokeStyle = 'rgba(0,0,0,0.3)';
                    ctx.strokeRect(x * cellWidth, y * cellHeight, cellWidth, cellHeight);
                }
                
                // 添加文本
                if (value > 0) {
                    ctx.fillStyle = '#fff';
                    ctx.font = '10px Arial';
                    ctx.textAlign = 'center';
                    ctx.textBaseline = 'middle';
                    ctx.fillText(value, (x + 0.5) * cellWidth, (y + 0.5) * cellHeight);
                }
                
                ctx.restore();
            }
        }
    });
    
    console.log('已注册Matrix图表类型');
} else {
    console.warn('无法注册Matrix图表类型，Chart.js未加载');
}

// Initialize MNIST namespace if it doesn't exist
if (typeof MNIST === 'undefined') {
    MNIST = {};
}

// Models module
MNIST.models = (function() {
    // Private variables
    let savedModels = [];
    let selectedModels = [];
    let accuracyChart = null;
    let lossChart = null;
    let sortConfig = {
        field: 'accuracy',
        direction: 'desc'
    };
    let searchQuery = '';
    let trainingProgressChart = null;
    let confusionMatrixChart = null;
    
    // DOM elements
    const modelsListEl = document.querySelector('.models-list');
    const emptyMessageEl = document.querySelector('.empty-message');
    const accuracyChartEl = document.getElementById('accuracy-chart');
    const lossChartEl = document.getElementById('loss-chart');
    const searchInput = document.getElementById('model-search');
    const sortSelect = document.getElementById('sort-models');
    const sortDirectionBtn = document.getElementById('sort-direction');
    const trainingProgressChartEl = document.getElementById('training-progress-chart');
    const confusionMatrixChartEl = document.getElementById('confusion-matrix-chart');
    const confusionMatrixModelSelect = document.getElementById('confusion-matrix-model');

    // Initialize charts
    const initCharts = () => {
        if (accuracyChartEl && lossChartEl) {
            accuracyChart = new Chart(accuracyChartEl, {
                type: 'bar',
                data: {
                    labels: [],
                    datasets: [{
                        label: 'Accuracy',
                        data: [],
                        backgroundColor: 'rgba(54, 162, 235, 0.7)',
                        borderColor: 'rgba(54, 162, 235, 1)',
                        borderWidth: 1
                    }]
                },
                options: {
                    responsive: true,
                    scales: {
                        y: {
                            beginAtZero: true,
                            max: 1,
                            title: {
                                display: true,
                                text: 'Accuracy'
                            }
                        }
                    },
                    plugins: {
                        title: {
                            display: true,
                            text: 'Model Accuracy Comparison'
                        }
                    }
                }
            });
            
            lossChart = new Chart(lossChartEl, {
                type: 'bar',
                data: {
                    labels: [],
                    datasets: [{
                        label: 'Loss',
                        data: [],
                        backgroundColor: 'rgba(255, 99, 132, 0.7)',
                        borderColor: 'rgba(255, 99, 132, 1)',
                        borderWidth: 1
                    }]
                },
                options: {
                    responsive: true,
                    scales: {
                        y: {
                            beginAtZero: true,
                            title: {
                                display: true,
                                text: 'Loss'
                            }
                        }
                    },
                    plugins: {
                        title: {
                            display: true,
                            text: 'Model Loss Comparison'
                        }
                    }
                }
            });
        }

        if (trainingProgressChartEl) {
            trainingProgressChart = new Chart(trainingProgressChartEl, {
                type: 'line',
                data: {
                    labels: [],
                    datasets: []
                },
                options: {
                    responsive: true,
                    interaction: {
                        mode: 'index',
                        intersect: false,
                    },
                    scales: {
                        x: {
                            title: {
                                display: true,
                                text: 'Epoch'
                            }
                        },
                        y: {
                            beginAtZero: true,
                            title: {
                                display: true,
                                text: 'Value'
                            }
                        }
                    },
                    plugins: {
                        title: {
                            display: true,
                            text: 'Training Progress'
                        },
                        tooltip: {
                            enabled: true,
                            mode: 'index',
                            intersect: false,
                            callbacks: {
                                title: (items) => `Epoch ${items[0].label}`,
                                label: (item) => {
                                    const metric = item.dataset.label;
                                    const value = item.raw;
                                    return `${metric}: ${value.toFixed(4)}`;
                                }
                            }
                        }
                    }
                }
            });
        }

        if (confusionMatrixChartEl) {
            confusionMatrixChart = new Chart(confusionMatrixChartEl, {
                type: 'matrix',
                data: {
                    labels: ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'],
                    datasets: [{
                        data: Array(100).fill(0),
                        width: 10,
                        height: 10,
                        backgroundColor: (context) => {
                            const value = context.dataset.data[context.dataIndex];
                            const alpha = Math.min(value / 100, 1);
                            return `rgba(54, 162, 235, ${alpha})`;
                        }
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                        title: {
                            display: true,
                            text: 'Confusion Matrix'
                        },
                        tooltip: {
                            callbacks: {
                                title: (items) => {
                                    const item = items[0];
                                    const x = Math.floor(item.dataIndex / 10);
                                    const y = item.dataIndex % 10;
                                    return `Predicted: ${x}, Actual: ${y}`;
                                },
                                label: (item) => {
                                    return `Count: ${item.raw}`;
                                }
                            }
                        }
                    },
                    scales: {
                        x: {
                            title: {
                                display: true,
                                text: 'Predicted Class'
                            }
                        },
                        y: {
                            title: {
                                display: true,
                                text: 'Actual Class'
                            }
                        }
                    }
                }
            });
        }
    };

    // Fetch saved models from the server
    const fetchSavedModels = async () => {
        try {
            console.log("正在从服务器获取模型列表...", new Date().toISOString());
            
            // 获取当前URL，构建完整的API路径
            const baseUrl = window.location.origin;
            const apiUrl = `${baseUrl}/api/models`;
            console.log("使用API URL:", apiUrl);
            
            const response = await fetch(apiUrl);
            console.log("API响应状态:", response.status, response.statusText);
            
            if (!response.ok) {
                throw new Error('获取模型列表失败');
            }
            
            const data = await response.json();
            console.log("API返回数据:", data);
            
            // 保存响应到全局变量以便调试
            window.DEBUG_MODEL_RESPONSE = data;
            
            if (data.status === 'success' && data.models) {
                savedModels = [];
                
                // 记录接收到的数据
                console.log('收到的模型数据:', data.models);
                console.log('模型对象类型:', typeof data.models);
                console.log('模型对象键:', Object.keys(data.models));
                
                // Convert object to array and add IDs
                Object.keys(data.models).forEach(modelId => {
                    const model = data.models[modelId];
                    model.id = modelId;
                    
                    // 确保性能属性存在
                    if (!model.performance) {
                        model.performance = {};
                    }
                    
                    // 使用性能对象中的准确度或设置默认值
                    model.accuracy = model.performance.accuracy || 0;
                    
                    // 添加其他必要字段
                    if (!model.hyperparams && model.params) {
                        model.hyperparams = model.params;
                    }
                    
                    // 确保timestamp字段存在
                    if (!model.timestamp && model.created_at) {
                        model.timestamp = model.created_at;
                    }
                    
                    savedModels.push(model);
                    console.log(`处理了模型: ${modelId}, 准确率: ${model.accuracy}`);
                });
                
                // Sort models by accuracy (descending)
                savedModels.sort((a, b) => b.accuracy - a.accuracy);
                
                // 如果找到了模型，隐藏空消息
                if (savedModels.length > 0) {
                    console.log(`找到 ${savedModels.length} 个模型，将隐藏空消息`);
                    if (emptyMessageEl) {
                        emptyMessageEl.style.display = 'none';
                        console.log('已隐藏空消息元素');
                    } else {
                        console.warn('未找到空消息元素!');
                    }
                } else {
                    console.log("没有找到可用模型，显示空消息");
                    if (emptyMessageEl) {
                        emptyMessageEl.style.display = 'block';
                        console.log('已显示空消息元素');
                    } else {
                        console.warn('未找到空消息元素!');
                    }
                }
            } else {
                console.warn('返回数据格式无效或无模型', data);
                savedModels = [];
                
                if (emptyMessageEl) {
                    emptyMessageEl.style.display = 'block';
                    console.log('显示空消息元素，因为返回数据无效');
                } else {
                    console.warn('未找到空消息元素!');
                }
            }
            
            // 更新模型列表视图
            console.log('准备渲染模型列表...');
            renderModelsList();
            // 更新混淆矩阵选择器
            updateConfusionMatrixSelector();
            
        } catch (error) {
            console.error('获取模型时发生错误:', error);
            // Show error message
            if (modelsListEl) {
                modelsListEl.innerHTML = `
                    <div class="error-message">
                        <p>加载模型失败，请稍后重试。</p>
                        <button id="retry-fetch" class="btn">重试</button>
                    </div>
                `;
            
                // Add retry button event listener
                const retryBtn = document.getElementById('retry-fetch');
                if (retryBtn) {
                    retryBtn.addEventListener('click', fetchSavedModels);
                }
            } else {
                console.error('未找到模型列表元素，无法显示错误消息!');
            }
        }
    };

    // Search and filter functionality
    const setupSearchAndSort = () => {
        if (searchInput) {
            searchInput.addEventListener('input', (e) => {
                searchQuery = e.target.value.toLowerCase();
                renderModelsList();
            });
        }

        if (sortSelect) {
            sortSelect.addEventListener('change', (e) => {
                sortConfig.field = e.target.value;
                renderModelsList();
            });
        }

        if (sortDirectionBtn) {
            sortDirectionBtn.addEventListener('click', () => {
                sortConfig.direction = sortConfig.direction === 'asc' ? 'desc' : 'asc';
                const icon = sortDirectionBtn.querySelector('i');
                icon.className = `fas fa-sort-amount-${sortConfig.direction}`;
                renderModelsList();
            });
        }
    };

    // Filter and sort models
    const getFilteredAndSortedModels = () => {
        let filtered = savedModels;
        
        // Apply search filter
        if (searchQuery) {
            filtered = filtered.filter(model => {
                const searchFields = [
                    model.name || '',
                    model.description || '',
                    model.id || '',
                    `${model.accuracy || 0}`,
                    model.params?.activation || '',
                    model.params?.hidden_size || ''
                ];
                return searchFields.some(field => 
                    String(field).toLowerCase().includes(searchQuery)
                );
            });
        }
        
        console.log(`过滤后有 ${filtered.length} 个模型，排序字段: ${sortConfig.field}, 方向: ${sortConfig.direction}`);
        
        // Apply sorting
        filtered.sort((a, b) => {
            let aValue, bValue;
            
            switch (sortConfig.field) {
                case 'accuracy':
                    aValue = a.accuracy || (a.performance && a.performance.accuracy) || 0;
                    bValue = b.accuracy || (b.performance && b.performance.accuracy) || 0;
                    break;
                case 'loss':
                    aValue = a.loss || 0;
                    bValue = b.loss || 0;
                    break;
                case 'date':
                    aValue = new Date(a.created_at || a.timestamp || 0).getTime();
                    bValue = new Date(b.created_at || b.timestamp || 0).getTime();
                    break;
                case 'name':
                    aValue = (a.name || a.id || '').toLowerCase();
                    bValue = (b.name || b.id || '').toLowerCase();
                    break;
                default:
                    aValue = a[sortConfig.field] || 0;
                    bValue = b[sortConfig.field] || 0;
            }
            
            // 确保值有效
            if (typeof aValue !== 'number') aValue = 0;
            if (typeof bValue !== 'number') bValue = 0;
            
            if (sortConfig.direction === 'asc') {
                return aValue > bValue ? 1 : -1;
            } else {
                return aValue < bValue ? 1 : -1;
            }
        });
        
        return filtered;
    };

    // Render the list of saved models
    const renderModelsList = () => {
        if (!modelsListEl) return;
        
        console.log("开始渲染模型列表...");
        console.log("当前保存的模型数量:", savedModels.length);
        console.log("当前选中的模型数量:", selectedModels.length);
        
        // Clear existing list
        modelsListEl.innerHTML = '';
        
        // Filter and sort models
        const filteredModels = getFilteredAndSortedModels();
        console.log("过滤排序后的模型数量:", filteredModels.length);
        
        // 直接检查并处理空消息元素
        // 无论如何，确保我们能找到并正确操作空消息元素
        const emptyMsg = document.querySelector('.empty-message');
        console.log("渲染时找到空消息元素:", emptyMsg !== null);
        
        if (emptyMsg) {
            if (savedModels.length > 0) {
                // 如果有模型，隐藏空消息
                emptyMsg.style.display = 'none';
                console.log("有模型，隐藏空消息");
            } else {
                // 如果没有模型，显示空消息
                emptyMsg.style.display = 'block';
                console.log("没有模型，显示空消息");
            }
        }
        
        if (filteredModels.length === 0) {
                if (searchQuery) {
                modelsListEl.innerHTML = `<p class="no-results">No models found matching "${searchQuery}"</p>`;
                } else {
                if (savedModels.length === 0) {
                    // This is handled by the empty message element
                    console.log("没有保存的模型，显示空消息");
                } else {
                    modelsListEl.innerHTML = `<p class="no-results">No models match the current filters</p>`;
                }
            }
            return;
        }
        
        // Create model items
        filteredModels.forEach(model => {
            const modelItem = document.createElement('div');
            modelItem.className = `model-item ${selectedModels.includes(model.id) ? 'selected' : ''}`;
            modelItem.dataset.id = model.id;
            
            // 获取准确率，确保即使未定义也不会出错
            const accuracy = model.accuracy || (model.performance && model.performance.accuracy) || 0;
            const accuracyFormatted = (accuracy * 100).toFixed(2);
            
            // 获取模型名称，后备为ID
            const modelName = model.name || `Model ${model.id}`;
            
            modelItem.innerHTML = `
                <div class="model-header">
                    <input type="checkbox" class="model-checkbox" 
                           ${selectedModels.includes(model.id) ? 'checked' : ''}>
                    <h4 class="model-name">${modelName}</h4>
                    </div>
                <div class="model-details">
                    <div class="model-metrics">
                        <div class="metric">
                            <span class="metric-label">Accuracy:</span>
                            <span class="metric-value">${accuracyFormatted}%</span>
                            </div>
                        <div class="metric">
                            <span class="metric-label">Hidden Size:</span>
                            <span class="metric-value">${model.params?.hidden_size || 'N/A'}</span>
                            </div>
                        <div class="metric">
                            <span class="metric-label">Activation:</span>
                            <span class="metric-value">${model.params?.activation || 'N/A'}</span>
                        </div>
                    </div>
                    <div class="model-actions">
                        <button class="btn btn-sm view-details-btn" data-id="${model.id}">
                            <i class="fas fa-info-circle"></i> Details
                        </button>
                    </div>
                </div>
            `;
        
            // Add click events
            modelItem.querySelector('.model-checkbox').addEventListener('change', (e) => {
                const checked = e.target.checked;
                if (checked) {
                    modelItem.classList.add('selected');
                } else {
                    modelItem.classList.remove('selected');
                }
                toggleModelSelection(model.id);
                e.stopPropagation();
            });
            
            modelItem.querySelector('.view-details-btn').addEventListener('click', (e) => {
                showModelDetails(model.id);
                e.stopPropagation();
            });
            
            // Click anywhere on item to select it
            modelItem.addEventListener('click', (e) => {
                if (e.target.tagName !== 'BUTTON' && e.target.tagName !== 'INPUT' && e.target.tagName !== 'I') {
                    const checkbox = modelItem.querySelector('.model-checkbox');
                    checkbox.checked = !checkbox.checked;
                    const event = new Event('change');
                    checkbox.dispatchEvent(event);
                }
            });
            
            modelsListEl.appendChild(modelItem);
            });
        
        // Update charts based on selected models
        updateCharts();
    };

    // Toggle model selection for comparison
    const toggleModelSelection = (modelId) => {
        console.log(`切换模型选择状态: ${modelId}`);
        
        const modelIndex = savedModels.findIndex(m => m.id === modelId);
        if (modelIndex === -1) {
            console.warn(`找不到ID为 ${modelId} 的模型`);
            return;
        }
        
        const model = savedModels[modelIndex];
        const selectedIndex = selectedModels.indexOf(modelId);
        const isSelected = selectedIndex !== -1;
        
        console.log(`当前模型 ${modelId} 是否已选择: ${isSelected}`);
        
        if (isSelected) {
            // 从已选择列表中移除
            selectedModels.splice(selectedIndex, 1);
            console.log(`已从选择列表中移除模型 ${modelId}`);
        } else {
            // 添加到已选择列表
            selectedModels.push(modelId);
            console.log(`已将模型 ${modelId} 添加到选择列表`);
        }
        
        renderModelsList();
        updateCharts();
    };

    // Helper function to format time in seconds to human-readable format
    const formatTime = (seconds) => {
        if (seconds < 60) return `${seconds}s`;
        if (seconds < 3600) return `${Math.floor(seconds / 60)}m ${seconds % 60}s`;
        return `${Math.floor(seconds / 3600)}h ${Math.floor((seconds % 3600) / 60)}m`;
    };

    // Render confusion matrix
    const renderConfusionMatrix = (matrix) => {
        if (!matrix || !Array.isArray(matrix) || matrix.length !== 10) {
            return '<p>Confusion matrix data unavailable</p>';
        }
        
        // Calculate totals for color scaling
        let maxValue = 0;
        for (let i = 0; i < 10; i++) {
            for (let j = 0; j < 10; j++) {
                maxValue = Math.max(maxValue, matrix[i][j]);
            }
        }
        
        // Generate HTML for confusion matrix
        let html = '<table class="confusion-matrix-table">';
        
        // Header row
        html += '<tr><th></th>';
        for (let i = 0; i < 10; i++) {
            html += `<th>Pred ${i}</th>`;
        }
        html += '<th>Total</th></tr>';
        
        // Data rows
        let correctTotal = 0;
        let total = 0;
        
        for (let i = 0; i < 10; i++) {
            html += `<tr><th>True ${i}</th>`;
            
            let rowSum = 0;
            for (let j = 0; j < 10; j++) {
                const value = matrix[i][j];
                rowSum += value;
                total += value;
                
                if (i === j) {
                    correctTotal += value;
                }
                
                // Calculate color intensity based on value
                const intensity = Math.min(0.9, value / maxValue * 0.9);
                let backgroundColor;
                
                if (i === j) {
                    // Correct predictions - green
                    backgroundColor = `rgba(0, 200, 0, ${intensity})`;
                } else {
                    // Incorrect predictions - red
                    backgroundColor = `rgba(255, 0, 0, ${intensity})`;
                }
                
                html += `<td style="background-color: ${backgroundColor}">${value}</td>`;
            }
            
            html += `<td>${rowSum}</td></tr>`;
        }
        
        // Footer row with column sums
        html += '<tr><th>Total</th>';
        for (let j = 0; j < 10; j++) {
            let colSum = 0;
            for (let i = 0; i < 10; i++) {
                colSum += matrix[i][j];
            }
            html += `<td>${colSum}</td>`;
        }
        html += `<td>${total}</td></tr>`;
        
        // Overall accuracy
        const accuracy = correctTotal / total;
        html += `<tr><td colspan="12" class="accuracy-row">Overall Accuracy: ${(accuracy * 100).toFixed(2)}%</td></tr>`;
        
        html += '</table>';
        
        return html;
    };

    // Update charts with selected models data
    const updateCharts = () => {
        if (!accuracyChart || !lossChart) return;
        
        console.log("更新图表，已选择模型数量:", selectedModels.length);
        
        const labels = selectedModels.map(model => model.name || `Model ${model.id.substring(0, 8)}`);
        const accuracyData = selectedModels.map(model => {
            // 返回模型准确率，确保是有效数值
            const accuracy = model.accuracy || (model.performance && model.performance.accuracy) || 0;
            console.log(`模型 ${model.id} 准确率: ${accuracy}`);
            return accuracy;
        });
        
        const lossData = selectedModels.map(model => {
            // 获取损失值，如果不存在返回0
            return model.loss || 0;
        });
        
        accuracyChart.data.labels = labels;
        accuracyChart.data.datasets[0].data = accuracyData;
        accuracyChart.update();
        
        lossChart.data.labels = labels;
        lossChart.data.datasets[0].data = lossData;
        lossChart.update();
        
        // 如果选择了单个模型，更新训练进度图表
        if (selectedModels.length === 1) {
            updateTrainingProgressChart(selectedModels[0]);
        }
    };

    // Update training progress chart
    const updateTrainingProgressChart = (selectedModel) => {
        if (!trainingProgressChart || !selectedModel) {
            console.log("没有进度图表或未选择模型");
            return;
        }

        console.log(`更新 ${selectedModel.id} 的训练进度图表`);
        
        // 检查模型是否有训练历史
        if (!selectedModel.training_history && !selectedModel.history) {
            console.log(`模型 ${selectedModel.id} 没有训练历史数据`);
            return;
        }
        
        // 确定历史数据源
        const history = selectedModel.training_history || selectedModel.history || {};
        
        // 检查历史数据中是否有准确率
        if (!history.accuracy || !Array.isArray(history.accuracy) || history.accuracy.length === 0) {
            console.log(`模型 ${selectedModel.id} 的历史数据缺失或格式不正确:`, history);
            return;
        }

        console.log(`找到历史数据，epochs: ${history.accuracy.length}`);
        const epochs = Array.from({length: history.accuracy.length}, (_, i) => i + 1);
        
        // 检查复选框元素是否存在
        const checkboxes = {
            accuracy: document.querySelector('input[value="accuracy"]'),
            loss: document.querySelector('input[value="loss"]'),
            val_accuracy: document.querySelector('input[value="val_accuracy"]'),
            val_loss: document.querySelector('input[value="val_loss"]')
        };
        
        const datasets = [];
        
        // 添加准确率数据集
        if (history.accuracy && history.accuracy.length > 0) {
            datasets.push({
                label: '训练准确率',
                data: history.accuracy,
                borderColor: 'rgb(54, 162, 235)',
                backgroundColor: 'rgba(54, 162, 235, 0.1)',
                hidden: checkboxes.accuracy ? !checkboxes.accuracy.checked : false
            });
        }
        
        // 添加损失数据集
        if (history.loss && history.loss.length > 0) {
            datasets.push({
                label: '训练损失',
                data: history.loss,
                borderColor: 'rgb(255, 99, 132)',
                backgroundColor: 'rgba(255, 99, 132, 0.1)',
                hidden: checkboxes.loss ? !checkboxes.loss.checked : false
            });
        }
        
        // 添加验证准确率数据集
        if (history.val_accuracy && history.val_accuracy.length > 0) {
            datasets.push({
                label: '验证准确率',
                data: history.val_accuracy,
                borderColor: 'rgb(75, 192, 192)',
                backgroundColor: 'rgba(75, 192, 192, 0.1)',
                hidden: checkboxes.val_accuracy ? !checkboxes.val_accuracy.checked : false
            });
        }
        
        // 添加验证损失数据集
        if (history.val_loss && history.val_loss.length > 0) {
            datasets.push({
                label: '验证损失',
                data: history.val_loss,
                borderColor: 'rgb(255, 159, 64)',
                backgroundColor: 'rgba(255, 159, 64, 0.1)',
                hidden: checkboxes.val_loss ? !checkboxes.val_loss.checked : false
            });
            }

        trainingProgressChart.data.labels = epochs;
        trainingProgressChart.data.datasets = datasets;
        trainingProgressChart.update();
        
        console.log("训练进度图表已更新");
    };

    // Update confusion matrix chart
    const updateConfusionMatrix = async (modelId) => {
        if (!confusionMatrixChart || !modelId) {
            return;
        }

        try {
            const response = await fetch(`/api/visualize/confusion_matrix?model_id=${modelId}`);
            if (!response.ok) {
                throw new Error('Failed to fetch confusion matrix');
            }

            const data = await response.json();
            if (data.status === 'success' && data.matrix) {
                const flatMatrix = data.matrix.flat();
                confusionMatrixChart.data.datasets[0].data = flatMatrix;
                confusionMatrixChart.update();
            }
        } catch (error) {
            console.error('Error fetching confusion matrix:', error);
        }
    };

    // Setup metric toggles
    const setupMetricToggles = () => {
        document.querySelectorAll('.metric-toggles input[type="checkbox"]').forEach(checkbox => {
            checkbox.addEventListener('change', () => {
                const metric = checkbox.value;
                const dataset = trainingProgressChart.data.datasets.find(d => d.label.toLowerCase().includes(metric));
                if (dataset) {
                    dataset.hidden = !checkbox.checked;
                    trainingProgressChart.update();
                }
            });
        });
    };

    // Setup confusion matrix model selector
    const setupConfusionMatrixSelector = () => {
        if (confusionMatrixModelSelect) {
            confusionMatrixModelSelect.addEventListener('change', (e) => {
                const modelId = e.target.value;
                updateConfusionMatrix(modelId);
            });
        }
    };

    // Update confusion matrix model selector options
    const updateConfusionMatrixSelector = () => {
        if (!confusionMatrixModelSelect) {
            return;
        }

        const options = savedModels.map(model => `
            <option value="${model.id}">
                ${model.name || 'Model ' + model.id.substring(0, 8)} (${(model.accuracy * 100).toFixed(2)}%)
            </option>
        `);

        confusionMatrixModelSelect.innerHTML = '<option value="">Select a model</option>' + options.join('');
    };

    // Initialize
    const init = () => {
        console.log("模型模块初始化...");
        
        // 首先获取DOM元素
        const emptyMsg = document.querySelector('.empty-message');
        if (emptyMsg) {
            console.log("找到空消息元素，当前显示状态:", window.getComputedStyle(emptyMsg).display);
        }
        
        initCharts();
        setupSearchAndSort();
        setupMetricToggles();
        setupConfusionMatrixSelector();
        
        // 获取模型
        console.log("开始获取模型数据...");
        fetchSavedModels();
        
        // Refresh button
        document.getElementById('refresh-models')?.addEventListener('click', fetchSavedModels);
    };

    // Update all visualizations when a model is selected
    const onModelSelected = (model) => {
        updateTrainingProgressChart(model);
        if (confusionMatrixModelSelect.value === '') {
            confusionMatrixModelSelect.value = model.id;
            updateConfusionMatrix(model.id);
        }
    };

    // Public API
    return {
        init: init,
        getSelectedModels: () => selectedModels,
        refresh: fetchSavedModels
    };
})();

// Initialize models module when DOM is ready
document.addEventListener('DOMContentLoaded', function() {
    console.log("%c MNIST Web App - Models Module %c DOM Content Loaded ", 
                "background:#3498db; color:white; font-size:12px; padding:3px;", 
                "background:#2ecc71; color:white; font-size:12px; padding:3px;");
    
    // 创建一个全局变量以方便检查
    window.DEBUG_MODEL_RESPONSE = null;

    // 检查DOM元素
    const modelsListEl = document.querySelector('.models-list');
    const emptyMessageEl = document.querySelector('.empty-message');
    
    console.log("模型列表元素存在:", modelsListEl !== null);
    console.log("空消息元素存在:", emptyMessageEl !== null);
    
    if (modelsListEl) {
        console.log("模型列表元素:", modelsListEl);
    }
    
    if (emptyMessageEl) {
        console.log("空消息元素:", emptyMessageEl);
        console.log("空消息元素当前显示状态:", window.getComputedStyle(emptyMessageEl).display);
    }
    
    // 检查MNIST命名空间
    console.log("MNIST命名空间:", typeof MNIST !== 'undefined');
    console.log("MNIST.models命名空间:", typeof MNIST !== 'undefined' && typeof MNIST.models !== 'undefined');
    
    // 调用初始化函数
    if (typeof MNIST !== 'undefined' && typeof MNIST.models !== 'undefined' && typeof MNIST.models.init === 'function') {
        console.log("调用模型模块初始化函数...");
        try {
            MNIST.models.init();
            console.log("%c模型模块初始化成功%c", "background:#2ecc71; color:white; padding:2px;", "");
        } catch (e) {
            console.error("模型模块初始化失败:", e);
        }
    } else {
        console.error("找不到MNIST.models.init函数!");
    }
});

// Model Details Modal Functions
window.showModelDetails = function(modelId) {
    const loadingOverlay = document.createElement('div');
    loadingOverlay.className = 'loading-overlay';
    loadingOverlay.innerHTML = '<div class="loading-spinner"></div>';
    document.body.appendChild(loadingOverlay);

    fetch(`/api/models/${modelId}`)
        .then(response => response.json())
        .then(model => {
            document.getElementById('modelDetailsTitle').textContent = model.name;
            
            // Fill basic information
            const infoGrid = document.getElementById('modelInfo');
            infoGrid.innerHTML = `
                <div class="info-item">
                    <span class="info-label">Architecture</span>
                    <span class="info-value">${model.architecture}</span>
                </div>
                <div class="info-item">
                    <span class="info-label">Created</span>
                    <span class="info-value">${new Date(model.created_at).toLocaleString()}</span>
                </div>
                <div class="info-item">
                    <span class="info-label">Training Time</span>
                    <span class="info-value">${model.training_time}</span>
                </div>
                <div class="info-item">
                    <span class="info-label">Best Accuracy</span>
                    <span class="info-value">${(model.best_accuracy * 100).toFixed(2)}%</span>
                </div>
            `;

            // Fill hyperparameters
            const hyperparamsGrid = document.getElementById('hyperparameters');
            hyperparamsGrid.innerHTML = Object.entries(model.hyperparameters)
                .map(([key, value]) => `
                    <div class="info-item">
                        <span class="info-label">${key}</span>
                        <span class="info-value">${value}</span>
                    </div>
                `).join('');

            // Initialize metrics charts
            initializeMetricsCharts(modelId);

            // Fill version history
            updateVersionHistory(modelId);

            document.getElementById('modelDetailsModal').classList.add('show');
            document.getElementById('modelDetailsModal').style.display = 'block';
            document.body.classList.add('modal-open');
            
            loadingOverlay.remove();
        })
        .catch(error => {
            console.error('Error loading model details:', error);
            loadingOverlay.remove();
            showErrorMessage('Failed to load model details');
        });
}

function initializeMetricsCharts(modelId) {
    const accuracyChart = new Chart(document.getElementById('accuracyChart'), {
        type: 'line',
        data: {
            labels: [],
            datasets: [
                {
                    label: 'Training Accuracy',
                    borderColor: '#28a745',
                    data: []
                },
                {
                    label: 'Validation Accuracy',
                    borderColor: '#17a2b8',
                    data: []
                }
            ]
        },
        options: {
            responsive: true,
            scales: {
                y: {
                    beginAtZero: true,
                    max: 1
                }
            }
        }
    });

    const lossChart = new Chart(document.getElementById('lossChart'), {
        type: 'line',
        data: {
            labels: [],
            datasets: [
                {
                    label: 'Training Loss',
                    borderColor: '#dc3545',
                    data: []
                },
                {
                    label: 'Validation Loss',
                    borderColor: '#ffc107',
                    data: []
                }
            ]
        },
        options: {
            responsive: true,
            scales: {
                y: {
                    beginAtZero: true
                }
            }
        }
    });

    updateMetricsCharts(modelId, accuracyChart, lossChart);
}

function updateMetricsCharts(modelId, accuracyChart, lossChart) {
    fetch(`/api/models/${modelId}/metrics`)
        .then(response => response.json())
        .then(data => {
            accuracyChart.data.labels = data.epochs;
            accuracyChart.data.datasets[0].data = data.train_accuracy;
            accuracyChart.data.datasets[1].data = data.val_accuracy;
            accuracyChart.update();

            lossChart.data.labels = data.epochs;
            lossChart.data.datasets[0].data = data.train_loss;
            lossChart.data.datasets[1].data = data.val_loss;
            lossChart.update();
        })
        .catch(error => {
            console.error('Error loading metrics:', error);
            showErrorMessage('Failed to load metrics data');
        });
}

function updateVersionHistory(modelId) {
    fetch(`/api/models/${modelId}/versions`)
        .then(response => response.json())
        .then(versions => {
            const versionsList = document.getElementById('versionsList');
            versionsList.innerHTML = versions.map(version => `
                <div class="version-item">
                    <div class="version-info">
                        <strong>Version ${version.version}</strong>
                        <span class="version-meta">
                            Created on ${new Date(version.created_at).toLocaleString()} 
                            by ${version.created_by}
                        </span>
                    </div>
                    <div class="version-actions">
                        <button class="btn btn-sm btn-outline-primary" 
                                onclick="restoreVersion('${modelId}', ${version.version})">
                            Restore
                        </button>
                        <button class="btn btn-sm btn-outline-danger"
                                onclick="deleteVersion('${modelId}', ${version.version})">
                            Delete
                        </button>
                    </div>
                </div>
            `).join('');
        })
        .catch(error => {
            console.error('Error loading version history:', error);
            showErrorMessage('Failed to load version history');
        });
}

window.showExportModal = function(modelId) {
    document.getElementById('exportModelId').value = modelId;
    document.getElementById('exportModal').classList.add('show');
    document.getElementById('exportModal').style.display = 'block';
    document.body.classList.add('modal-open');
}

window.exportModel = function() {
    const modelId = document.getElementById('exportModelId').value;
    const format = document.querySelector('input[name="exportFormat"]:checked').value;
    const includeHistory = document.getElementById('includeHistory').checked;
    const includeMetadata = document.getElementById('includeMetadata').checked;

    const loadingOverlay = document.createElement('div');
    loadingOverlay.className = 'loading-overlay';
    loadingOverlay.innerHTML = '<div class="loading-spinner"></div>';
    document.body.appendChild(loadingOverlay);

    fetch(`/api/models/${modelId}/export`, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({
            format,
            include_history: includeHistory,
            include_metadata: includeMetadata
        })
    })
    .then(response => {
        if (!response.ok) {
            throw new Error('Export failed');
        }
        return response.blob();
    })
    .then(blob => {
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `model_${modelId}_export.${format.toLowerCase()}`;
        document.body.appendChild(a);
        a.click();
        window.URL.revokeObjectURL(url);
        a.remove();
        hideExportModal();
        showSuccessMessage('Model exported successfully');
    })
    .catch(error => {
        console.error('Error exporting model:', error);
        showErrorMessage('Failed to export model');
    })
    .finally(() => {
        loadingOverlay.remove();
    });
}

window.hideModelDetails = function() {
    document.getElementById('modelDetailsModal').classList.remove('show');
    document.getElementById('modelDetailsModal').style.display = 'none';
    document.body.classList.remove('modal-open');
}

window.hideExportModal = function() {
    document.getElementById('exportModal').classList.remove('show');
    document.getElementById('exportModal').style.display = 'none';
    document.body.classList.remove('modal-open');
}

window.restoreVersion = function(modelId, version) {
    if (!confirm(`Are you sure you want to restore version ${version}?`)) {
        return;
    }

    fetch(`/api/models/${modelId}/versions/${version}/restore`, {
        method: 'POST'
    })
    .then(response => {
        if (!response.ok) {
            throw new Error('Failed to restore version');
        }
        showSuccessMessage(`Version ${version} restored successfully`);
        updateVersionHistory(modelId);
    })
    .catch(error => {
        console.error('Error restoring version:', error);
        showErrorMessage('Failed to restore version');
    });
}

window.deleteVersion = function(modelId, version) {
    if (!confirm(`Are you sure you want to delete version ${version}? This action cannot be undone.`)) {
        return;
    }

    fetch(`/api/models/${modelId}/versions/${version}`, {
        method: 'DELETE'
    })
    .then(response => {
        if (!response.ok) {
            throw new Error('Failed to delete version');
        }
        showSuccessMessage(`Version ${version} deleted successfully`);
        updateVersionHistory(modelId);
    })
    .catch(error => {
        console.error('Error deleting version:', error);
        showErrorMessage('Failed to delete version');
    });
}

function showSuccessMessage(message) {
    const toast = document.createElement('div');
    toast.className = 'toast show';
    toast.setAttribute('role', 'alert');
    toast.innerHTML = `
        <div class="toast-header bg-success text-white">
            <strong class="me-auto">Success</strong>
            <button type="button" class="btn-close" data-bs-dismiss="toast"></button>
        </div>
        <div class="toast-body">${message}</div>
    `;
    document.body.appendChild(toast);
    setTimeout(() => {
        toast.remove();
    }, 3000);
}

function showErrorMessage(message) {
    const toast = document.createElement('div');
    toast.className = 'toast show';
    toast.setAttribute('role', 'alert');
    toast.innerHTML = `
        <div class="toast-header bg-danger text-white">
            <strong class="me-auto">Error</strong>
            <button type="button" class="btn-close" data-bs-dismiss="toast"></button>
        </div>
        <div class="toast-body">${message}</div>
    `;
    document.body.appendChild(toast);
    setTimeout(() => {
        toast.remove();
    }, 3000);
} 
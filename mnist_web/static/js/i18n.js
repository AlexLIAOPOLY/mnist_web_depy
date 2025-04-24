// i18n.js - 多语言支持模块
window.MNIST = window.MNIST || {};

MNIST.i18n = (function() {
    // 默认语言
    let currentLang = localStorage.getItem('mnistLang') || 'en';
    // 可用语言
    const availableLangs = ['en', 'zh'];
    // 翻译文本
    const translations = {
        'en': {
            // 导航
            'nav_home': 'Home',
            'nav_train': 'Train Model',
            'nav_draw': 'Test Predictions',
            'nav_explore': 'Explore Dataset',
            'nav_models': 'Compare Models',
            
            // 首页内容
            'app_title': 'MNIST Web Application',
            'app_subtitle': 'Interactive Neural Network Training & Visualization',
            'welcome_title': 'Welcome to the MNIST Neural Network Visualizer',
            'welcome_desc': 'A lightweight application for training, testing and visualizing neural networks on the MNIST dataset.',
            'start_training': 'Start Training',
            'test_model': 'Test Model',
            'features': 'Features',
            
            // 特性卡片
            'feature_training': 'Interactive Training',
            'feature_training_desc': 'Train neural networks with custom parameters and watch the learning process in real-time.',
            'feature_draw': 'Draw & Predict',
            'feature_draw_desc': 'Draw your own digits and see how the model predicts them with probability visualization.',
            'feature_viz': 'Live Visualization',
            'feature_viz_desc': 'Watch accuracy and loss metrics change during training with interactive charts.',
            'feature_explore': 'Dataset Explorer',
            'feature_explore_desc': 'Browse through MNIST dataset samples and understand what the model is learning from.',
            
            // 快速指南
            'quickstart': 'Quick Start Guide',
            'step_train': 'Train a Model',
            'step_train_desc': 'Go to the Train page to create and train a neural network with custom parameters.',
            'step_test': 'Test Predictions',
            'step_test_desc': 'Visit the Draw page to test your trained model by drawing digits.',
            'step_explore': 'Explore Dataset',
            'step_explore_desc': 'Check out the Explore page to see samples from the MNIST dataset.',
            'step_compare': 'Compare Models',
            'step_compare_desc': 'Use the Models page to compare performance of different neural network configurations.',
            
            // 训练页面
            'train_title': 'Train Neural Network Model',
            'train_subtitle': 'Customize the parameters and start training a model on the MNIST dataset.',
            'train_params': 'Training Parameters',
            'hidden_size': 'Hidden Layer Size:',
            'batch_size': 'Batch Size:',
            'learning_rate': 'Learning Rate:',
            'epochs': 'Epochs:',
            'train_size': 'Training Set Size:',
            'activation': 'Activation Function:',
            'start_training_btn': 'Start Training',
            'stop_training_btn': 'Stop Training',
            'reset_model': 'Reset Model',
            'training_status': 'Training Status',
            'status_label': 'Status:',
            'current_epoch': 'Current Epoch:',
            'best_accuracy': 'Best Accuracy:',
            'current_loss': 'Current Loss:',
            'progress': 'Progress:',
            'training_visualization': 'Training Visualization',
            'sample_predictions': 'Sample Predictions',
            'sample_predictions_desc': 'See how the model performs on random test samples as it trains:',
            
            // 图表
            'train_accuracy': 'Training Accuracy',
            'val_accuracy': 'Validation Accuracy',
            'train_loss': 'Training Loss',
            'val_loss': 'Validation Loss',
            'accuracy_chart': 'Accuracy over Epochs',
            'loss_chart': 'Loss over Epochs',
            'epoch': 'Epoch',
            'accuracy': 'Accuracy',
            'loss': 'Loss',
            
            // 绘制页面
            'draw_title': 'Draw & Predict Numbers',
            'draw_subtitle': 'Draw a digit (0-9) on the canvas below and see the model\'s prediction.',
            'draw_instructions': 'Draw a single digit (0-9) here',
            'clear_canvas': 'Clear Canvas',
            'brush_size': 'Brush Size:',
            'predict_digit': 'Predict Digit',
            'save_image': 'Save Image',
            'prediction_result': 'Prediction Results',
            'predicted_digit': 'Predicted Digit:',
            'probability': 'Probability:',
            'probability_distribution': 'Probability Distribution:',
            'draw_prompt': 'Draw and click "Predict" to see results',
            'current_model': 'Current Model:',
            'model_type': 'Model Type:',
            'model_accuracy': 'Accuracy:',
            'model_trained': 'Last Trained:',
            'select_model': 'Select Model:',
            'latest_model': 'Latest Model',
            'best_model': 'Best Accuracy Model',
            'recent_predictions': 'Recent Predictions',
            'no_predictions': 'No predictions yet. Draw a digit and click "Predict" to start.',
            
            // 底部
            'footer_copyright': '© 2023 MNIST Web App. Built with',
            'footer_source': 'Source code available on',
            'and': 'and',
            
            // 状态文本
            'not_started': 'Not started',
            'training': 'Training...',
            'completed': 'Completed',
            'stopped': 'Stopped',
            
            // 语言
            'language': 'Language',
            'lang_en': 'English',
            'lang_zh': '中文'
        },
        'zh': {
            // 导航
            'nav_home': '首页',
            'nav_train': '训练模型',
            'nav_draw': '测试预测',
            'nav_explore': '探索数据集',
            'nav_models': '比较模型',
            
            // 首页内容
            'app_title': 'MNIST网页应用',
            'app_subtitle': '交互式神经网络训练与可视化',
            'welcome_title': '欢迎使用MNIST神经网络可视化工具',
            'welcome_desc': '一个轻量级应用程序，用于在MNIST数据集上训练、测试和可视化神经网络。',
            'start_training': '开始训练',
            'test_model': '测试模型',
            'features': '功能特点',
            
            // 特性卡片
            'feature_training': '交互式训练',
            'feature_training_desc': '使用自定义参数训练神经网络，并实时观察学习过程。',
            'feature_draw': '绘制与预测',
            'feature_draw_desc': '绘制您自己的数字，并通过概率可视化查看模型如何预测它们。',
            'feature_viz': '实时可视化',
            'feature_viz_desc': '通过交互式图表观察训练过程中准确率和损失度量的变化。',
            'feature_explore': '数据集浏览器',
            'feature_explore_desc': '浏览MNIST数据集样本，了解模型从中学习的内容。',
            
            // 快速指南
            'quickstart': '快速入门指南',
            'step_train': '训练模型',
            'step_train_desc': '转到训练页面，使用自定义参数创建和训练神经网络。',
            'step_test': '测试预测',
            'step_test_desc': '访问绘制页面，通过绘制数字测试您训练的模型。',
            'step_explore': '探索数据集',
            'step_explore_desc': '查看探索页面以查看MNIST数据集中的样本。',
            'step_compare': '比较模型',
            'step_compare_desc': '使用模型页面比较不同神经网络配置的性能。',
            
            // 训练页面
            'train_title': '训练神经网络模型',
            'train_subtitle': '自定义参数并在MNIST数据集上开始训练模型。',
            'train_params': '训练参数',
            'hidden_size': '隐藏层大小：',
            'batch_size': '批次大小：',
            'learning_rate': '学习率：',
            'epochs': '训练周期：',
            'train_size': '训练集大小：',
            'activation': '激活函数：',
            'start_training_btn': '开始训练',
            'stop_training_btn': '停止训练',
            'reset_model': '重置模型',
            'training_status': '训练状态',
            'status_label': '状态：',
            'current_epoch': '当前周期：',
            'best_accuracy': '最佳准确率：',
            'current_loss': '当前损失：',
            'progress': '进度：',
            'training_visualization': '训练可视化',
            'sample_predictions': '样本预测',
            'sample_predictions_desc': '查看模型在训练时如何在随机测试样本上表现：',
            
            // 图表
            'train_accuracy': '训练准确率',
            'val_accuracy': '验证准确率',
            'train_loss': '训练损失',
            'val_loss': '验证损失',
            'accuracy_chart': '准确率随周期变化',
            'loss_chart': '损失随周期变化',
            'epoch': '周期',
            'accuracy': '准确率',
            'loss': '损失',
            
            // 绘制页面
            'draw_title': '绘制与预测数字',
            'draw_subtitle': '在下方画布上绘制一个数字（0-9），并查看模型的预测结果。',
            'draw_instructions': '在此处绘制单个数字（0-9）',
            'clear_canvas': '清除画布',
            'brush_size': '笔刷大小：',
            'predict_digit': '预测数字',
            'save_image': '保存图像',
            'prediction_result': '预测结果',
            'predicted_digit': '预测数字：',
            'probability': '概率：',
            'probability_distribution': '概率分布：',
            'draw_prompt': '绘制并点击"预测数字"查看结果',
            'current_model': '当前模型：',
            'model_type': '模型类型：',
            'model_accuracy': '准确率：',
            'model_trained': '最后训练时间：',
            'select_model': '选择模型：',
            'latest_model': '最新模型',
            'best_model': '最佳准确率模型',
            'recent_predictions': '最近预测',
            'no_predictions': '暂无预测。绘制数字并点击"预测数字"开始。',
            
            // 底部
            'footer_copyright': '© 2023 MNIST网页应用。使用',
            'footer_source': '源代码可在以下位置获取',
            'and': '和',
            
            // 状态文本
            'not_started': '未开始',
            'training': '训练中...',
            'completed': '已完成',
            'stopped': '已停止',
            
            // 语言
            'language': '语言',
            'lang_en': 'English',
            'lang_zh': '中文'
        }
    };

    // 初始化
    function init() {
        console.log('MNIST i18n 模块初始化');
        
        // 确保所有资源加载完毕后再执行
        if (document.readyState === 'loading') {
            document.addEventListener('DOMContentLoaded', function() {
                setupLanguageSwitcher();
                applyTranslations();
                updateDynamicContent();
            });
        } else {
            // 如果文档已经加载完毕，直接执行
            setupLanguageSwitcher();
            applyTranslations();
            updateDynamicContent();
        }
    }

    // 设置语言切换器
    function setupLanguageSwitcher() {
        console.log('设置语言切换器');
        
        // 如果已经存在语言选择器，就不再创建
        if (document.getElementById('language-switcher')) {
            console.log('语言切换器已存在');
            return;
        }

        // 创建语言选择下拉菜单
        const langSwitcher = document.createElement('div');
        langSwitcher.id = 'language-switcher';
        langSwitcher.className = 'lang-switcher';
        
        const langLabel = document.createElement('span');
        langLabel.className = 'lang-label';
        langLabel.textContent = getTranslation('language') + ': ';
        
        const langSelect = document.createElement('select');
        langSelect.id = 'lang-select';
        
        for (const lang of availableLangs) {
            const option = document.createElement('option');
            option.value = lang;
            option.textContent = getTranslation('lang_' + lang);
            if (lang === currentLang) {
                option.selected = true;
            }
            langSelect.appendChild(option);
        }
        
        langSelect.addEventListener('change', function() {
            setLanguage(this.value);
        });
        
        langSwitcher.appendChild(langLabel);
        langSwitcher.appendChild(langSelect);
        
        // 添加到页面右上角固定位置
        const headerLangSwitcher = document.createElement('div');
        headerLangSwitcher.className = 'header-lang-switcher';
        headerLangSwitcher.appendChild(langSwitcher);
        document.body.appendChild(headerLangSwitcher);
    }

    // 应用翻译到页面
    function applyTranslations() {
        const elements = document.querySelectorAll('[data-i18n]');
        elements.forEach(el => {
            const key = el.getAttribute('data-i18n');
            const translation = getTranslation(key);
            if (translation) {
                if (el.tagName === 'INPUT' && el.type === 'button') {
                    el.value = translation;
                } else if (el.tagName === 'INPUT' || el.tagName === 'TEXTAREA') {
                    el.placeholder = translation;
                } else {
                    el.textContent = translation;
                }
            }
        });
    }

    // 更新动态内容
    function updateDynamicContent() {
        // 更新训练状态文本
        const trainingStatusValue = document.getElementById('training-status-value');
        if (trainingStatusValue) {
            const currentText = trainingStatusValue.textContent.trim().toLowerCase();
            if (currentText === 'not started') {
                trainingStatusValue.textContent = getTranslation('not_started');
            } else if (currentText === 'training...') {
                trainingStatusValue.textContent = getTranslation('training');
            } else if (currentText === 'completed') {
                trainingStatusValue.textContent = getTranslation('completed');
            } else if (currentText === 'stopped') {
                trainingStatusValue.textContent = getTranslation('stopped');
            }
        }
        
        // 更新绘制页面的文本
        const confidenceText = document.getElementById('confidence-text');
        if (confidenceText && confidenceText.textContent.includes('Draw and click')) {
            confidenceText.textContent = getTranslation('draw_prompt');
        }
        
        // 更新下拉菜单选项
        const modelSelector = document.getElementById('model-selector');
        if (modelSelector) {
            Array.from(modelSelector.options).forEach(option => {
                if (option.value === 'latest') {
                    option.textContent = getTranslation('latest_model');
                } else if (option.value === 'best') {
                    option.textContent = getTranslation('best_model');
                }
            });
        }
    }

    // 获取翻译文本
    function getTranslation(key) {
        return translations[currentLang] && translations[currentLang][key] ? 
            translations[currentLang][key] : 
            (translations['en'][key] || key);
    }

    // 设置语言
    function setLanguage(lang) {
        console.log('设置语言为: ' + lang);
        if (availableLangs.includes(lang)) {
            currentLang = lang;
            localStorage.setItem('mnistLang', lang);
            
            // 添加过渡效果
            document.body.classList.add('lang-changing');
            
            setTimeout(() => {
                applyTranslations();
                updateDynamicContent();
                
                // 更新选择器
                const langSelect = document.getElementById('lang-select');
                if (langSelect) {
                    langSelect.value = lang;
                }
                
                // 移除过渡效果
                setTimeout(() => {
                    document.body.classList.remove('lang-changing');
                }, 200);
                
                // 触发自定义事件
                document.dispatchEvent(new CustomEvent('languageChanged', { detail: { language: lang } }));
            }, 100);
        }
    }

    // 公共API
    return {
        init: init,
        getTranslation: getTranslation,
        setLanguage: setLanguage,
        getCurrentLanguage: function() { return currentLang; }
    };
})();

// 自动初始化
MNIST.i18n.init();

// 添加样式
document.addEventListener('DOMContentLoaded', function() {
    const style = document.createElement('style');
    style.textContent = `
    .header-lang-switcher {
        position: absolute;
        top: 10px;
        right: 10px;
        padding: 5px 10px;
        border-radius: 4px;
        z-index: 1000;
    }
    
    .header-lang-switcher select {
        background-color: white;
        color: #333;
        border: 1px solid #ccc;
        padding: 2px 5px;
        border-radius: 3px;
    }
    `;
    document.head.appendChild(style);
}); 
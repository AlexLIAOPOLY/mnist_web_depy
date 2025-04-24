#!/usr/bin/env python
# 导入修复 - 确保markupsafe兼容性
try:
    import sys
    import os
    
    # 检查并输出环境变量
    port = os.environ.get('PORT', '8080')
    print(f"PORT environment variable: {port}")
    
    # 尝试先导入numpy，如果失败则尝试安装
    try:
        print("Importing numpy in app.py...")
        import numpy as np
        print(f"NumPy imported successfully in app.py: {np.__version__}")
    except ImportError as e:
        print(f"Failed to import numpy in app.py: {e}")
        try:
            import subprocess
            print("Trying to install numpy...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", "numpy==1.16.6"])
            print("NumPy installed successfully!")
            import numpy as np
            print(f"NumPy imported after installation: {np.__version__}")
        except Exception as install_err:
            print(f"Failed to install numpy: {install_err}")
            sys.exit(1)
    
    # 将项目根目录添加到Python路径
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, project_root)
    print(f"Added project root to path: {project_root}")
    
    # 导入修复模块
    try:
        import fix_markupsafe
        print("MarkupSafe fix applied successfully in app.py")
    except Exception as e:
        print(f"Warning: Failed to apply markupsafe fix: {e}")
except Exception as e:
    print(f"Error during initialization: {e}")

# MNIST Web Application - Lightweight Flask web interface for MNIST dataset exploration and model training
try:
    from flask import Flask, render_template, request, jsonify, send_file, make_response
    import pickle
    import json
    import base64
    import time
    import threading
    import logging
    from io import BytesIO
    from PIL import Image
    from collections import defaultdict
    from werkzeug.exceptions import HTTPException
    from datetime import datetime, timedelta
    
    print("All standard libraries imported successfully!")
except ImportError as e:
    print(f"Error importing standard libraries: {e}")
    sys.exit(1)

# 确保必要的目录存在
try:
    os.makedirs('data', exist_ok=True)
    models_dir = os.path.join(os.path.dirname(__file__), 'static', 'models')
    os.makedirs(models_dir, exist_ok=True)
    print(f"Created necessary directories: data, {models_dir}")
except Exception as e:
    print(f"Error creating directories: {e}")

# Import MNIST model
try:
    # 尝试导入两种不同的路径
    model_import_paths = [
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),  # mnist_web_deploy
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),  # 项目根目录
    ]
    
    mnist_imported = False
    import_errors = []
    
    for path in model_import_paths:
        if path not in sys.path:
            sys.path.append(path)
            print(f"Added {path} to sys.path")
            
        try:
            print(f"Trying to import mnist_example from {path}")
            from mnist_example import SimpleNeuralNetwork, load_mnist, one_hot_encode
            print(f"Successfully imported mnist_example from {path}")
            mnist_imported = True
            break
        except ImportError as e:
            import_errors.append(f"Failed to import from {path}: {e}")
            continue
    
    if not mnist_imported:
        error_msg = "Failed to import mnist_example from all paths:\n" + "\n".join(import_errors)
        print(error_msg)
        raise ImportError(error_msg)
    
except Exception as e:
    print(f"Critical error importing mnist_example: {e}")
    print(f"Current sys.path: {sys.path}")
    sys.exit(1)

# Configure logging
try:
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(os.path.dirname(__file__), 'app.log')),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)
    print("Logging configured successfully")
except Exception as e:
    print(f"Error configuring logging: {e}")
    logger = None

# 初始化Flask应用
try:
    app = Flask(__name__)
    print("Flask app initialized successfully")
    
    # 输出端口绑定信息
    port = int(os.environ.get('PORT', 8080))
    print(f"Flask will bind to port: {port}")
except Exception as e:
    print(f"Error initializing Flask app: {e}")
    sys.exit(1)

# Global variables to store training state
training_thread = None
training_log = []
is_training = False
current_model = None
current_accuracy = 0
training_progress = 0
training_history = {
    'accuracy': [],
    'val_accuracy': [],
    'loss': [],
    'val_loss': []
}
model_registry = {}
last_training_time = None

# Preload MNIST dataset if environment variable is set
try:
    preload = os.environ.get('MNIST_PRELOAD_DATA', '0') == '1'
    if preload:
        logger.info("Preloading MNIST dataset...")
        print("Preloading MNIST dataset...")
        x_train, y_train, x_test, y_test = load_mnist()
        logger.info("MNIST dataset loaded successfully")
        print("MNIST dataset loaded successfully")
    else:
        x_train, y_train, x_test, y_test = None, None, None, None
except Exception as e:
    logger.error(f"Failed to load MNIST dataset: {e}")
    print(f"Failed to load MNIST dataset: {e}")
    x_train, y_train, x_test, y_test = None, None, None, None

def lazy_load_mnist():
    """Lazy load MNIST dataset if not already loaded"""
    global x_train, y_train, x_test, y_test
    if x_train is None:
        try:
            logger.info("Lazy loading MNIST dataset...")
            print("Lazy loading MNIST dataset...")
            x_train, y_train, x_test, y_test = load_mnist()
            logger.info("MNIST dataset loaded successfully")
            print("MNIST dataset loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load MNIST dataset: {e}")
            print(f"Failed to load MNIST dataset: {e}")
            raise RuntimeError(f"Failed to load MNIST dataset: {e}")

def train_model_thread(hidden_size, batch_size, learning_rate, epochs, train_size, activation='relu'):
    """Train model in a background thread"""
    global training_log, is_training, current_model, current_accuracy, training_progress, training_history, last_training_time, model_registry
    
    # Initialize logs and status
    training_log = []
    is_training = True
    training_progress = 0
    training_history = {
        'accuracy': [],
        'val_accuracy': [],
        'loss': [],
        'val_loss': []
    }
    
    # Ensure dataset is loaded
    lazy_load_mnist()
    
    try:
        # Use training subset
        x_train_subset = x_train[:train_size]
        y_train_subset = y_train[:train_size]
        
        # Split validation set
        val_size = min(1000, int(train_size * 0.1))
        x_val = x_train[train_size:train_size+val_size]
        y_val = y_train[train_size:train_size+val_size]
        
        # Initialize model
        input_size = 28 * 28
        output_size = 10
        model = SimpleNeuralNetwork(input_size, hidden_size, output_size, activation=activation)
        model_id = f"model_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Manually implement training loop to log progress
        num_samples = x_train_subset.shape[0]
        num_batches = num_samples // batch_size
        
        # Convert labels to one-hot encoding
        y_one_hot = one_hot_encode(y_train_subset)
        y_val_one_hot = one_hot_encode(y_val)
        
        best_val_accuracy = 0
        best_model = None
        
        for epoch in range(epochs):
            if not is_training:  # Check if training should stop
                break
                
            epoch_start_time = time.time()
            epoch_loss = 0
            
            # Shuffle training data
            indices = np.random.permutation(num_samples)
            X_shuffled = x_train_subset[indices]
            y_shuffled = y_one_hot[indices]
            
            # Mini-batch gradient descent
            for i in range(num_batches):
                if not is_training:  # Check if training should stop
                    break
                    
                start_idx = i * batch_size
                end_idx = min((i + 1) * batch_size, num_samples)
                
                X_batch = X_shuffled[start_idx:end_idx]
                y_batch = y_shuffled[start_idx:end_idx]
                
                # Forward pass
                output = model.forward(X_batch)
                
                # Calculate loss
                batch_loss = model.calculate_loss(output, y_batch)
                epoch_loss += batch_loss
                
                # Backward pass
                model.backward(X_batch, y_batch, output, learning_rate)
                
                # Update progress
                training_progress = (epoch * num_batches + i + 1) / (epochs * num_batches)
            
            # Calculate average epoch loss
            avg_epoch_loss = epoch_loss / num_batches
            
            # Calculate training accuracy
            train_predictions = np.argmax(model.forward(x_train_subset), axis=1)
            train_accuracy = np.mean(train_predictions == y_train_subset)
            
            # Calculate validation accuracy and loss
            val_output = model.forward(x_val)
            val_predictions = np.argmax(val_output, axis=1)
            val_accuracy = np.mean(val_predictions == y_val)
            val_loss = model.calculate_loss(val_output, y_val_one_hot)
            
            # Check if this is the best model so far
            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                best_model = model.copy()
                
            epoch_time = time.time() - epoch_start_time
            
            # Log entry
            log_entry = {
                "epoch": epoch + 1,
                "time": epoch_time,
                "loss": float(avg_epoch_loss),
                "val_loss": float(val_loss),
                "train_accuracy": float(train_accuracy),
                "val_accuracy": float(val_accuracy)
            }
            training_log.append(log_entry)
            
            # Update training history
            training_history['accuracy'].append(float(train_accuracy))
            training_history['val_accuracy'].append(float(val_accuracy))
            training_history['loss'].append(float(avg_epoch_loss))
            training_history['val_loss'].append(float(val_loss))
            
            logger.info(f"Epoch {epoch+1}/{epochs}, Time: {epoch_time:.2f}s, Loss: {avg_epoch_loss:.4f}, "
                        f"Train Accuracy: {train_accuracy:.4f}, Val Accuracy: {val_accuracy:.4f}")
        
        # Evaluate test accuracy
        test_accuracy = model.evaluate(x_test, y_test)
        current_accuracy = float(test_accuracy)
        
        # Save both final and best models
        models_dir = os.path.join(os.path.dirname(__file__), 'static', 'models')
        os.makedirs(models_dir, exist_ok=True)
        
        # Save final model
        final_model_path = os.path.join(models_dir, f"{model_id}_final.pkl")
        model.save(final_model_path)
        
        # Save best model
        best_model_path = os.path.join(models_dir, f"{model_id}_best.pkl")
        if best_model:
            best_model.save(best_model_path)
            best_test_accuracy = best_model.evaluate(x_test, y_test)
        else:
            best_test_accuracy = test_accuracy
            best_model_path = final_model_path
        
        # Save current model for predictions
        current_model_path = os.path.join(models_dir, 'current_model.pkl')
        model.save(current_model_path)
        current_model = model
        
        # Update model registry
        model_registry[model_id] = {
            'id': model_id,
            'name': f"Model {len(model_registry) + 1}",
            'description': f"Hidden Size: {hidden_size}, Activation: {activation}",
            'created_at': datetime.now().isoformat(),
            'params': {
                'hidden_size': hidden_size,
                'activation': activation,
                'learning_rate': learning_rate,
                'batch_size': batch_size,
                'epochs': epoch + 1 if not is_training else epochs,
                'train_size': train_size
            },
            'performance': {
                'accuracy': float(test_accuracy),
                'val_accuracy': float(best_test_accuracy),
                'training_time': sum([log['time'] for log in training_log])
            },
            'final_model_path': final_model_path,
            'best_model_path': best_model_path
        }
        
        # Save model registry
        with open(os.path.join(models_dir, 'registry.json'), 'w') as f:
            json.dump(model_registry, f, indent=2)
        
        last_training_time = datetime.now().isoformat()
        logger.info(f"Model training completed. Test accuracy: {test_accuracy:.4f}")
    except Exception as e:
        logger.error(f"Error during model training: {e}")
    finally:
        is_training = False
        training_progress = 1.0 if training_progress > 0 else 0

def preprocess_image(image_data):
    """Preprocess image data for prediction
    
    Args:
        image_data: Either a PIL Image object or base64 encoded image string
        
    Returns:
        Preprocessed numpy array ready for model prediction
    """
    try:
        if isinstance(image_data, str):
            # 处理Base64编码的图像数据
            if ',' in image_data:
                image_data = image_data.split(',')[1]
            image_bytes = base64.b64decode(image_data)
            image = Image.open(BytesIO(image_bytes)).convert('L')  # Convert to grayscale
        elif isinstance(image_data, Image.Image):
            # 已经是PIL Image对象
            image = image_data.convert('L')  # 确保是灰度图
        else:
            raise ValueError("Unsupported image data type. Expected PIL Image or base64 string.")
            
        # Resize to 28x28
        image = image.resize((28, 28))
        
        # Convert to numpy array and normalize
        img_array = np.array(image).astype(float)
        
        # Invert colors if needed (assuming white digit on black background in MNIST)
        # Check if the image is mostly bright (assumes drawing is black on white)
        if img_array.mean() > 128:
            img_array = 255 - img_array
            
        # Normalize to 0-1 range
        img_array = img_array / 255.0
        
        # Threshold to help with anti-aliasing and make it more MNIST-like
        img_array = (img_array > 0.3).astype(float)
        
        # Reshape for the model (1, 784)
        img_array = img_array.reshape(1, 28*28)
        
        return img_array
        
    except Exception as e:
        logger.error(f"Image preprocessing error: {e}")
        raise

def load_model_registry():
    """Load model registry from disk"""
    global model_registry
    registry_path = os.path.join(os.path.dirname(__file__), 'static', 'models', 'registry.json')
    if os.path.exists(registry_path):
        try:
            logger.info(f"正在从 {registry_path} 加载模型注册表")
            with open(registry_path, 'r') as f:
                model_registry = json.load(f)
            logger.info(f"模型注册表加载成功，包含 {len(model_registry)} 个模型")
            
            # 记录每个模型的基本信息
            for model_id, model_info in model_registry.items():
                logger.info(f"模型 {model_id}: {model_info.get('name', 'Unnamed')}, "
                          f"准确度: {model_info.get('performance', {}).get('accuracy', 0)}, "
                          f"创建时间: {model_info.get('created_at', 'Unknown')}")
                
                # 检查模型文件是否存在
                model_path = model_info.get('best_model_path')
                if model_path and os.path.exists(model_path):
                    logger.info(f"模型文件存在: {model_path}")
                else:
                    logger.warning(f"模型文件不存在: {model_path}")
                    
            return model_registry
        except Exception as e:
            logger.error(f"加载模型注册表失败: {e}")
            import traceback
            logger.error(traceback.format_exc())
    else:
        logger.warning(f"模型注册表文件不存在: {registry_path}")
    
    # 返回空字典而不是False，保持一致性
    return {}

def load_specific_model(model_path):
    """Load a specific model from path"""
    if os.path.exists(model_path):
        try:
            return SimpleNeuralNetwork.load(model_path)
        except Exception as e:
            logger.error(f"Failed to load model from {model_path}: {e}")
    return None

@app.route('/')
def index():
    """Home page"""
    return render_template('index.html')

@app.route('/train')
def train():
    """Training page"""
    return render_template('train.html')

@app.route('/draw')
def test():
    """Drawing/testing page"""
    return render_template('draw.html')

@app.route('/explore')
def explore():
    """Dataset exploration page"""
    return render_template('explore.html')

@app.route('/visualize')
def visualize():
    """Visualization page"""
    return render_template('visualize.html')

@app.route('/models')
def models():
    """Model comparison page"""
    return render_template('models.html')

@app.route('/models_fix')
def models_fix():
    """Modified model comparison page with server-side rendering"""
    # 加载模型注册表
    success = load_model_registry()
    
    # 获取models.html的原始内容
    template_path = os.path.join(os.path.dirname(__file__), 'templates', 'models.html')
    with open(template_path, 'r') as file:
        content = file.read()
    
    # 如果有模型，修改页面内容
    if model_registry:
        # 移除"No models have been trained yet"消息
        content = content.replace('<p class="empty-message">No models have been trained yet. Go to the <a href="/train" onclick="return MNIST.animations.pageTransition(\'/train\')">Train</a> page to create your first model.</p>', 
                                 '<p class="empty-message" style="display:none;">No models have been trained yet. Go to the <a href="/train" onclick="return MNIST.animations.pageTransition(\'/train\')">Train</a> page to create your first model.</p>')
        
        # 构建模型列表HTML
        models_html = ""
        for model_id, model_info in model_registry.items():
            accuracy = model_info.get('performance', {}).get('accuracy', 0) * 100
            hidden_size = model_info.get('params', {}).get('hidden_size', 'N/A')
            activation = model_info.get('params', {}).get('activation', 'relu')
            name = model_info.get('name', f'Model {model_id[-8:]}')
            
            models_html += f'''
            <div class="model-item" data-id="{model_id}">
                <div class="model-header">
                    <input type="checkbox" class="model-checkbox">
                    <h4 class="model-name">{name}</h4>
                </div>
                <div class="model-details">
                    <div class="model-metrics">
                        <div class="metric">
                            <span class="metric-label">Accuracy:</span>
                            <span class="metric-value">{accuracy:.2f}%</span>
                        </div>
                        <div class="metric">
                            <span class="metric-label">Hidden Size:</span>
                            <span class="metric-value">{hidden_size}</span>
                        </div>
                        <div class="metric">
                            <span class="metric-label">Activation:</span>
                            <span class="metric-value">{activation}</span>
                        </div>
                    </div>
                    <div class="model-actions">
                        <button class="btn btn-sm view-details-btn" data-id="{model_id}">
                            <i class="fas fa-info-circle"></i> Details
                        </button>
                    </div>
                </div>
            </div>
            '''
        
        # 插入模型列表到页面
        content = content.replace('<!-- Model items will be dynamically generated -->', 
                                  f'<!-- Model items are server-side generated -->\n{models_html}')
    
    # 返回修改后的内容
    response = make_response(content)
    response.headers['Content-Type'] = 'text/html'
    return response

@app.route('/api/train', methods=['POST'])
def api_train():
    """API endpoint to start training process"""
    global training_thread, is_training
    
    if is_training:
        return jsonify({"status": "error", "message": "A training task is already running"})
    
    try:
        data = request.get_json()
        hidden_size = int(data.get('hidden_size', 128))
        batch_size = int(data.get('batch_size', 64))
        learning_rate = float(data.get('learning_rate', 0.1))
        epochs = int(data.get('epochs', 5))
        train_size = int(data.get('train_size', 10000))
        activation = data.get('activation', 'relu')
        
        # Start training thread
        training_thread = threading.Thread(
            target=train_model_thread,
            args=(hidden_size, batch_size, learning_rate, epochs, train_size, activation)
        )
        training_thread.start()
        
        return jsonify({"status": "success", "message": "Training started"})
    except Exception as e:
        logger.error(f"Failed to start training: {e}")
        return jsonify({"status": "error", "message": str(e)})

@app.route('/api/stop_training', methods=['POST'])
def api_stop_training():
    """API endpoint to stop training"""
    global is_training
    
    if not is_training:
        return jsonify({"status": "error", "message": "No training task is running"})
    
    is_training = False
    return jsonify({"status": "success", "message": "Signal sent to stop training"})

@app.route('/api/training_status')
def api_training_status():
    """API endpoint to get training status"""
    return jsonify({
        "is_training": is_training,
        "progress": training_progress,
        "log": training_log,
        "accuracy": current_accuracy,
        "history": training_history,
        "last_training_time": last_training_time
    })

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """API endpoint for making predictions"""
    try:
        # 获取请求数据
        data = request.get_json()
        logger.info("接收到预测请求")
        
        if 'image' not in data:
            logger.warning("请求中没有图像数据")
            return jsonify({'status': 'error', 'message': 'No image data provided'})
        
        # 获取模型ID，默认为latest
        model_id = data.get('model_id', 'latest')
        logger.info(f"使用模型ID: {model_id}")
        
        # 处理图像数据
        image_data = data['image'].split(',')[1]
        image_bytes = base64.b64decode(image_data)
        img = Image.open(BytesIO(image_bytes)).convert('L')  # Convert to grayscale
        logger.info(f"图像尺寸: {img.size}")
        
        # 预处理图像
        img_array = preprocess_image(img)
        logger.info(f"预处理后图像形状: {img_array.shape}")
        
        # 选择要使用的模型
        model = None
        if model_id == 'latest':
            # 使用最新训练的模型
            logger.info("尝试加载最新模型")
            if current_model:
                logger.info("使用当前内存中的模型")
                model = current_model
            else:
                # 尝试加载保存的当前模型
                model_path = os.path.join(os.path.dirname(__file__), 'static', 'models', 'current_model.pkl')
                logger.info(f"尝试从文件加载当前模型: {model_path}")
                if os.path.exists(model_path):
                    model = load_specific_model(model_path)
                    if model:
                        logger.info("成功加载当前模型")
                    else:
                        logger.warning("加载当前模型失败")
                else:
                    logger.warning(f"当前模型文件不存在: {model_path}")
        elif model_id == 'best':
            # 使用最佳精度模型
            logger.info("尝试加载最佳精度模型")
            registry = load_model_registry()
            if registry:
                # 按精度排序，找出最佳模型
                try:
                    best_model_id = max(registry.items(), key=lambda x: x[1]['performance']['accuracy'])[0]
                    best_model_path = registry[best_model_id]['best_model_path']
                    logger.info(f"找到最佳模型: {best_model_id}, 路径: {best_model_path}")
                    if os.path.exists(best_model_path):
                        model = load_specific_model(best_model_path)
                        if model:
                            logger.info("成功加载最佳模型")
                        else:
                            logger.warning("加载最佳模型失败")
                    else:
                        logger.warning(f"最佳模型文件不存在: {best_model_path}")
                except Exception as e:
                    logger.error(f"在查找最佳模型时出错: {e}")
            else:
                logger.warning("没有找到模型注册表或注册表为空")
        else:
            # 使用指定ID的模型
            logger.info(f"尝试加载指定ID的模型: {model_id}")
            registry = load_model_registry()
            if registry and model_id in registry:
                model_path = registry[model_id]['best_model_path']
                logger.info(f"找到指定模型: {model_id}, 路径: {model_path}")
                if os.path.exists(model_path):
                    model = load_specific_model(model_path)
                    if model:
                        logger.info("成功加载指定模型")
                    else:
                        logger.warning("加载指定模型失败")
                else:
                    logger.warning(f"指定模型文件不存在: {model_path}")
            else:
                logger.warning(f"在注册表中未找到指定ID的模型: {model_id}")
        
        if not model:
            logger.error("没有可用的模型进行预测")
            return jsonify({'status': 'error', 'message': 'No model available for prediction'})
        
        # 进行预测
        probs = model.forward(img_array)
        prediction = np.argmax(probs)
        probabilities = probs[0].tolist()
        
        # 记录预测结果
        logger.info(f"预测结果: {prediction}, 最高概率: {max(probabilities):.4f}")
        
        return jsonify({
            'status': 'success',
            'prediction': int(prediction),  # Convert to int for JSON serialization
            'probabilities': probabilities
        })
        
    except Exception as e:
        logger.error(f"预测错误: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/api/dataset_stats')
def api_dataset_stats():
    """API endpoint to get dataset statistics"""
    try:
        lazy_load_mnist()
        
        # Calculate digit distribution
        train_distribution = [0] * 10
        test_distribution = [0] * 10
        
        for label in y_train:
            train_distribution[label] += 1
            
        for label in y_test:
            test_distribution[label] += 1
        
        return jsonify({
            "status": "success",
            "stats": {
                "total_images": len(y_train) + len(y_test),
                "training_set": len(y_train),
                "test_set": len(y_test),
                "image_size": "28x28",
                "class_count": 10,
                "train_distribution": train_distribution,
                "test_distribution": test_distribution
            }
        })
    except Exception as e:
        logger.error(f"Failed to get dataset statistics: {e}")
        return jsonify({"status": "error", "message": str(e)})

@app.route('/api/dataset_samples', methods=['POST'])
def api_dataset_samples():
    """Get random samples from the dataset"""
    try:
        # Get parameters from request
        data = request.json
        dataset_type = data.get('dataset', 'test')
        digit = data.get('digit')
        count = data.get('count', 100)
        with_predictions = data.get('with_predictions', False)
        
        # Ensure dataset is loaded
        lazy_load_mnist()
        
        # Choose dataset
        if dataset_type == 'train':
            x_data, y_data = x_train, y_train
        else:
            x_data, y_data = x_test, y_test
        
        # Filter by digit if specified
        if digit is not None:
            indices = np.where(y_data == digit)[0]
        else:
            indices = np.arange(len(y_data))
        
        # Handle edge case where there are fewer samples than requested
        if len(indices) <= count:
            selected_indices = indices
        else:
            selected_indices = np.random.choice(indices, size=count, replace=False)
        
        # Load model if predictions are requested
        model = None
        if with_predictions:
            # Check if current model exists
            if current_model is not None:
                model = current_model
            # If no model exists, try to load a model from the registry
            elif os.path.exists(os.path.join(os.path.dirname(__file__), 'static', 'models', 'current_model.pkl')):
                try:
                    model_path = os.path.join(os.path.dirname(__file__), 'static', 'models', 'current_model.pkl')
                    model = load_specific_model(model_path)
                except Exception as e:
                    logger.error(f"Failed to load model for predictions: {e}")
        
        # Prepare sample data
        samples = []
        for idx in selected_indices:
            image = x_data[idx].reshape(28, 28)
            true_label = int(y_data[idx])
            
            # Convert image to base64 for sending to client
            img_pil = Image.fromarray((image * 255).astype(np.uint8))
            buffered = BytesIO()
            img_pil.save(buffered, format="PNG")
            img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
            
            sample_data = {
                'index': int(idx),
                'true_label': true_label,
                'image': img_base64
            }
            
            # Add prediction if requested and model is available
            if with_predictions and model:
                try:
                    # Get confidence scores
                    x = x_data[idx].reshape(1, -1)
                    
                    # Forward pass
                    output = model.forward(x)
                    confidences = output[0]
                    predicted_label = np.argmax(confidences)
                    
                    # Add prediction data
                    sample_data.update({
                        'predicted_label': int(predicted_label),
                        'confidence': float(confidences[predicted_label]),
                        'all_confidences': {str(i): float(conf) for i, conf in enumerate(confidences)}
                    })
                except Exception as e:
                    logger.error(f"Error generating prediction for sample {idx}: {e}")
                    # Add placeholder values if prediction fails
                    sample_data.update({
                        'predicted_label': 0,
                        'confidence': 0.0,
                        'all_confidences': {str(i): 0.0 for i in range(10)}
                    })
            elif with_predictions:
                # If predictions were requested but no model is available
                sample_data.update({
                    'predicted_label': 0,
                    'confidence': 0.0,
                    'all_confidences': {str(i): 0.0 for i in range(10)}
                })
            
            # Add sample to results
            samples.append(sample_data)
        
        # Return samples
        return jsonify({
            'status': 'success',
            'dataset': dataset_type,
            'samples': samples,
            'filtered_count': len(indices),
            'total_count': len(y_data),
            'with_predictions': with_predictions,
            'model_available': model is not None
        })
        
    except Exception as e:
        logger.error(f"Error getting dataset samples: {e}")
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/api/models/<model_id>')
def api_model_detail(model_id):
    """API endpoint to get detailed information about a specific model"""
    load_model_registry()
    
    if model_id not in model_registry:
        return jsonify({
            "status": "error",
            "message": f"Model with ID {model_id} not found"
        }), 404
    
    model_info = model_registry[model_id]
    
    # Try to load the model to get additional information if available
    try:
        model_path = model_info.get('best_model_path', model_info.get('final_model_path'))
        if model_path and os.path.exists(model_path):
            model = load_specific_model(model_path)
            
            # Add architecture details if available
            if model:
                arch_info = {
                    "input_size": model.input_size,
                    "hidden_size": model.hidden_size,
                    "output_size": model.output_size,
                    "activation": model.activation,
                    "description": f"Simple Neural Network ({model.input_size}-{model.hidden_size}-{model.output_size})"
                }
                model_info["architecture"] = arch_info
    except Exception as e:
        logger.error(f"Error getting model details for {model_id}: {e}")
        # Continue without additional details
    
    # Calculate training time if possible
    if "history" in model_info and len(model_info["history"].get("accuracy", [])) > 0:
        # Estimate training time
        training_time = len(model_info["history"]["accuracy"]) * 5  # Rough estimate: 5 seconds per epoch
        model_info["estimated_training_time"] = training_time
    
    # Add formatted timestamps
    if "timestamp" in model_info:
        try:
            # Try to format the timestamp for display
            from datetime import datetime
            timestamp = model_info["timestamp"]
            if isinstance(timestamp, str):
                timestamp = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                model_info["created_at"] = timestamp.strftime("%Y-%m-%d %H:%M:%S")
        except Exception as e:
            logger.error(f"Error formatting timestamp for {model_id}: {e}")
    
    # Transform parameters for easier display
    if "parameters" in model_info:
        model_info["hyperparams"] = model_info["parameters"]
    
    # Add accuracy and loss information
    model_info["accuracy"] = model_info.get("best_accuracy", model_info.get("final_accuracy", 0))
    
    # If we have training history, get the final loss
    if "history" in model_info and len(model_info["history"].get("loss", [])) > 0:
        model_info["loss"] = model_info["history"]["loss"][-1]
    
    return jsonify({
        "status": "success",
        "model": model_info
    })

@app.route('/api/visualize/confusion_matrix', methods=['POST'])
def api_visualize_confusion_matrix():
    """Generate confusion matrix visualization data"""
    try:
        # Get parameters from request
        data = request.json
        model_id = data.get('model_id')
        dataset_type = data.get('dataset', 'test')
        
        # Ensure dataset is loaded
        lazy_load_mnist()
        
        # Choose dataset
        if dataset_type == 'train':
            x_data, y_data = x_train, y_train
        else:
            x_data, y_data = x_test, y_test
        
        # Load model
        if model_id:
            if model_id not in model_registry:
                return jsonify({'status': 'error', 'message': f'Model {model_id} not found'})
            model_path = model_registry[model_id]['best_model_path']
            model = load_specific_model(model_path)
        else:
            if current_model is None:
                return jsonify({'status': 'error', 'message': 'No current model available'})
            model = current_model
        
        # Make predictions
        y_pred = model.predict(x_data)
        
        # Calculate confusion matrix
        confusion_matrix = np.zeros((10, 10), dtype=int)
        for i in range(len(y_data)):
            true_label = y_data[i]
            pred_label = y_pred[i]
            confusion_matrix[true_label][pred_label] += 1
        
        # Calculate accuracy
        accuracy = np.sum(np.diag(confusion_matrix)) / np.sum(confusion_matrix)
        
        # Return data
        return jsonify({
            'status': 'success',
            'confusion_matrix': confusion_matrix.tolist(),
            'accuracy': float(accuracy),
            'dataset': dataset_type,
            'model_id': model_id or 'current'
        })
        
    except Exception as e:
        logger.error(f"Error generating confusion matrix: {e}")
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/api/visualize/prediction_confidence', methods=['POST'])
def api_visualize_prediction_confidence():
    """Generate prediction confidence visualization data"""
    try:
        # Get parameters from request
        data = request.json
        model_id = data.get('model_id')
        dataset_type = data.get('dataset', 'test')
        digit = data.get('digit')
        count = data.get('count', 100)
        
        # Ensure dataset is loaded
        lazy_load_mnist()
        
        # Choose dataset
        if dataset_type == 'train':
            x_data, y_data = x_train, y_train
        else:
            x_data, y_data = x_test, y_test
        
        # Filter by digit if specified
        if digit is not None:
            indices = np.where(y_data == digit)[0]
        else:
            indices = np.arange(len(y_data))
        
        # Handle edge case where there are fewer samples than requested
        if len(indices) <= count:
            selected_indices = indices
        else:
            selected_indices = np.random.choice(indices, size=count, replace=False)
        
        # Load model
        if model_id:
            if model_id not in model_registry:
                return jsonify({'status': 'error', 'message': f'Model {model_id} not found'})
            model_path = model_registry[model_id]['best_model_path']
            model = load_specific_model(model_path)
        else:
            if current_model is None:
                return jsonify({'status': 'error', 'message': 'No current model available'})
            model = current_model
        
        # Make predictions with confidence scores
        predictions = []
        for idx in selected_indices:
            # Get confidence scores
            x = x_data[idx].reshape(1, -1)
            true_label = int(y_data[idx])
            
            # Forward pass
            output = model.forward(x)
            confidences = output[0]
            predicted_label = np.argmax(confidences)
            
            # Convert image to base64 for sending to client
            image = x_data[idx].reshape(28, 28)
            img_pil = Image.fromarray((image * 255).astype(np.uint8))
            buffered = BytesIO()
            img_pil.save(buffered, format="PNG")
            img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
            
            # Add to results
            predictions.append({
                'index': int(idx),
                'true_label': true_label,
                'predicted_label': int(predicted_label),
                'confidence': float(confidences[predicted_label]),
                'all_confidences': {str(i): float(conf) for i, conf in enumerate(confidences)},
                'image': img_base64
            })
        
        # Return data
        return jsonify({
            'status': 'success',
            'predictions': predictions,
            'dataset': dataset_type,
            'digit': digit,
            'model_id': model_id or 'current'
        })
        
    except Exception as e:
        logger.error(f"Error generating prediction confidence data: {e}")
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/api/visualize/misclassified', methods=['POST'])
def api_visualize_misclassified():
    """Get misclassified examples"""
    try:
        # Get parameters from request
        data = request.json
        model_id = data.get('model_id')
        dataset_type = data.get('dataset', 'test')
        digit = data.get('digit')
        count = data.get('count', 100)
        
        # Ensure dataset is loaded
        lazy_load_mnist()
        
        # Choose dataset
        if dataset_type == 'train':
            x_data, y_data = x_train, y_train
        else:
            x_data, y_data = x_test, y_test
        
        # Load model
        if model_id:
            if model_id not in model_registry:
                return jsonify({'status': 'error', 'message': f'Model {model_id} not found'})
            model_path = model_registry[model_id]['best_model_path']
            model = load_specific_model(model_path)
        else:
            if current_model is None:
                return jsonify({'status': 'error', 'message': 'No current model available'})
            model = current_model
        
        # Make predictions
        outputs = model.forward(x_data)
        y_pred = np.argmax(outputs, axis=1)
        
        # Find misclassified examples
        misclassified_indices = np.where(y_pred != y_data)[0]
        
        # Filter by digit if specified
        if digit is not None:
            misclassified_indices = np.array([idx for idx in misclassified_indices if y_data[idx] == digit])
        
        # Handle edge case where there are fewer samples than requested
        if len(misclassified_indices) <= count:
            selected_indices = misclassified_indices
        else:
            selected_indices = np.random.choice(misclassified_indices, size=count, replace=False)
        
        # Prepare misclassified data
        misclassified = []
        for idx in selected_indices:
            # Get confidence scores
            x = x_data[idx].reshape(1, -1)
            true_label = int(y_data[idx])
            
            # Forward pass
            output = model.forward(x)
            confidences = output[0]
            predicted_label = np.argmax(confidences)
            
            # Convert image to base64 for sending to client
            image = x_data[idx].reshape(28, 28)
            img_pil = Image.fromarray((image * 255).astype(np.uint8))
            buffered = BytesIO()
            img_pil.save(buffered, format="PNG")
            img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
            
            # Add to results
            misclassified.append({
                'index': int(idx),
                'true_label': true_label,
                'predicted_label': int(predicted_label),
                'confidence': float(confidences[predicted_label]),
                'all_confidences': {str(i): float(conf) for i, conf in enumerate(confidences)},
                'image': img_base64
            })
        
        # Return data
        return jsonify({
            'status': 'success',
            'misclassified': misclassified,
            'dataset': dataset_type,
            'digit': digit,
            'total_misclassified': len(misclassified_indices),
            'total_samples': len(y_data),
            'error_rate': float(len(misclassified_indices) / len(y_data)),
            'model_id': model_id or 'current'
        })
        
    except Exception as e:
        logger.error(f"Error getting misclassified examples: {e}")
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/api/visualize/tsne', methods=['POST'])
def api_visualize_tsne():
    """Generate t-SNE visualization data"""
    try:
        # Check if sklearn is available
        try:
            from sklearn.manifold import TSNE
        except ImportError:
            return jsonify({
                'status': 'error', 
                'message': 'scikit-learn is not installed. Please install it with pip install scikit-learn'
            })
        
        # Get parameters from request
        data = request.json
        model_id = data.get('model_id')
        dataset_type = data.get('dataset', 'test')
        sample_size = min(data.get('sample_size', 500), 2000)  # Limit to 2000 samples max
        
        # Ensure dataset is loaded
        lazy_load_mnist()
        
        # Choose dataset
        if dataset_type == 'train':
            x_data, y_data = x_train, y_train
        else:
            x_data, y_data = x_test, y_test
        
        # Sample data to avoid slow computation
        if len(x_data) > sample_size:
            indices = np.random.choice(len(x_data), size=sample_size, replace=False)
            x_sampled = x_data[indices]
            y_sampled = y_data[indices]
            original_indices = indices
        else:
            x_sampled = x_data
            y_sampled = y_data
            original_indices = np.arange(len(x_data))
        
        # Load model if specified (for feature extraction)
        if model_id:
            if model_id not in model_registry:
                return jsonify({'status': 'error', 'message': f'Model {model_id} not found'})
            model_path = model_registry[model_id]['best_model_path']
            model = load_specific_model(model_path)
            
            # Extract features from hidden layer (if model supports it)
            if hasattr(model, 'get_hidden_features'):
                features = model.get_hidden_features(x_sampled)
            else:
                # Use raw pixel data
                features = x_sampled
        else:
            # Use raw pixel data
            features = x_sampled
        
        # Apply t-SNE
        tsne = TSNE(n_components=2, random_state=42)
        features_2d = tsne.fit_transform(features)
        
        # Prepare results
        tsne_results = []
        for i in range(len(features_2d)):
            tsne_results.append({
                'x': float(features_2d[i, 0]),
                'y': float(features_2d[i, 1]),
                'label': int(y_sampled[i]),
                'index': int(original_indices[i])
            })
        
        # Return data
        return jsonify({
            'status': 'success',
            'tsne_results': tsne_results,
            'dataset': dataset_type,
            'sample_size': len(tsne_results),
            'model_id': model_id or 'raw_pixels'
        })
        
    except Exception as e:
        logger.error(f"Error generating t-SNE visualization: {e}")
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/api/sample_image/<dataset>/<int:index>')
def api_sample_image(dataset, index):
    """Get a specific image from the dataset"""
    try:
        # Get dataset
        lazy_load_mnist()
        
        if dataset == 'train':
            x_data = x_train
        else:
            x_data = x_test
            
        # Check index
        if index < 0 or index >= len(x_data):
            return jsonify({'status': 'error', 'message': f'Invalid index {index}'})
            
        # Get the image
        image = x_data[index].reshape(28, 28) * 255
        
        # Convert to PNG image
        buffer = BytesIO()
        Image.fromarray(image.astype('uint8')).save(buffer, format='PNG')
        buffer.seek(0)
        
        # Send the image
        return send_file(buffer, mimetype='image/png')
        
    except Exception as e:
        logger.error(f"Error getting sample image: {e}")
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/api/models')
def api_models():
    """API endpoint to get the list of all models"""
    try:
        logger.info("接收到获取模型列表的请求")
        registry = load_model_registry()
        
        if not registry:
            logger.info("没有找到可用的模型")
            return jsonify({'status': 'success', 'message': 'No models available', 'models': []})
        
        logger.info(f"找到 {len(registry)} 个模型，准备返回")
        
        # 将registry字典转换为列表
        models_list = [model_info for model_info in registry.values()]
        
        # 按创建时间排序，最新的排在前面
        models_list.sort(key=lambda x: x.get('created_at', ''), reverse=True)
        
        response = {
            'status': 'success',
            'models': models_list
        }
        logger.info(f"返回 {len(models_list)} 个模型")
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Error retrieving models: {e}")
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/api/models/latest')
def api_latest_model():
    """API endpoint to get the latest model"""
    try:
        logger.info("接收到获取最新模型的请求")
        registry = load_model_registry()
        
        if not registry:
            logger.info("没有找到可用的模型")
            return jsonify({'status': 'error', 'message': 'No models available'})
        
        # 将所有模型转换为列表并按创建时间排序
        models_list = [model_info for model_info in registry.values()]
        models_list.sort(key=lambda x: x.get('created_at', ''), reverse=True)
        
        # 获取最新模型
        if models_list:
            latest_model = models_list[0]
            logger.info(f"返回最新模型: {latest_model.get('id')}")
            return jsonify({
                'status': 'success',
                'model': latest_model
            })
        else:
            logger.info("没有找到可用的模型")
            return jsonify({'status': 'error', 'message': 'No models available'})
            
    except Exception as e:
        logger.error(f"Error retrieving latest model: {e}")
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/api/models/best')
def api_best_model():
    """API endpoint to get the model with the best accuracy"""
    try:
        logger.info("接收到获取最佳精度模型的请求")
        registry = load_model_registry()
        
        if not registry:
            logger.info("没有找到可用的模型")
            return jsonify({'status': 'error', 'message': 'No models available'})
        
        # 找出精度最高的模型
        best_model_id = max(registry.items(), key=lambda x: x[1]['performance']['accuracy'])[0]
        best_model = registry[best_model_id]
        logger.info(f"返回最佳精度模型: {best_model_id}, 精度: {best_model['performance']['accuracy']}")
        
        return jsonify({
            'status': 'success',
            'model': best_model
        })
            
    except Exception as e:
        logger.error(f"Error retrieving best model: {e}")
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/api/models/<model_id>/metrics', methods=['GET'])
def get_model_metrics(model_id):
    """获取模型的训练指标历史数据"""
    try:
        # 检查模型是否存在
        if model_id not in model_registry:
            return jsonify({
                'status': 'error',
                'message': f'Model with ID {model_id} not found'
            }), 404
        
        model_info = model_registry[model_id]
        
        # 如果模型没有训练历史，返回空数据
        if 'history' not in model_info or not model_info['history']:
            return jsonify({
                'status': 'success',
                'epochs': [],
                'train_accuracy': [],
                'val_accuracy': [],
                'train_loss': [],
                'val_loss': []
            })
        
        # 提取训练历史数据
        history = model_info['history']
        epochs = list(range(1, len(history['accuracy']) + 1))
        
        return jsonify({
            'status': 'success',
            'epochs': epochs,
            'train_accuracy': history['accuracy'],
            'val_accuracy': history['val_accuracy'],
            'train_loss': history['loss'],
            'val_loss': history['val_loss']
        })
    
    except Exception as e:
        app.logger.error(f"Error getting model metrics: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': f'Failed to get model metrics: {str(e)}'
        }), 500

@app.route('/api/models/<model_id>/versions', methods=['GET'])
def get_model_versions(model_id):
    """获取模型的版本历史"""
    try:
        # 检查模型是否存在
        if model_id not in model_registry:
            return jsonify({
                'status': 'error',
                'message': f'Model with ID {model_id} not found'
            }), 404
        
        # 从版本历史存储中获取数据（实际应用中这可能是从数据库获取）
        # 在此示例中，我们返回模拟数据
        versions = [
            {
                'version': 1,
                'created_at': (datetime.now() - timedelta(days=10)).isoformat(),
                'created_by': 'system',
                'accuracy': 0.92,
                'changes': 'Initial training'
            },
            {
                'version': 2,
                'created_at': (datetime.now() - timedelta(days=5)).isoformat(),
                'created_by': 'user@example.com',
                'accuracy': 0.95,
                'changes': 'Fine-tuned with additional data'
            },
            {
                'version': 3,
                'created_at': datetime.now().isoformat(),
                'created_by': 'user@example.com',
                'accuracy': 0.97,
                'changes': 'Hyperparameter optimization'
            }
        ]
        
        return jsonify(versions)
    
    except Exception as e:
        app.logger.error(f"Error getting model versions: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': f'Failed to get model versions: {str(e)}'
        }), 500

@app.route('/api/models/<model_id>/versions/<int:version>', methods=['DELETE'])
def delete_model_version(model_id, version):
    """删除模型的特定版本"""
    try:
        # 检查模型是否存在
        if model_id not in model_registry:
            return jsonify({
                'status': 'error',
                'message': f'Model with ID {model_id} not found'
            }), 404
        
        # 检查版本是否存在（实际应用中这可能涉及数据库操作）
        # 这里我们简单返回成功，表示版本已删除
        return jsonify({
            'status': 'success',
            'message': f'Version {version} of model {model_id} deleted successfully'
        })
    
    except Exception as e:
        app.logger.error(f"Error deleting model version: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': f'Failed to delete model version: {str(e)}'
        }), 500

@app.route('/api/models/<model_id>/versions/<int:version>/restore', methods=['POST'])
def restore_model_version(model_id, version):
    """恢复模型到特定版本"""
    try:
        # 检查模型是否存在
        if model_id not in model_registry:
            return jsonify({
                'status': 'error',
                'message': f'Model with ID {model_id} not found'
            }), 404
        
        # 检查版本是否存在并进行恢复操作（实际应用中这可能涉及数据库操作）
        # 这里我们简单返回成功，表示版本已恢复
        return jsonify({
            'status': 'success',
            'message': f'Model {model_id} restored to version {version} successfully'
        })
    
    except Exception as e:
        app.logger.error(f"Error restoring model version: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': f'Failed to restore model version: {str(e)}'
        }), 500

@app.route('/api/models/<model_id>/export', methods=['POST'])
def export_model(model_id):
    """导出模型到指定格式"""
    try:
        # 检查模型是否存在
        if model_id not in model_registry:
            return jsonify({
                'status': 'error',
                'message': f'Model with ID {model_id} not found'
            }), 404
        
        model_info = model_registry[model_id]
        
        # 获取请求参数
        data = request.get_json() or {}
        export_format = data.get('format', 'pickle').lower()
        include_history = data.get('include_history', False)
        include_metadata = data.get('include_metadata', False)
        
        # 加载模型对象
        model_path = model_info['best_model_path']
        model = load_specific_model(model_path)
        if not model:
            return jsonify({
                'status': 'error',
                'message': f'Failed to load model from {model_path}'
            }), 500
        
        # 根据请求的格式导出模型
        if export_format == 'pickle':
            # 导出为pickle格式
            export_data = pickle.dumps(model)
            mimetype = 'application/octet-stream'
            filename = f'model_{model_id}.pkl'
        
        elif export_format == 'onnx':
            # 模拟ONNX导出（实际应用中应使用适当的ONNX导出逻辑）
            export_data = b'ONNX MODEL BINARY DATA'
            mimetype = 'application/octet-stream'
            filename = f'model_{model_id}.onnx'
        
        elif export_format == 'json':
            # 导出为JSON格式
            model_data = {
                'id': model_id,
                'architecture': model_info.get('architecture', 'Neural Network'),
                'hyperparameters': model_info.get('parameters', {}),
                'weights': [w.tolist() for w in model.coefs_],
                'biases': [b.tolist() for b in model.intercepts_]
            }
            
            if include_history and 'history' in model_info:
                model_data['training_history'] = model_info['history']
            
            if include_metadata:
                model_data['metadata'] = {
                    'created_at': model_info.get('timestamp', datetime.now().isoformat()),
                    'accuracy': model_info.get('best_accuracy', 0),
                    'loss': model_info.get('loss', 0),
                    'export_timestamp': datetime.now().isoformat()
                }
            
            export_data = json.dumps(model_data).encode('utf-8')
            mimetype = 'application/json'
            filename = f'model_{model_id}.json'
        
        else:
            return jsonify({
                'status': 'error',
                'message': f'Unsupported export format: {export_format}'
            }), 400
        
        # 返回文件作为响应
        response = make_response(export_data)
        response.headers.set('Content-Type', mimetype)
        response.headers.set('Content-Disposition', f'attachment; filename={filename}')
        
        return response
    
    except Exception as e:
        app.logger.error(f"Error exporting model: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': f'Failed to export model: {str(e)}'
        }), 500

@app.route('/api/visualize/model_weights', methods=['POST'])
def api_visualize_model_weights():
    """获取模型层的权重和参数可视化数据"""
    try:
        # 获取请求参数
        data = request.json
        model_id = data.get('model_id')
        
        # 加载模型
        if model_id:
            if model_id not in model_registry:
                return jsonify({'status': 'error', 'message': f'Model {model_id} not found'})
            model_path = model_registry[model_id]['best_model_path']
            model = load_specific_model(model_path)
        else:
            if current_model is None:
                return jsonify({'status': 'error', 'message': 'No current model available'})
            model = current_model
        
        # 获取模型架构信息
        architecture = {
            'input_size': model.input_size,
            'hidden_size': model.hidden_size,
            'output_size': model.output_size,
            'activation': model.activation
        }
        
        # 获取模型权重和偏置
        weights = {
            'layer1': {
                'weights': model.W1.tolist(),
                'bias': model.b1.tolist(),
                'shape': model.W1.shape
            },
            'layer2': {
                'weights': model.W2.tolist(),
                'bias': model.b2.tolist(),
                'shape': model.W2.shape
            }
        }
        
        # 计算权重统计信息
        stats = {
            'layer1': {
                'min': float(np.min(model.W1)),
                'max': float(np.max(model.W1)),
                'mean': float(np.mean(model.W1)),
                'std': float(np.std(model.W1))
            },
            'layer2': {
                'min': float(np.min(model.W2)),
                'max': float(np.max(model.W2)),
                'mean': float(np.mean(model.W2)),
                'std': float(np.std(model.W2))
            }
        }
        
        # 返回数据
        return jsonify({
            'status': 'success',
            'architecture': architecture,
            'weights': weights,
            'stats': stats,
            'model_id': model_id or 'current'
        })
        
    except Exception as e:
        logger.error(f"Error generating model weights visualization: {e}")
        return jsonify({'status': 'error', 'message': str(e)})

@app.errorhandler(HTTPException)
def handle_exception(e):
    """Handle HTTP exceptions"""
    response = e.get_response()
    response.data = json.dumps({
        "code": e.code,
        "name": e.name,
        "description": e.description,
    })
    response.content_type = "application/json"
    return response

@app.errorhandler(Exception)
def handle_error(e):
    """Handle all other exceptions"""
    logger.error(f"Unhandled exception: {e}")
    return jsonify({
        "status": "error",
        "message": str(e)
    }), 500

if __name__ == '__main__':
    # Load model registry
    load_model_registry()
    
    # Try to load current model
    model_path = os.path.join(os.path.dirname(__file__), 'static', 'models', 'current_model.pkl')
    if os.path.exists(model_path):
        try:
            current_model = SimpleNeuralNetwork.load(model_path)
            logger.info("Loaded current model successfully")
        except Exception as e:
            logger.error(f"Failed to load current model: {e}")
    
    # 确保必要的目录存在
    models_dir = os.path.join(os.path.dirname(__file__), 'static', 'models')
    debug_dir = os.path.join(os.path.dirname(__file__), 'static', 'debug')
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(debug_dir, exist_ok=True)
    
    # 从环境变量中获取端口，Render 会设置 PORT 环境变量
    port = int(os.environ.get('PORT', 8080))
    print(f"Starting app with host='0.0.0.0', port={port}")
    
    # 设置服务器选项，确保正确绑定到 IP 和端口
    app.run(host='0.0.0.0', port=port, debug=False, threaded=True) 
#!/usr/bin/env python
# 轻量级MNIST分类器 - 使用简单的NumPy实现
import numpy as np
import urllib.request
import os
import gzip
import time
import pickle
from os.path import exists

# MNIST数据集URL - 使用备用镜像
urls = [
    'https://storage.googleapis.com/cvdf-datasets/mnist/train-images-idx3-ubyte.gz',
    'https://storage.googleapis.com/cvdf-datasets/mnist/train-labels-idx1-ubyte.gz',
    'https://storage.googleapis.com/cvdf-datasets/mnist/t10k-images-idx3-ubyte.gz',
    'https://storage.googleapis.com/cvdf-datasets/mnist/t10k-labels-idx1-ubyte.gz'
]

# 设置随机种子以确保可重复性
np.random.seed(42)

def download_mnist():
    """下载MNIST数据集"""
    print("下载MNIST数据集...")
    os.makedirs("data", exist_ok=True)
    for url in urls:
        filename = os.path.join("data", url.split('/')[-1])
        if not exists(filename):
            print(f"正在下载 {url}")
            try:
                urllib.request.urlretrieve(url, filename)
                print(f"下载完成: {filename}")
            except Exception as e:
                print(f"下载失败: {e}")
                return False
    print("所有文件下载完成")
    return True

def load_mnist():
    """加载MNIST数据集"""
    print("加载MNIST数据集...")
    
    # 检查是否已有处理好的数据
    if exists('data/mnist.pkl'):
        with open('data/mnist.pkl', 'rb') as f:
            return pickle.load(f)
    
    # 确保数据已下载
    if not download_mnist():
        raise ValueError("无法下载MNIST数据集。请检查网络连接后重试。")
    
    try:
        # 解析数据
        with gzip.open('data/train-images-idx3-ubyte.gz', 'rb') as f:
            x_train = np.frombuffer(f.read(), np.uint8, offset=16).reshape(-1, 28*28)
        
        with gzip.open('data/train-labels-idx1-ubyte.gz', 'rb') as f:
            y_train = np.frombuffer(f.read(), np.uint8, offset=8)
        
        with gzip.open('data/t10k-images-idx3-ubyte.gz', 'rb') as f:
            x_test = np.frombuffer(f.read(), np.uint8, offset=16).reshape(-1, 28*28)
        
        with gzip.open('data/t10k-labels-idx1-ubyte.gz', 'rb') as f:
            y_test = np.frombuffer(f.read(), np.uint8, offset=8)
        
        # 归一化图像数据到[0,1]
        x_train = x_train / 255.0
        x_test = x_test / 255.0
        
        # 将数据存储到pkl文件以便未来快速加载
        mnist_data = (x_train, y_train, x_test, y_test)
        with open('data/mnist.pkl', 'wb') as f:
            pickle.dump(mnist_data, f)
        
        return mnist_data
    except Exception as e:
        print(f"处理MNIST数据时出错: {e}")
        raise

def one_hot_encode(y, num_classes=10):
    """将标签转换为one-hot编码"""
    return np.eye(num_classes)[y]

def sigmoid(x):
    """Sigmoid激活函数"""
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))  # 使用clip防止上溢/下溢

def softmax(x):
    """Softmax激活函数"""
    # 为了数值稳定性，减去每行的最大值
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

class SimpleNeuralNetwork:
    """简单的两层神经网络"""
    
    def __init__(self, input_size, hidden_size, output_size, activation='sigmoid'):
        """初始化网络参数"""
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.activation = activation
        
        # 使用Xavier初始化权重
        self.W1 = np.random.randn(input_size, hidden_size) * np.sqrt(1.0 / input_size)
        self.b1 = np.zeros(hidden_size)
        self.W2 = np.random.randn(hidden_size, output_size) * np.sqrt(1.0 / hidden_size)
        self.b2 = np.zeros(output_size)
    
    def _apply_activation(self, x):
        """应用激活函数"""
        if self.activation == 'sigmoid':
            return sigmoid(x)
        elif self.activation == 'relu':
            return np.maximum(0, x)
        else:
            # 默认使用sigmoid
            return sigmoid(x)
    
    def _activation_derivative(self, x):
        """激活函数的导数"""
        if self.activation == 'sigmoid':
            s = sigmoid(x)
            return s * (1 - s)
        elif self.activation == 'relu':
            return (x > 0).astype(float)
        else:
            # 默认使用sigmoid
            s = sigmoid(x)
            return s * (1 - s)
    
    def forward(self, X):
        """前向传播"""
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = self._apply_activation(self.z1)
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = softmax(self.z2)
        return self.a2
    
    def get_hidden_features(self, X):
        """获取隐藏层的特征表示，用于t-SNE可视化"""
        z1 = np.dot(X, self.W1) + self.b1
        a1 = self._apply_activation(z1)
        return a1
    
    def backward(self, X, y, output, learning_rate=0.01):
        """反向传播更新权重"""
        batch_size = X.shape[0]
        
        # 计算输出层的误差
        delta2 = output - y
        
        # 计算隐藏层的误差
        delta1 = np.dot(delta2, self.W2.T) * self._activation_derivative(self.z1)
        
        # 更新权重和偏置
        self.W2 -= learning_rate * np.dot(self.a1.T, delta2) / batch_size
        self.b2 -= learning_rate * np.sum(delta2, axis=0) / batch_size
        self.W1 -= learning_rate * np.dot(X.T, delta1) / batch_size
        self.b1 -= learning_rate * np.sum(delta1, axis=0) / batch_size
    
    def calculate_loss(self, output, y):
        """计算交叉熵损失"""
        # 添加小值epsilon避免log(0)
        epsilon = 1e-15
        output = np.clip(output, epsilon, 1 - epsilon)
        return -np.sum(y * np.log(output)) / output.shape[0]
    
    def train(self, X, y, X_val, y_val, epochs=10, batch_size=32, learning_rate=0.01):
        """训练网络"""
        num_samples = X.shape[0]
        num_batches = num_samples // batch_size
        
        # 将标签转换为one-hot编码
        y_one_hot = one_hot_encode(y)
        y_val_one_hot = one_hot_encode(y_val)
        
        for epoch in range(epochs):
            start_time = time.time()
            
            # 随机打乱训练数据
            indices = np.random.permutation(num_samples)
            X_shuffled = X[indices]
            y_shuffled = y_one_hot[indices]
            
            # 小批量梯度下降
            for i in range(num_batches):
                start_idx = i * batch_size
                end_idx = min((i + 1) * batch_size, num_samples)
                
                X_batch = X_shuffled[start_idx:end_idx]
                y_batch = y_shuffled[start_idx:end_idx]
                
                # 前向传播
                output = self.forward(X_batch)
                
                # 反向传播
                self.backward(X_batch, y_batch, output, learning_rate)
            
            # 计算训练集准确率
            train_predictions = np.argmax(self.forward(X), axis=1)
            train_accuracy = np.mean(train_predictions == y)
            
            # 计算验证集准确率
            val_predictions = np.argmax(self.forward(X_val), axis=1)
            val_accuracy = np.mean(val_predictions == y_val)
            
            epoch_time = time.time() - start_time
            print(f"Epoch {epoch+1}/{epochs}, 时间: {epoch_time:.2f}s, 训练集准确率: {train_accuracy:.4f}, 验证集准确率: {val_accuracy:.4f}")
    
    def evaluate(self, X, y):
        """评估模型"""
        predictions = np.argmax(self.forward(X), axis=1)
        accuracy = np.mean(predictions == y)
        return accuracy
    
    def predict(self, X):
        """预测类别"""
        return np.argmax(self.forward(X), axis=1)
    
    def copy(self):
        """创建模型的拷贝"""
        model_copy = SimpleNeuralNetwork(self.input_size, self.hidden_size, self.output_size, self.activation)
        model_copy.W1 = np.copy(self.W1)
        model_copy.b1 = np.copy(self.b1)
        model_copy.W2 = np.copy(self.W2)
        model_copy.b2 = np.copy(self.b2)
        return model_copy
    
    def save(self, filename):
        """保存模型参数"""
        model_params = {
            'W1': self.W1,
            'b1': self.b1,
            'W2': self.W2,
            'b2': self.b2,
            'architecture': {
                'input_size': self.input_size,
                'hidden_size': self.hidden_size,
                'output_size': self.output_size,
                'activation': self.activation
            }
        }
        
        with open(filename, 'wb') as f:
            pickle.dump(model_params, f)
    
    @classmethod
    def load(cls, filename):
        """从文件加载模型"""
        with open(filename, 'rb') as f:
            model_params = pickle.load(f)
        
        architecture = model_params['architecture']
        activation = architecture.get('activation', 'sigmoid')  # 向后兼容
        
        model = cls(
            architecture['input_size'],
            architecture['hidden_size'],
            architecture['output_size'],
            activation=activation
        )
        
        model.W1 = model_params['W1']
        model.b1 = model_params['b1']
        model.W2 = model_params['W2']
        model.b2 = model_params['b2']
        
        return model

def main():
    """主函数"""
    # 加载数据
    x_train, y_train, x_test, y_test = load_mnist()
    
    print(f"训练集形状: {x_train.shape}")
    print(f"测试集形状: {x_test.shape}")
    
    # 使用较小的训练子集
    train_size = 10000  # 使用10000个样本进行训练，适合低性能MAC
    x_train_subset = x_train[:train_size]
    y_train_subset = y_train[:train_size]
    
    # 划分验证集
    val_size = 1000
    x_val = x_train[train_size:train_size+val_size]
    y_val = y_train[train_size:train_size+val_size]
    
    print(f"使用的训练子集形状: {x_train_subset.shape}")
    print(f"验证集形状: {x_val.shape}")
    
    # 初始化模型
    input_size = 28 * 28  # MNIST图像大小
    hidden_size = 128     # 隐藏层神经元数量
    output_size = 10      # 输出类别数量 (0-9)
    
    print("创建模型...")
    model = SimpleNeuralNetwork(input_size, hidden_size, output_size)
    
    # 训练模型
    print("开始训练...")
    model.train(
        x_train_subset, y_train_subset,
        x_val, y_val,
        epochs=5,         # 适合低性能MAC的周期数
        batch_size=64,    # 较小的批量大小
        learning_rate=0.1 # 学习率
    )
    
    # 在测试集上评估模型
    test_accuracy = model.evaluate(x_test, y_test)
    print(f"测试集准确率: {test_accuracy:.4f}")
    
    # 保存模型
    model.save('mnist_model.pkl')
    print("模型已保存到 'mnist_model.pkl'")
    
    # 展示一些预测结果
    num_examples = 5
    predictions = model.predict(x_test[:num_examples])
    
    print("\n前几个测试样本的预测结果:")
    for i in range(num_examples):
        print(f"样本 {i+1}: 预测 = {predictions[i]}, 实际 = {y_test[i]}")

if __name__ == "__main__":
    main()

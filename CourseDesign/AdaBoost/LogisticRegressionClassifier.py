from .BaseModel import BaseWeakLearner
import numpy as np

class LogisticRegressionClassifier(BaseWeakLearner):
    """
    逻辑回归基学习器
    """

    def __init__(self, learning_rate=0.01, n_iterations=5000, tol=1e-8):
        super().__init__()
        self.learning_rate = learning_rate  # 学习率
        self.n_iterations = n_iterations    # 最大迭代次数
        self.tol = tol                      # 收敛阈值
        self.weights = None                 # 权重
        self.bias = None                    # 偏置项
        self.alpha = None                   # 该分类器在AdaBoost集成中的权重
        self.min_error = None               # 最小的加权错误率
        self.threshold = 0.5                # 预测阈值


    def _sigmoid(self, z):
        """
        Sigmoid函数，用于将线性组合映射到0和1之间

        :param z: 输入值
        :return:
        """
        # 使用截断避免溢出
        z = np.clip(z, -500, 500)
        return 1/(1 + np.exp(-z))

    def fit(self, X, y, weights):
        """
        训练函数

        :param X: 训练集特征
        :param y: 训练集标签，值应为+1或-1
        :param weights: 样本权重
        :return: self
        """
        if not isinstance(X, np.ndarray):
            X = np.array(X)
        if not isinstance(y, np.ndarray):
            y = np.array(y)
        if not isinstance(weights, np.ndarray):
            weights = np.array(weights)

        # 检查标签是否合法
        y = self._label_correct(y)
        y_binary = (y + 1) / 2

        n_samples, n_features = X.shape

        # 初始化权重和偏置
        self.weights = np.zeros(n_features)
        self.bias = 0

        # 梯度下降法
        best_error = float('inf')
        best_weights = None
        best_bias = None
        best_threshold = 0.5

        for _ in range(self.n_iterations):
            # 计算预测值
            pred = np.dot(X, self.weights) + self.bias
            y_pred = self._sigmoid(pred)

            # 计算加权梯度
            dw = np.dot(X.T, (y_pred - y_binary) * weights) / np.sum(weights)
            db = np.sum((y_pred - y_binary) * weights) / np.sum(weights)

            # 更新权重和偏置
            w_previous = self.weights.copy()
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

            # 尝试不同的阈值
            for threshold in np.linspace(0.3, 0.7, 5):
                y_pred_labels = np.where(y_pred >= threshold, 1, -1)
                err = np.sum(weights[y_pred_labels != y])
                
                if err < best_error:
                    best_error = err
                    best_weights = self.weights.copy()
                    best_bias = self.bias
                    best_threshold = threshold

            # 检查收敛
            if np.mean(np.abs(self.weights - w_previous)) < self.tol:
                break

        # 保存最佳参数
        self.weights = best_weights
        self.bias = best_bias
        self.threshold = best_threshold
        self.min_error = best_error

        return self

    def predict(self, X):
        """
        预测函数

        :param X: 预测集特征
        :return: 预测标签，值为+1或-1
        """
        pred = np.dot(X, self.weights) + self.bias
        y_pred = self._sigmoid(pred)

        # 将概率转换为二分类标签(+1/-1)
        return np.where(y_pred >= self.threshold, 1, -1)



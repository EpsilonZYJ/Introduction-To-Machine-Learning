from Utils import accuracy, load_feature_data, load_label_data
import numpy as np

class BaseWeakLearner:
    """弱学习器的基类，定义基本接口"""
    def fit(self, X, y, weights):
        """使用加权样本进行训练"""
        pass

    def predict(self, X):
        """对样本进行预测"""
        pass


class DecisionStump(BaseWeakLearner):
    """决策树桩实现 - 一个简单的单层决策树"""

    def __init__(self):
        self.feature_idx = None  # 用于分类的特征索引
        self.threshold = None    # 分类阈值
        self.polarity = 1        # 极性，决定符号方向（+1 或 -1）
        self.alpha = None        # 该分类器在集成中的权重

    def fit(self, X, y, weights):
        """
        训练决策树桩，在加权数据上找到最佳分割点

        参数:
            X: 形状为[n_samples, n_features]的训练特征
            y: 形状为[n_samples]的训练标签，值为+1或-1
            weights: 样本权重

        返回:
            min_error: 最小的加权错误率
        """
        n_samples, n_features = X.shape
        min_error = float('inf')

        # 对每个特征尝试找到最佳的阈值和极性
        for feature_idx in range(n_features):
            feature_values = X[:, feature_idx]
            thresholds = np.unique(feature_values)

            # 对每个可能的阈值进行评估
            for threshold in thresholds:
                # 尝试两种极性
                for polarity in [1, -1]:
                    # 预测结果
                    predictions = np.ones(n_samples)
                    if polarity == 1:
                        predictions[feature_values < threshold] = -1
                    else:
                        predictions[feature_values >= threshold] = -1

                    # 计算加权错误率
                    misclassified = predictions != y
                    error = np.sum(weights[misclassified])

                    # 更新找到更好的参数组合
                    if error < min_error:
                        min_error = error
                        self.feature_idx = feature_idx
                        self.threshold = threshold
                        self.polarity = polarity

        return min_error

    def predict(self, X):
        """根据选定的特征和阈值预测样本类别"""
        n_samples = X.shape[0]
        feature_values = X[:, self.feature_idx]

        # 初始化预测结果
        predictions = np.ones(n_samples)

        # 应用阈值和极性进行分类
        if self.polarity == 1:
            predictions[feature_values < self.threshold] = -1
        else:
            predictions[feature_values >= self.threshold] = -1

        return predictions


class LogisticRegression(BaseWeakLearner):
    """逻辑回归作为基学习器的实现"""

    def __init__(self, learning_rate=0.1, n_iterations=1000, tol=1e-10):
        """
        初始化逻辑回归分类器

        参数:
            learning_rate: 梯度下降的学习率
            n_iterations: 最大迭代次数
            tol: 收敛阈值，当参数变化小于此值时停止迭代
        """
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.tol = tol
        self.weights = None
        self.bias = None
        self.alpha = None  # 该分类器在AdaBoost集成中的权重

    def _sigmoid(self, z):
        """Sigmoid激活函数"""
        # 使用截断来避免溢出
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y, weights):
        """
        使用加权样本训练逻辑回归模型

        参数:
            X: 形状为[n_samples, n_features]的训练特征
            y: 形状为[n_samples]的训练标签，值为+1或-1
            weights: 样本权重

        返回:
            error: 加权分类错误率
        """
        n_samples, n_features = X.shape

        # 将y转换为0/1标签用于逻辑回归
        y_binary = (y + 1) / 2

        # 初始化模型参数
        self.weights = np.zeros(n_features)
        self.bias = 0

        # 梯度下降优化
        for i in range(self.n_iterations):
            # 计算线性预测
            linear_pred = np.dot(X, self.weights) + self.bias
            # 计算sigmoid激活后的预测
            y_pred = self._sigmoid(linear_pred)

            # 计算加权梯度
            dw = (1 / n_samples) * np.dot(X.T, (y_pred - y_binary) * weights)
            db = (1 / n_samples) * np.sum((y_pred - y_binary) * weights)

            # 更新参数
            w_prev = self.weights.copy()
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

            # 检查收敛
            if np.mean(np.abs(self.weights - w_prev)) < self.tol:
                break

        # 计算加权错误率
        y_pred_labels = self.predict(X)
        misclassified = y_pred_labels != y
        error = np.sum(weights[misclassified])

        return error

    def predict(self, X):
        """
        预测样本类别

        参数:
            X: 形状为[n_samples, n_features]的测试特征

        返回:
            预测类别，值为+1或-1
        """
        linear_pred = np.dot(X, self.weights) + self.bias
        y_pred = self._sigmoid(linear_pred)

        # 将概率转换为二分类标签(+1/-1)
        return np.where(y_pred >= 0.5, 1, -1)


class MyAdaBoost:
    """AdaBoost分类器实现"""

    def __init__(self, base_learner_class=DecisionStump, n_estimators=50, **base_params):
        """
        初始化AdaBoost分类器

        参数:
            base_learner_class: 基学习器类（默认为决策树桩）
            n_estimators: 弱分类器的数量
            base_params: 传递给基学习器的参数
        """
        self.base_learner_class = base_learner_class
        self.n_estimators = n_estimators
        self.base_params = base_params
        self.estimators = []

    def fit(self, X, y):
        """
        训练AdaBoost分类器

        参数:
            X: 形状为[n_samples, n_features]的训练特征
            y: 形状为[n_samples]的训练标签，值为+1或-1

        返回:
            self: 训练后的分类器
        """
        n_samples = X.shape[0]

        # 初始化样本权重为均匀分布
        weights = np.ones(n_samples) / n_samples

        # 逐步训练弱分类器
        for _ in range(self.n_estimators):
            # 创建并训练基学习器
            learner = self.base_learner_class(**self.base_params)
            min_error = learner.fit(X, y, weights)

            # 防止除零错误或错误率太高的情况
            epsilon = 1e-10
            if min_error >= 0.5:
                # 如果错误率大于0.5，就放弃这个分类器
                continue

            # 计算弱分类器权重
            alpha = 0.5 * np.log((1.0 - min_error + epsilon) / (min_error + epsilon))
            learner.alpha = alpha

            # 获取当前弱分类器的预测
            predictions = learner.predict(X)

            # 更新样本权重
            weights *= np.exp(-alpha * y * predictions)
            # 归一化权重
            weights /= np.sum(weights)

            # 保存弱分类器
            self.estimators.append(learner)

            # 如果错误率为0，提前结束训练
            if min_error == 0:
                break

        return self

    def predict(self, X):
        """
        使用AdaBoost集成进行预测

        参数:
            X: 形状为[n_samples, n_features]的测试特征

        返回:
            预测的类别标签，值为+1或-1
        """
        n_samples = X.shape[0]
        y_pred = np.zeros(n_samples)

        # 确保至少有一个基学习器
        if not self.estimators:
            return np.ones(n_samples)

        # 计算所有弱分类器的加权投票
        for learner in self.estimators:
            y_pred += learner.alpha * learner.predict(X)

        # 返回集成结果的符号
        return np.sign(y_pred)

    def score(self, X, y):
        """
        计算分类准确率

        参数:
            X: 形状为[n_samples, n_features]的测试特征
            y: 形状为[n_samples]的真实标签

        返回:
            分类准确率（0到1之间）
        """
        y_pred = self.predict(X)
        return np.mean(y_pred == y)

train_feature = load_feature_data('data.csv')
train_label = load_label_data('targets.csv')

model = MyAdaBoost(n_estimators=1)
train_label =np.array([1 if label == 1 else -1 for label in train_label])
model.fit(train_feature, train_label)
y_pred = model.predict(train_feature)
print("Predictions:", y_pred)
print("Accuracy:", model.score(train_feature, train_label))
print("Acc:", accuracy(train_label, y_pred))

model = MyAdaBoost(base_learner_class=LogisticRegression, n_estimators=10, learning_rate=0.1, n_iterations=10000,tol=1e-12)
train_label =np.array([1 if label == 1 else -1 for label in train_label])
model.fit(train_feature, train_label)
y_pred = model.predict(train_feature)
print("Predictions:", y_pred)
print("Accuracy:", model.score(train_feature, train_label))
print("Acc:", accuracy(train_label, y_pred))
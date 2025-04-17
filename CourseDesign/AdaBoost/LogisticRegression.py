import numpy as np

class LogisticRegression:
    def __init__(self, learning_rate=0.05, n_iters: int=1000, epsilon=1e-8):
        """
        :param learning_rate: 学习率
        :param n_iters: 最大迭代次数
        :param epsilon: 容忍误差范围

        lr: 学习率
        its: 最大迭代次数
        epsilon: 容忍误差范围
        w: 权重
        isTrain: 是否是训练状态
        """
        self.lr = learning_rate
        self.its = n_iters
        self.epsilon = epsilon
        self.w = None
        self.isTrain = False

    def _feature_extend(self, x):
        """
        原函数为z=b+wx，将特征进行拓展为z'=w'x'，
        其中x'前面的列与x相同，第一列为1
        :param x: 原始特征
        :return: 返回第一列补上1后的特征
        """
        if not isinstance(x, np.ndarray):
            x = np.array(x)
        if len(x.shape) == 2:
            x_new = np.zeros((x.shape[0], x.shape[1] + 1))
            x_new[:, 1:] = x
            x_new[:, 0] = 1
        elif len(x.shape) == 1:
            x_new = np.zeros(x.shape[0] + 1)
            x_new[1:] = x
            x_new[0] = 1
        else:
            raise ValueError("不支持的维度空间")
        return x_new

    def _sigmoid(self, x):
        """
        sigmoid函数
        :param x: 输入值，
        :return: sigmoid后的概率值
        """
        return 1/(1 + np.exp(-x))

    def _predict_single(self, x):
        """
        预测单个样本
        :param x: 输入的单个样本的特征值
        :return: 预测值
        """
        if not self.isTrain:
            x = self._feature_extend(x)
        z = np.dot(x, self.w)
        return self._sigmoid(z)

    def _predict_probability(self, X):
        """
        预测概率值，用于训练时的预测
        :param X: 特征值
        :return: 预测值
        """
        if not isinstance(X, np.ndarray):
            X = np.array(X)
        y_predict = np.zeros(X.shape[0])
        for i in range(X.shape[0]):
            y_predict[i] = self._predict_single(X[i])
        return y_predict

    def _CrossEntropyLoss(self, y, y_hat, epsilon=1e-12):
        """
        交叉熵损失函数
        :param y: 真实值
        :param y_hat: 预测值
        :return: 损失值
        """
        # TODO: 看后期是否可以加上L2正则化
        if not isinstance(y, np.ndarray):
            y = np.array(y)
        if not isinstance(y_hat, np.ndarray):
            y_hat = np.array(y_hat)
        y_hat = np.clip(y_hat, epsilon, 1.0 - epsilon)  # 防止log(0)
        loss = -np.sum(y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat)) / y.shape[0]
        return loss

    def predict(self, X):
        """
        预测函数
        :param X: 测试数据集
        :return: 预测值
        """
        if not isinstance(X, np.ndarray):
            X = np.array(X)
        y_predict = np.zeros(X.shape[0])
        for i in range(X.shape[0]):
            y_predict[i] = 1 if self._predict_single(X[i]) >= 0.5 else 0
        return y_predict

    def fit(self, X, y, batch_size=32):
        """
        使用批量随机梯度下降训练逻辑回归函数
        :param X: 训练集特征值
        :param y: 训练集标签值
        :param batch_size: 每个小批量的大小
        :return:
        """
        self.isTrain = True
        if not isinstance(X, np.ndarray):
            X = np.array(X)
        if not isinstance(y, np.ndarray):
            y = np.array(y)

        # 特征拓展
        X = self._feature_extend(X)

        # 初始化权重
        self.w = np.zeros(X.shape[1])

        # 获取样本数量
        n_samples = X.shape[0]

        # 迭代训练
        for iter in range(self.its):
            indices = np.random.permutation(n_samples)
            X_shuffled = X[indices]
            y_shuffled = y[indices]

            # 分批进行训练
            for start in range(0, n_samples, batch_size):
                end = min(start + batch_size, n_samples)
                X_batch = X_shuffled[start:end]
                y_batch = y_shuffled[start:end]

                # 预测值
                y_hat = self._predict_probability(X_batch)
                # 计算梯度
                gradient = np.dot(X_batch.T, (y_hat - y_batch)) / X_batch.shape[0]

                # 更新权重
                self.w -= self.lr * gradient

            y_hat = self._predict_probability(X)
            loss = self._CrossEntropyLoss(y, y_hat)
            if loss < self.epsilon:
                break
            print(f'iteration {iter} loss: {loss}')
        self.isTrain = False

import numpy as np

class LogisticRegression:
    def __init__(self, learning_rate=0.05, n_iters: int=1000, epsilon=1e-8):
        """
        :param learning_rate: 学习率
        :param n_iters: 最大迭代次数
        :param epsilon: 容忍误差范围
        """
        self.lr = learning_rate
        self.its = n_iters
        self.epsilon = epsilon
        self.w = None

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
        if not isinstance(x, np.ndarray):
            x = np.array(x)
        return 1/(1 + np.exp(-x))

    def _predict_single(self, x):
        """
        预测单个样本
        :param x: 输入的单个样本的特征值
        :return: 预测值
        """
        x = self._feature_extend(x)
        z = np.dot(x, self.w)
        return self._sigmoid(z)

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


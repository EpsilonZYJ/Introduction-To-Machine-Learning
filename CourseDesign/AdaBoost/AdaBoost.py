import copy
import numpy as np
from .LogisticRegression import LogisticRegression
from .DecisionStump import DecisionStump
from tqdm import trange

class AdaBoost:
    def __init__(self, n_estimators=100, base_estimator=DecisionStump(), learning_rate=1.0, n_iters=1000):
        """
        AdaBoost算法的初始化函数

        :param n_estimators: 基学习器的数量
        :param base_estimator: 基学习器的类型
        :param learning_rate: 学习率
        :param n_iters: 最大迭代次数

        self.n_estimators: 基学习器的数量
        self.estimators: 存储基学习器的列表
        self.lr: 学习率
        self.base_estimator: 基学习器
        self.alphas: 存储每个基学习器的权重
        self.n_iters: 最大迭代次数
        """
        self.n_estimators = n_estimators
        self.estimators = []
        self.lr = learning_rate
        self.base_estimator = base_estimator
        self.alphas = []
        self.n_iters = n_iters

    def _label_encode(self, y):
        """
        将标签编码，把标签由0和1转换为-1和1

        :param y: 原始标签
        :return: 转化后的标签
        """
        if not isinstance(y, np.ndarray):
            y = np.array(y)
        for i in range(y.shape[0]):
            if y[i] == 0:
                y[i] = -1
        return y

    def _label_decode(self, y):
        """
        标签解码，将标签由-1和1转换为0和1

        :param y: 标记为-1和1的标签
        :return: 标记为0和1的标签
        """
        if not isinstance(y, np.ndarray):
            y = np.array(y)
        for i in range(y.shape[0]):
            if y[i] == -1:
                y[i] = 0
        return y

    def _exponential_loss(self, y_true, y_pred):
        """
        计算指数损失函数

        :param y_true: 真实标签
        :param y_pred: 预测标签
        :return: 损失值
        """
        if not isinstance(y_true, np.ndarray):
            y_true = np.array(y_true)
        if not isinstance(y_pred, np.ndarray):
            y_pred = np.array(y_pred)
        return np.mean(np.exp(-y_true * y_pred))

    def fit(self, X, y):
        """
        AdaBoost算法的训练函数

        :param X: 训练集特征
        :param y: 训练集标签
        :return:
        """
        if not isinstance(X, np.ndarray):
            X = np.array(X)
        if not isinstance(y, np.ndarray):
            y = np.array(y)

        # 初始化样本权重
        sample_weights = np.ones(X.shape[0]) / X.shape[0]

        # 训练模型
        for t in trange(self.n_estimators):
            model = copy.deepcopy(self.base_estimator)

            # 在数据集上以权重sample_weights训练基学习器
            model.train(X, y, sample_weights)

            # 计算加权错误率
            y_pred = model.predict(X)
            err = np.sum(sample_weights * (y_pred != y)) / np.sum(sample_weights)

            # 若基分类器比随机猜测还差则终止算法
            if err >= 0.5:
                continue
            alpha = 1/2 * np.log((1 - err) / (err + 1e-12))

            # 更新样本权重
            sample_weights = sample_weights * np.exp(-alpha * (2 * y - 1) * (2 * y_pred - 1))

            # 归一化样本权重
            sample_weights /= np.sum(sample_weights)

            # 保存基学习器
            self.estimators.append(model)
            self.alphas.append(alpha)

    def predict(self, X, label_decode=True):
        """
        AdaBoost算法的预测函数

        :param X: 特征值
        :param label_decode: 是否需要标签解码
        :return: 预测的数组
        """
        if not isinstance(X, np.ndarray):
            X = np.array(X)
        y_pred = np.zeros(X.shape[0])
        for i in range(len(self.estimators)):
            model = self.estimators[i]
            alpha = self.alphas[i]
            y_pred += alpha * model.predict(X)
        y_pred = np.sign(y_pred)
        if not label_decode:
            return y_pred
        else:
            return self._label_decode(y_pred)

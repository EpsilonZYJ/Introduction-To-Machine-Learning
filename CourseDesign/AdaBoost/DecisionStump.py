import numpy as np

class DecisionStump:
    def __init__(self):
        """
        初始化决策树桩
        """
        self.tree = {}

    def fit(self, X, y):
        """
        训练决策树桩
        :param X: 训练集的输入特征
        :param y: 训练集的标签
        :return:
        """
        if not isinstance(X, np.ndarray):
            X = np.array(X)
        if not isinstance(y, np.ndarray):
            y = np.ndarray(y)


    def predict(self, X):
        """
        预测函数
        :param X: 测试数据集
        :return: 预测值
        """
        if not isinstance(X, np.ndarray):
            X = np.array(X)

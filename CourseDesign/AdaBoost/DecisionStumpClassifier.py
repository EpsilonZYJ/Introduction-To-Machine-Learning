import numpy as np
from .BaseModel import BaseWeakLearner

class DecisionStumpClassifier(BaseWeakLearner):
    """
    决策树桩基学习器
    """

    def __init__(self):
        super().__init__()
        self.feature_idx = None         # 用于分类的特征索引
        self.threshold = None           # 分类阈值
        self.polarity = 1               # 极性，决定符号方向（+1 或 -1），
                                        # +1 代表小于阈值的样本为-1，-1 代表大于等于阈值的样本为-1
        self.alpha = None               # 该分类器在集成中的权重
        self.min_error = float('inf')   # 最小的加权错误率




    def fit(self, X, y, weights):
        """
        决策树桩训练函数

        :param X: 训练特征
        :param y: 训练标签，值应为+1或-1
        :param weights: 样本权重
        :return: self
        """
        # 转化为numpy数组
        if not isinstance(X, np.ndarray):
            X = np.array(X)
        if not isinstance(y, np.ndarray):
            y = np.array(y)
        if not isinstance(weights, np.ndarray):
            weights = np.array(weights)

        # 检查标签是否合法
        y = self._label_correct(y)

        # 获取样本数和特征数
        n_samples, n_features = X.shape

        # 遍历每个特征，找到最好的阈值和极性
        for feature_idx in range(n_features):
            feature = X[:, feature_idx]
            thresholds = np.unique(feature)

            # 遍历所有可能的阈值
            for threshold in thresholds:

                # 尝试两种不同的极性哪个更好
                for polarity in [1, -1]:
                    predictions = np.ones(n_samples)
                    if polarity == 1:
                        predictions[feature < threshold] = -1
                    else:
                        predictions[feature >= threshold] = -1

                    # 计算加权错误率
                    err = np.sum(weights[predictions != y])

                    if err < self.min_error:
                        self.min_error = err
                        self.feature_idx = feature_idx
                        self.threshold = threshold
                        self.polarity = polarity
        return self


    def predict(self, X):
        """
        预测函数

        :param X: 测试数据集
        :return: 预测标签，值为+1或-1
        """
        if not isinstance(X, np.ndarray):
            X = np.array(X)
        n_samples = X.shape[0]
        feature = X[:, self.feature_idx]
        predictions = np.ones(n_samples)

        if self.polarity == 1:
            predictions[feature < self.threshold] = 0
        else:
            predictions[feature >= self.threshold] = 0
        return predictions
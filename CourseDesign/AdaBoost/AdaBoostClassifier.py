import warnings

from .BaseModel import BaseModel
import numpy as np
from .DecisionStumpClassifier import DecisionStumpClassifier


class AdaBoostClassifier(BaseModel):
    """
    AdaBoost分类器
    """
    def __init__(self, estimator_class=DecisionStumpClassifier, n_estimators=50, **base_params):
        """
        :param estimator_class: 基学习器类（默认为决策树桩）
        :param n_estimators: 若分类器的数量
        :param base_params: 传递给基学习器的参数
        """
        super().__init__()
        self.estimator_class = estimator_class  # 基学习器类
        self.n_estimators = n_estimators        # 基学习器的数量
        self.estimators = []                     # 存储基学习器的列表
        self.base_params = base_params          # 传递给基学习器的参数
        self._epsilon = 1e-12                   # 极小常数，用于除法、取对数中出现的一些错误


    def fit(self, X, y):
        """
        训练函数

        :param X: 训练集特征
        :param y: 训练集标签
        :return: self
        """
        if not isinstance(X, np.ndarray):
            X = np.array(X)
        if not isinstance(y, np.ndarray):
            y = np.array(y)

        n_samples, n_features = X.shape

        # 初始化权重分布
        weights = np.ones(n_samples) / n_samples

        # 训练基学习器
        for _ in range(self.n_estimators):
            # 创建基学习器实例
            model = self.estimator_class(**self.base_params)
            # 以相应权重训练学习器并计算加权错误率
            min_error = model.fit(X, y, weights).min_error

            # 若基分类器错误率比随机猜测还差则终止算法
            if min_error > 0.5:
                continue

            # 计算分类器的权重
            alpha = 0.5 * np.log((1 - min_error + self._epsilon)
                                 / (min_error + self._epsilon))
            model.alpha = alpha

            prediction = model.predict(X)
            # 更新样本权重
            weights *= np.exp(-alpha * y * prediction)
            # 归一化权重
            weights /= np.sum(weights)

            self.estimators.append(model)

            if min_error == 0:
                break

        return self


    def predict(self, X):
        """
        预测函数

        :param X: 测试集特征
        :return: 预测的标签，值为+1或-1
        """
        if not isinstance(X, np.ndarray):
            X = np.array(X)

        n_samples = X.shape[0]
        y_pred = np.zeros(n_samples)

        if not self.estimators:
            warnings.warn("没有训练基学习器，返回全1数组")
            return np.ones(n_samples)

        for model in self.estimators:
            y_pred += model.predict(X) * model.alpha

        return np.sign(y_pred)






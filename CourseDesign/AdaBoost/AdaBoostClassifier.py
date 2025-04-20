from .BaseModel import BaseModel
import numpy as np
import copy

"""
模仿sklearn的AdaBoostClassifier
自己实现了一个简化版的AdaBoostClassifier
"""


class BaseWeightBoostingModel(BaseModel):
    def __init__(self, estimator=None, n_estimators=50, learning_rate=0.5):
        """
        初始化基于权重的Boosting模型
        :param estimator: 基学习器
        :param n_estimators: 基学习器的数量
        :param learning_rate: 学习率
        :param random_state: 随机数种子
        """
        super().__init__()
        self.estimator = estimator
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.estimators_ = None
        self.estimator_weights_ = None
        self.estimator_errors = None

    def fit(self, X, y, sample_weight=None):
        """
        训练模型
        :param X: 训练集特征
        :param y: 训练集标签
        :param sample_weight: 样本权重
        :return:
        """
        # 检查输入数据
        if not isinstance(X, np.ndarray):
            X = np.array(X)
        if not isinstance(y, np.ndarray):
            y = np.array(y)

        # 检查样本权重
        if sample_weight is None:
            sample_weight = np.ones(X.shape[0]) / X.shape[0]
        else:
            if not isinstance(sample_weight, np.ndarray):
                sample_weight = np.array(sample_weight)
        sample_weight /= sample_weight.sum()

        # 检查基学习器
        if self.estimator is None:
            raise ValueError("基学习器不能为空")
        self.estimators_ = []
        self.estimator_weights_ = np.zeros(self.n_estimators, dtype=np.float64)
        self.estimator_errors = np.zeros(self.n_estimators, dtype=np.float64)

        epsilon = np.finfo(sample_weight.dtype).eps

        zero_weight_mask = sample_weight == 0.0

        for iboost in range(self.n_estimators):
            sample_weight = np.clip(sample_weight, a_min=epsilon, a_max=None) # 避免样本权重为0
            sample_weight[zero_weight_mask] = 0.0 # 将原石样本权重为0的样本的权重设置为0

            # 训练基学习器
            sample_weight, estimator_weight, estimator_error = self._boost(iboost, X, y, sample_weight)

            # 早停机制
            if sample_weight is None:
                break
            self.estimator_weights_[iboost] = estimator_weight
            self.estimator_errors[iboost] = estimator_error
            if estimator_error == 0:
                break
            sample_weight_sum = np.sum(sample_weight)

            if sample_weight_sum <= 0:
                break
            if iboost < self.n_estimators - 1:
                sample_weight /= sample_weight_sum

        return self

    def _boost(self, iboost, X, y, sample_weight):
        """
        训练基学习器

        :param iboost: 基学习器的索引
        :param X: 训练集特征
        :param y: 训练集标签
        :param sample_weight: 样本权重
        :return:
        """
        pass

    def predict(self, X):
        """
        预测函数
        :param X:
        :return:
        """
        pass



class AdaBoostClassifier(BaseWeightBoostingModel):
    """
    AdaBoost分类器
    """

    def __init__(self, estimator=None, n_estimators=50, learning_rate=1.0):
        """
        初始化AdaBoost分类器
        :param estimator: 基学习器
        :param n_estimators: 基学习器的数量
        :param learning_rate: 学习率
        """
        super().__init__(estimator, n_estimators, learning_rate)

    def _boost(self, iboost, X, y, sample_weight):
        """
        训练基学习器

        :param iboost: 基学习器的索引
        :param X: 特征集合
        :param y: 标签集合
        :param sample_weight: 样本权重
        :return: 更新后的样本权重、基学习器权重和基学习器错误率
        """
        estimator = copy.deepcopy(self.estimator)
        # estimator._random_init(X, y)
        self.estimators_.append(estimator)

        # 以权重训练基学习器
        estimator.fit(X, y, sample_weight=sample_weight)

        y_predict = estimator.predict(X)

        # 计算加权错误率
        incorrect = y_predict != y

        # estimator_error = np.mean(np.average(incorrect, weights=sample_weight, axis=0))
        estimator_error = np.average(incorrect, weights=sample_weight, axis=0)  # 此处不考虑多维

        # 早停机制
        if estimator_error <= 0:
            return sample_weight, 1.0, 0.0

        if estimator_error >= 0.5:  # 此处不考虑多分类
            self.estimators_.pop(-1)
            if len(self.estimators_) == 0:
                raise ValueError("无法进行分类")
            return None, None, None

        # 基学习器权重更新
        estimator_weight = self.learning_rate * np.log((1.0 - estimator_error) / estimator_error)

        # 更新样本权重
        if not iboost == self.n_estimators - 1:
            # sample_weight *= np.exp(estimator_weight * incorrect * (sample_weight > 0))
            sample_weight *= np.exp(estimator_weight * incorrect * (sample_weight > 0))
            sample_weight *= np.exp(-estimator_weight * (1 - incorrect) * (sample_weight > 0))

        # 归一化样本权重
        sample_weight /= np.sum(sample_weight)
        return sample_weight, estimator_weight, estimator_error

    def predict(self, X, is_boolean_label=True):
        """
        预测函数
        :param X: 预测集特征
        :param is_boolean_label: 是否将标签转化为0和1
        :return: 预测结果
        """
        if not isinstance(X, np.ndarray):
            X = np.array(X)
        pred = np.zeros(X.shape[0], dtype=np.float64)
        for i in range(pred.shape[0]):
            for estimator, weight in zip(self.estimators_, self.estimator_weights_):
                pred[i] += weight * (estimator.predict(X[i, :].reshape(1, -1)))

        pred = np.sign(pred)
        if is_boolean_label:
            pred[pred == -1] = 0
        return np.array(pred, dtype=int)


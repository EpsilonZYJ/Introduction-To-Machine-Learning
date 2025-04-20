import numpy as np

class BaseModel(object):
    def __init__(self):
        """
        基模型的初始化函数
        """
        pass

    def fit(self, X, y):
        """
        基模型的训练函数
        :param X: 训练集特征
        :param y: 训练集标签
        :return:
        """
        raise NotImplementedError("fit方法未实现")

    def predict(self, X):
        """
        基模型的预测函数
        :param X: 测试集特征
        :return: 预测结果
        """
        raise NotImplementedError("predict方法未实现")
    
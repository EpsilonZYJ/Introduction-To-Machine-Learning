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


class BaseWeakLearner(BaseModel):
    """
    弱学习器的基类，定义基本接口
    """
    def fit(self, X, y, weights):
        """
        使用加权样本进行训练

        :param X: 输入特征
        :param y: 训练集标签
        :param weights: 样本权重
        :return:
        """
        raise NotImplementedError("fit方法未实现")


    def _label_correct(self, y):
        """
        检查标签是否合法,若为0和1，则将0转化为-1

        :param y: 标签
        :return: 修正后的标签
        """
        label_set = set(y)
        if len(label_set) != 2:
            raise ValueError("标签必须为二分类")
        if -1 not in label_set:
            y = 2 * y - 1
        return np.array(y)


def binary_label(y):
    """
    将标签转化为0和1
    :param y: 标签
    :return: 转化后的标签
    """
    if not isinstance(y, np.ndarray):
        y = np.array(y)
    return np.array((y + 1)/2 , dtype=int)

def pos_neg_label(y):
    """
    将标签转化为-1和1
    :param y: 标签
    :return: 转化后的标签
    """
    if not isinstance(y, np.ndarray):
        y = np.array(y)
    return 2 * y - 1
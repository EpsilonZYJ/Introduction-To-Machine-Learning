import numpy as np
from.debug_wrapper import Debug

@Debug
def accuracy(y_true, y_pred):
    """
    计算预测准确率
    :param y_true: 真实标签
    :param y_pred: 预测标签
    :return: 准确率，类型为float，范围在[0, 1]之间
    """
    if not isinstance(y_true, np.ndarray):
        y_true = np.array(y_true)
    if not isinstance(y_pred, np.ndarray):
        y_pred = np.array(y_pred)
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true和y_pred的形状不一致")
    correct = np.sum(y_true == y_pred)
    total = y_true.shape[0]
    acc = correct / total
    return acc

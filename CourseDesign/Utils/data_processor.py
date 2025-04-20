import pandas as pd
import numpy as np
import copy

from AdaBoost import AdaBoost
from .Accuracy import accuracy
from .debug_wrapper import Debug


@Debug
def load_feature_data(filepath: str):
    """
    从csv文件加载特征数据并将其作为numpy数组返回。
    :param filepath: 相对文件路径，类型为str
    :return: 返回特征数组，类型为ndarray
    """
    data = pd.read_csv(filepath, header=None)
    return data.to_numpy()

@Debug
def load_label_data(filepath: str):
    """
    从csv文件加载标签数据并将其作为numpy数组返回。
    :param filepath: 相对文件路径，类型为str
    :return: 返回标签数组，类型为ndarray
    """
    data = pd.read_csv(filepath, header=None)
    data = data.to_numpy()
    data = data.flatten()
    return data


@Debug
def save_data(filepath: str, data, index):
    """
    将预测数据保存到csv文件中。
    :param index: 索引
    :param filepath: 存储路径，类型为str
    :param data: 预测结果
    :return:
    """
    if isinstance(data, np.ndarray):
        data = data.tolist()
    if isinstance(index, np.ndarray):
        index = index.tolist()
    df = pd.DataFrame(
        {'index': index, 'predict': data}
    )
    df.to_csv(filepath, index=False, header=False)

@Debug
def get_k_fold_data(k, i, X, y):
    """
    获取k折交叉验证的数据

    :param k: 折数
    :param i: 第i折
    :param X: 特征集合
    :param y: 标签集合
    :return: 第i折的训练集和测试集
    """
    assert k > 0
    if not isinstance(X, np.ndarray):
        X = np.array(X)
    if not isinstance(y, np.ndarray):
        y = np.array(y)
    fold_size = X.shape[0] // k
    X_train, y_train = None, None
    for j in range(k):
        idx = slice(j * fold_size, min((j + 1) * fold_size, X.shape[0]))
        X_part = X[idx, :]
        y_part = y[idx]
        # 如果是第i折，则将其作为验证集
        if j == i:
            X_valid, y_valid = X_part, y_part
        # 否则将其作为训练集
        elif X_train is None:
            X_train, y_train = X_part, y_part
        else:
            X_train = np.concatenate((X_train, X_part), axis=0)
            y_train = np.concatenate((y_train, y_part), axis=0)
    return X_train, y_train, X_valid, y_valid, np.array(range(i * fold_size + 1, min((i + 1) * fold_size, X.shape[0]) + 1))

@Debug
def k_fold(k, X_train, y_train, n_estimators, base_model):
    train_acc_sum, valid_acc_sum = 0, 0
    for i in range(k):
        data = get_k_fold_data(k, i, X_train, y_train)
        model = AdaBoost(n_estimators=n_estimators, base_estimator=copy.deepcopy(base_model))
        model.fit(data[0], data[1])
        train_acc_sum += accuracy(data[1], model.predict(data[0]))
        valid_acc_sum += accuracy(data[3], model.predict(data[2]))
        print(f'Fold {i + 1}: train acc: {train_acc_sum / (i + 1):.4f}, valid acc: {valid_acc_sum / (i + 1):.4f}')
        save_data(f'./experiments/base{n_estimators}_fold{i + 1}.csv', model.predict(data[2]), data[4])
    return train_acc_sum / k, valid_acc_sum / k


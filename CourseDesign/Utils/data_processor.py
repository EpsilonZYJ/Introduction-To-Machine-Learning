import pandas as pd
import numpy as np
import copy

from AdaBoost import AdaBoostClassifier, pos_neg_label, binary_label
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
    if len(data) != len(index):
        raise ValueError("数据长度和索引长度不一致")
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
def class_balance(X, y):
    """
    处理训练集类别不平衡的问题
    :param X: 训练集特征
    :param y: 训练集标签
    :return:
    """
    if not isinstance(X, np.ndarray):
        X = np.array(X)
    if not isinstance(y, np.ndarray):
        y = np.array(y)
    feature0 = X[y == 0]
    feature1 = X[y == 1]
    label0 = y[y == 0]
    label1 = y[y == 1]
    count0 = len(label0)
    count1 = len(label1)
    cnt = min(count0, count1)

    # 如果类别不平衡的比例过大，则不进行处理
    if count0 * 1e2 <count1 or count1 * 1e2 < count0:
        return X, y

    # 随机选择count个样本
    indices = np.random.choice(len(label0), cnt, replace=False)
    feature0 = feature0[indices]
    label0 = label0[indices]

    indices = np.random.choice(len(label1), cnt, replace=False)
    feature1 = feature1[indices]
    label1 = label1[indices]

    # 合并数据
    X = np.concatenate((feature0, feature1), axis=0)
    y = np.concatenate((label0, label1), axis=0)
    # 打乱数据
    indices = np.random.permutation(X.shape[0])
    X = X[indices]
    y = y[indices]
    return X, y



@Debug
def k_fold(k, X_train, y_train, n_estimators, base_model_class, isClassBalanced=True, isShuffled=False, **base_model_params):
    """
    k折交叉验证，训练模型并保存结果

    :param k: 折数
    :param X_train: 训练集特征
    :param y_train: 训练集标签
    :param n_estimators: 基学习器的数量
    :param base_model_class: 基学习器的类
    :param isClassBalanced: 是否进行类别平衡
    :param isShuffled: 是否打乱数据
    :param base_model_params: 传给基学习器的参数
    :return: 总的训练集准确率和验证集准确率
    """
    train_acc_sum, valid_acc_sum = 0, 0
    for i in range(k):
        data = get_k_fold_data(k, i, X_train, y_train)
        model = AdaBoostClassifier(n_estimators=n_estimators, estimator_class=base_model_class, **base_model_params)
        train_feature = data[0]
        train_label = data[1]

        # 处理类别不平衡
        if isClassBalanced:
            train_feature, train_label = class_balance(train_feature, train_label)

        if isShuffled:
            # 打乱数据
            indices = np.random.permutation(train_feature.shape[0])
            train_feature = train_feature[indices]
            train_label = train_label[indices]

        # 处理标签
        train_label = pos_neg_label(train_label)

        # 训练模型
        model.fit(train_feature, train_label)

        train_pred = model.predict(data[0])
        valid_pred = model.predict(data[2])
        # 处理标签
        train_pred = binary_label(train_pred)
        valid_pred = binary_label(valid_pred)

        train_acc_sum += accuracy(data[1], train_pred)
        valid_acc_sum += accuracy(data[3], valid_pred)
        print(f'Fold {i + 1}, Accumulated accuracy: train acc: {train_acc_sum / (i + 1):.4f}, valid acc: {valid_acc_sum / (i + 1):.4f}')
        # print(valid_pred)
        # print(data[3])
        save_data(f'./experiments/base{n_estimators}_fold{i + 1}.csv', valid_pred, data[4])
    return train_acc_sum / k, valid_acc_sum / k


import pandas as pd
import numpy as np
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
    fout = [index, data]
    df = pd.DataFrame(fout)
    df.to_csv(filepath, index=False, header=False)

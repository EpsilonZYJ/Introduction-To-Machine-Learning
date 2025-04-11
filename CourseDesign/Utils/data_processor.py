import pandas as pd
import numpy as np
from .debug_wrapper import Debug

@Debug
def load_data(filepath: str):
    """
    从csv文件加载数据并将其作为numpy数组返回。
    :param filepath: 相对文件路径，类型为str
    :return: 返回的数组，类型为ndarray
    """
    data = pd.read_csv(filepath, header=None)
    return data.to_numpy()

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

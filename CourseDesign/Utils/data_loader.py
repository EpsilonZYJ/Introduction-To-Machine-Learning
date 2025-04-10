import pandas as pd
import numpy as np
from .debug_wrapper import Debug

@Debug
def load_data(filepath):
    """
    从csv文件加载数据并将其作为numpy数组返回。
    :param filepath: 相对文件路径
    :return: 返回的数组，类型为ndarray
    """
    data = pd.read_csv(filepath, header=None)
    return data.to_numpy()

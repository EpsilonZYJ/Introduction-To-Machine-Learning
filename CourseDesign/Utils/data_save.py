import pandas as pd
from .debug_wrapper import Debug

@Debug
def save_data(data, base, fold):
    """
    将预测数据保存到csv文件中。
    :param data: 预测结果
    :param base: 基分类器的数目
    :param fold: 折数
    :return:
    """

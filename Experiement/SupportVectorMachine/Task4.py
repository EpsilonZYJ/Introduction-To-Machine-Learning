# encoding=utf8

import numpy as np


# 实现核函数
def kernel(x, sigma=1.0):
    '''
    input:x(ndarray):样本
    output:x(ndarray):转化后的值
    '''
    # ********* Begin *********#
    z = x
    x = (np.dot(x, z.T) + 1) ** 3
    # ********* End *********#
    return x

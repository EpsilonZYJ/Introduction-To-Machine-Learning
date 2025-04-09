from sklearn.preprocessing import StandardScaler

def scaler(data):
    '''
    返回标准化后的红酒数据
    :param data: 红酒数据对象
    :return: 标准化后的红酒数据，类型为ndarray
    '''

    #********* Begin *********#
    scale = StandardScaler()
    dataSet = data['data']
    dataSet = scale.fit_transform(dataSet)
    return dataSet
    #********* End **********#

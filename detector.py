# -*- coding: utf-8 -*-
# @Time : 2022/5/17
# @Author : fanze
# @Email : 184286692@qq.com
# @File : detector

import pandas as pd
import numpy as np
from pandas import DataFrame


class Detector(object):
    # 探测器父类
    NAME = 'detector'

    def __init__(self, data: DataFrame, name_Y: str):
        self.data = data
        self.name_Y = name_Y
        return

    def detectorname(self):
        return self.NAME

    def train(self):
        # to train the model
        # first to check to parameer
        x, y = checkparameter(self.data, self.name_Y)

        return x, y
    def putout(self):
        pass

def checkparameter(data, name_Y):
    """
    Use to check the correctness of X and Y, get X and Y at the same time
    :param data:
    :param name_Y:
    :return:
    """
    # check type of data and name_Y
    if not isinstance(data, pd.DataFrame):
        raise Exception("X should be of type dataframe")
    if not isinstance(name_Y, str):
        raise Exception("Y should be of type string")
    if name_Y not in list(data.columns.values):
        raise Exception("There is no name_Y column in data")
    else:
        temp_list = list(data.columns.values)
        index_Y = temp_list.index(name_Y)
        temp_list = temp_list.remove(name_Y)
    # get the index of name_Y
    a = data.iloc[:, :].values
    X = np.delete(a, index_Y, axis=1)
    Y = a[:, index_Y]
    name_factors = temp_list
    return X, Y, name_factors

if __name__ == '__main__':
    Detector.NAME = 'DAF'
    a = Detector(X = 1, Y = 4)
    a.NAME = 'daf'
    print(a.detectorname())
# -*- coding: utf-8 -*-
# @Time : 2022/5/17
# @Author : fanze
# @Email : 184286692@qq.com
# @File : difacdetector
from detector import Detector
from collections import Counter
import numpy as np

class DifferentiationFactorDetector(Detector):
    NAME = 'DifferentiationFactorDetector'

    def __init__(self, data, name_Y):
        super().__init__(self, data, name_Y)

    def train(self):
        # Calls error checking for the parent class and returns x and y
        x, y, name_factors = super().train()
        # train the Anomaly and factor detection models . res is list of q value
        res = mdoel(x, y)
        resdict = dict(zip(name_factors, res))
        return resdict

    def putout(self):

        pass

def mdoel(x, y):
    """
    Anomaly and factor detection models
    :param x:
    :param y:
    :return: list of q
    """
    list_q = []
    for i in range(x.shape[1]):
        a = x[:, i]
        list_q.append(calculateq(a, y))

    return list_q

def calculateq(a, y):
    '''
    calculate q values
    :param a:
    :param y:
    :return: ;list of q value
    '''

    set1 = set(a)
    result = Counter(a)
    dict_fangcha = {}
    for i in set1:
        dict_fangcha[i] = []
    for j in range(len(a)):
        dict_fangcha[a[j]].append(y[j])
    SSW = 0.0
    for i in range(len(set1)):
        SSW = SSW + result[i] * np.var(dict_fangcha[i])
    SST = len(a) * np.var(y)
    q = 1 - (SSW / SST)
    return q

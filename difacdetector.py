# -*- coding: utf-8 -*-
# @Time : 2022/5/17
# @Author : fanze
# @Email : 184286692@qq.com
# @File : difacdetector
from detector import Detector
from collections import Counter
import numpy as np
from scipy.stats import ncf

class DifferentiationFactorDetector(Detector):
    NAME = 'DifferentiationFactorDetector'

    def __init__(self, data, name_Y):
        super().__init__(self, data, name_Y)

    def train(self):
        # Calls error checking for the parent class and returns x and y
        x, y, name_factors = super().train()
        # train the Anomaly and factor detection models . res is list of q value and F value
        list_q, list_F = mdoel(x, y)
        resqdict = dict(zip(name_factors, list_q))
        resFdict = dict(zip(name_factors, list_F))

        return


    def putout(self):
        # Output an ndrray, q value and F value

        pass

def mdoel(x, y):
    """
    Anomaly and factor detection models
    :param x:
    :param y:
    :return: list of q
    """
    list_q = []
    list_F = []
    for i in range(x.shape[1]):
        a = x[:, i]
        q, F = calculateq(a, y)
        list_q.append(q)
        list_F.append(F)

    return list_q, list_F

def calculateq(a, y):
    '''
    calculate q values
    :param a:
    :param y:
    :return: ;list of q value，list of F value
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

    F = ((len(a) - len(result)) / (len(result) - 1)) * (( q ) / (q - 1))

    cau = 0.0
    cau1 = 0.0
    for i in range(len(set1)):
        cau = cau + np.mean(dict_fangcha[i])**2
        cau1 = cau1 + pow(result[i],2)* np.mean(dict_fangcha[i])

    λ = (1/np.var(y)) * (cau **2 - (1/len(result))* (cau1**2))
    v = ncf.pdf(0.05, (len(result) - 1)), (len(a) - len(result), λ)
    F = v

    return q, F

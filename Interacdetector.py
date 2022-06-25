# -*- coding: utf-8 -*-
# @Time : 2022/6/25
# @Author : fanze
# @Email : 184286692@qq.com
# @File : Interacdetector
# -*- coding: utf-8 -*-
# @Time : 2022/5/17
# @Author : fanze
# @Email : 184286692@qq.com
# @File : difacdetector

from detector import Detector
from collections import Counter
import numpy as np
from scipy.stats import ncf

class InteractionDetector(Detector):
    ## InteractionDetector kind of like DifferentiationFactorDetector， so we make it inherit DifferentiationFactorDetector
    NAME = 'InteractionDetector'

    def __init__(self, data, name_Y):
        super().__init__(self, data, name_Y)

    def train(self):
        # Calls error checking for the parent class and returns x and y
        x, y, name_factors = super().train()
        # Regenerate X and name_factors
        changedx, changedname_factors = Interactiontransform(x, name_factors)
        # train the Anomaly and factor detection models . res is list of q value and F value
        changedlist_q, changedlist_F = mdoel(x, y)
        resqdict = dict(zip(name_factors, changedlist_q))
        resFdict = dict(zip(name_factors, changedlist_F))

        # train the  changedx, changedname_factors. res is list of q value and F value
        list_q, list_F = mdoel(x, y)
        changedresqdict = dict(zip(changedname_factors, list_q))
        changedresFdict = dict(zip(changedname_factors, list_F))
        return

    def putout(self):
        # Output an ndrray, q value and F value

        pass

def Interactiontransform(x, name_factors):
    """
    :param x:
    :param y:
    :return: changedx, name_factors
    """
    # Regenerate X  N columns x have 1+... + n - 1 column

    # Compute the number of columns of changedx,
    numbercolumnschangedx = (x.shape[1]*(x.shape[1]-1))/2

    changedname_factors = []
    changedx = []
    #np.zeros((x.shape[0], numbercolumnschangedx))
    # Start Regenerate
    for i in range(x.shape[1]):
        for j in range(i+1, x.shape[1]):
            changedname_factors.append(name_factors[i]+'AND' + name_factors[j])
            count1 = set(x[ : i])
            count2 = set(x[ : j])
            for m, n in zip(x[ : i], x[ : j]):
                for k in range(len(count1)):
                    for l in range(len(count2)):
                        if m == count1[k] and n == count2[l]:
                            changedx.append[(k)*len(count2) + l]

    changedx = np.array(changedx).reshape((x.shape[0], numbercolumnschangedx))
    changedx = np.transpose(changedx)
    return changedx , changedname_factors

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

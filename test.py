# -*- coding: utf-8 -*-
# @Time : 2022/5/17
# @Author : fanze
# @Email : 184286692@qq.com
# @File : test

from difacdetector import DifferentiationFactorDetector
import pandas as pd
import numpy as np
# Load the data
data = pd.read_csv("testdata/test.csv")
# Initialize  model
a = DifferentiationFactorDetector(data, '类型')
# model train
res = a.train()

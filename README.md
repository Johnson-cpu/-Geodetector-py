# -Geodetector-py

# Overview
Spatial differentiation is one of the basic characteristics of geographical phenomena. Geographic detectors are tools for detecting and exploiting spatial heterogeneity. The geodetector consists of four detectors.

Differentiation and factor detection: detect the spatial differentiation of Y; And probing the extent to which a factor X explains the spatial differentiation of attribute Y. Measure it in terms of q

Interaction detection: Identify the interaction between different risk factors Xs, that is, evaluate whether the combined action of factors X1 and X2 will increase or decrease the explanatory power of dependent variable Y, or the influence of these factors on Y is independent of each other.

Risk area detection: Used to judge whether there is a significant difference in the average value of attributes between two sub-areas, tested by t statistics:

Ecological detection: it is used to compare whether there is significant difference in the influence of two factors X1 and X2 on the spatial distribution of attribute Y, measured by F statistic

# Usage
```python
from difacdetector import DifferentiationFactorDetector
import pandas as pd
import numpy as np
# Load the data
data = pd.read_csv("testdata/test.csv")
# Initialize  model
a = DifferentiationFactorDetector(data, '类型')
# model train
res = a.train()
```

# Reference
[1] http://www.geog.com.cn/article/2017/0375-5444/0375-5444-72-1-116.shtml
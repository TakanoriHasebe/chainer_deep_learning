#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 28 08:51:00 2018

@author: Takanori
"""

import numpy as np

X = np.array([[1,2,3,4],[5,6,7,8],[1,2,3,4],[5,6,7,8]]).astype(np.float32)
print('X.shape : '+str(X.shape))

N = X[0].size
print('N : '+str(N))

index = np.arange(N)
print(index)

xtrain = X[index[index % 2 != 0], :]
print(xtrain)

print(X[0,:])





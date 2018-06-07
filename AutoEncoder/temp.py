#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  7 11:15:02 2018

@author: Takanori
"""
# よく用いるモジュール
import numpy as np
import chainer
from chainer import cuda, Function, \
        report, training, utils, Variable
from chainer import datasets, iterators, optimizers, serializers
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L
from chainer.training import extensions
# Trainerを用いる場合
from chainer.datasets import tuple_dataset
from chainer import training
from chainer.training import extensions
from sklearn import datasets
from zodbpickle import pickle
import matplotlib.pyplot as plt

list1 = [[1,2,3],[4,5,6]]
print(list1)

arr1 = np.array(list1)
print(arr1)
print(type(arr1))
print(arr1[0:2,2])









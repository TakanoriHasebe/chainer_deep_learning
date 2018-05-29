#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 28 10:44:55 2018

@author: Takanori
"""
import numpy as np
import chainer
from chainer import cuda, Function, \
        report, training, utils, Variable
from chainer import datasets, iterators, optimizers, serializers
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L
from chainer.training import extensions

class IrisLogi(Chain):
    def __init__(self):
        super(IrisLogi, self).__init__(
                l1 = L.Linear(4,3),
        )
        
    def __call__(self,x,y):
        return F.mean_squared_error(self.fwd(x), y)
    
    def fwd(self,x):
        return F.softmax(self.l1(x))
    
model = IrisLogi()
optimizer = optimizers.Adam()
optimizer.setup(model)




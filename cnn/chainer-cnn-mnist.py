#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 30 10:18:15 2018

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

train, test = datasets.get_mnist(ndim=3)
# print(type(train), type(test)) # tuple型になっている事に注意
# 以下でもtuple型に変更できる

class MyModel(Chain):
    def __init__(self):
        super(IrisChain, self).__init__(
            cn1 = L.Convolution2D(1,20,5),
            cn2 = L.Convolution2D(20,50,5),
            l1=L.Linear(4,6),
            l2=L.Linear(6,3),
        )
        
    def __call__(self,x,y):
        return F.softmax_cross_entropy(self.fwd(x), t)
    
    def fwd(self,x):
         h1 = F.max_pooling_2d(F.relu(self.cn1(x)), 2)
         h2 = F.max_pooling_2d(F.relu(self.cn2(x)), 2)
         h3 = F.dropout(F.relu(self.l1(h2)))
         return h3
     
model = MyModel()
# optimizer = optimizers.SGD()
optimizer = optimizers.Adam()
optimizer.setup(model)

### trainerを用いた場合
train_iter = iterators.SerialIterator(train, 1000) # ミニバッチサイズ
updater = training.StandardUpdater(train_iter, optimizer) # 
trainer = training.Trainer(updater, (10, 'epoch')) # 5000epoch
trainer.extend(extensions.ProgressBar()) # どれだけ学習が進んでいるかをみる
trainer.run() # 走らせる















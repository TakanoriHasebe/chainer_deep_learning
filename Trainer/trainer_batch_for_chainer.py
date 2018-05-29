#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 29 11:12:08 2018

@author: Takanori
"""

#!/usr/bin/env python

import numpy as np
import chainer
from chainer import cuda, Function, gradient_check, report, training, utils, Variable
from chainer import datasets, iterators, optimizers, serializers
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L
from chainer.datasets import tuple_dataset
from chainer import training
from chainer.training import extensions

# Set data

from sklearn import datasets
iris = datasets.load_iris()
X = iris.data.astype(np.float32)
Y = iris.target
N = Y.size
Y2 = np.zeros(3 * N).reshape(N,3).astype(np.float32)
for i in range(N):
    Y2[i,Y[i]] = 1.0

index = np.arange(N)
xtrain = X[index[index % 2 != 0],:]
ytrain = Y2[index[index % 2 != 0],:]
xtest = X[index[index % 2 == 0],:]
yans = Y[index[index % 2 == 0]]

train = tuple_dataset.TupleDataset(xtrain, ytrain)

# Define model

class IrisChain(Chain):
    def __init__(self):
        super(IrisChain, self).__init__(
            l1=L.Linear(4,6),
            l2=L.Linear(6,3),
        )
        
    def __call__(self,x,y):
        return F.mean_squared_error(self.fwd(x), y)
    
    def fwd(self,x):
         h1 = F.sigmoid(self.l1(x))
         h2 = self.l2(h1)
         return h2
     
# Initialize model

model = IrisChain()
optimizer = optimizers.SGD()
optimizer.setup(model)

# Learn
# ミニバッチのサイズを指定してfor文を回す
def bunkai(batch, batchsize): 
    x = []; t = []
    for j in range(batchsize):
        x.append(batch[j][0])
        t.append(batch[j][1])
    return Variable(np.array(x)), Variable(np.array(t)) 

bsize = 25
for n in range(5000):
    for i in iterators.SerialIterator(train, bsize, repeat=False): # ミニバッチの作成
        x, t = bunkai(i, bsize)
        model.cleargrads()
        loss = model(x, t)
        loss.backward()
        optimizer.update()

# Test

xt = Variable(xtest)
yy = model.fwd(xt)

ans = yy.data
nrow, ncol = ans.shape
ok = 0
for i in range(nrow):
    cls = np.argmax(ans[i,:])
    print(ans[i,:], cls)            
    if cls == yans[i]:
        ok += 1
        
print(ok, "/", nrow, " = ", (ok * 1.0)/nrow)


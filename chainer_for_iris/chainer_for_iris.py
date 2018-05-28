#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 28 08:40:48 2018

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

from sklearn import datasets

iris = datasets.load_iris()

X = iris.data.astype(np.float32)
Y = iris.target
N = Y.size

# print('X.shape:'+str(X.shape))
# print(Y.shape)
# print(Y)

Y2 = np.zeros(3 * N).reshape(N,3).astype(np.float32)

# one_hot_labelに変換
for i in range(N):
    Y2[i, Y[i]] = 1.0

# print(Y2.shape)
index = np.arange(N)
# print('index'+str(index))

# 訓練データ、教師データの作成
xtrain = X[index[index % 2 != 0], :] # 奇数番目の行の取り出し 
ytrain = Y2[index[index % 2 != 0], :] # 同様
xtest = X[index[index % 2 == 0], :] # 偶数番目の行の取り出し
yans = Y[index[index % 2 == 0]] # 同様

print(xtrain.shape)
# print(ytrain.shape)
# print(xtest.shape)

# NNの概要を作成
class IrisChain(Chain):
    def __init__(self):
        super(IrisChain, self).__init__(
                l1 = L.Linear(4, 6),
                # l2 = L.Linear(6, 3),
                l2 = L.Linear(6, 5),
                l3 = L.Linear(5, 3),
        )
        
    def __call__(self,x,y):
        return F.mean_squared_error(self.fwd(x), y)
        # return F.softmax_cross_entropy(self.fwd(x), y)
    
    def fwd(self,x):
        h1 = F.sigmoid(self.l1(x))
        h2 = F.sigmoid(self.l2(h1))
        # h2 = self.l2(h1)
        h3 = self.l3(h2)
        # h4 = F.softmax(h3)
        return h3

# 勾配法の設定
model = IrisChain()
# optimizer = optimizers.SGD()
optimizer = optimizers.Adam()
optimizer.setup(model)

n = 75
bs = 25
# NNの学習
for j in range(5000):
    sffindx = np.random.permutation(n)
    for i in range(0, n, bs):
        """
        # バッチ処理
        x = Variable(xtrain)
        y = Variable(ytrain)
        """
        x = Variable(xtrain[sffindx[i : (i + bs) if (i + bs) < n else n]])
        y = Variable(ytrain[sffindx[i : (i + bs) if (i + bs) < n else n]])
        model.cleargrads()
        loss = model(x, y)
        loss.backward()
        optimizer.update()

xt = Variable(xtest) # Chainerで用いれるように変換
yt = model.fwd(xt) # 出力
ans = yt.data # 答え
nrow, ncol = ans.shape
ok = 0

for i in range(nrow):
    cls = np.argmax(ans[i, :])
    if cls == yans[i]:
        ok += 1
        
print(str(ok)+"/"+str(nrow)+"="+str((ok*1.0)/nrow))





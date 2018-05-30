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

# 作成手順が記されている

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

### trainerを用いない場合の訓練
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
        # ミニバッチ処理
        x = Variable(xtrain[sffindx[i : (i + bs) if (i + bs) < n else n]])
        y = Variable(ytrain[sffindx[i : (i + bs) if (i + bs) < n else n]])
        model.cleargrads()
        loss = model(x, y)
        loss.backward()
        optimizer.update()

### trainerを用いた場合
train_iter = iterators.SerialIterator(train, 25) # バッチサイズ
updater = training.StandardUpdater(train_iter, optimizer) # 
trainer = training.Trainer(updater, (5000, 'epoch')) # 5000epoch
trainer.extend(extensions.ProgressBar()) # どれだけ学習が進んでいるかをみる
trainer.run() # 走らせる

# 結果の出力
xt = Variable(xtest) # Chainerで用いれるように変換
yt = model.fwd(xt) # 作成したモデルで出力を得る

ans = yt.data # 答え
nrow, ncol = ans.shape
ok = 0

for i in range(nrow):
    cls = np.argmax(ans[i, :])
    if cls == yans[i]: # 結果 == 教師データ
        ok += 1
        
print(str(ok)+"/"+str(nrow)+"="+str((ok*1.0)/nrow))





























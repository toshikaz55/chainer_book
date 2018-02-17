import chainer
import chainer.functions as F
import chainer.links as L
from chainer import training, datasets, iterators, optimizers
from chainer.training import extensions
import numpy as np
import os
import math
from PIL import Image

# バッチサイズ 128
batch_size = 256
# GPU 0を使用
uses_device = 0

# GPU使用時とCPU使用時でデータ形式が変わる
if uses_device >= 0:
    import cupy as cp
    import chainer.cuda
else:
    cp = np

class SuperResolution_NN(chainer.Chain):

    def __init__(self):
        # 重みデータの初期値を設定する
        w1 = chainer.initializers.Normal(scale = 0.0378, dtype=None)
        w2 = chainer.initializers.Normal(scale = 0.3536, dtype=None)
        w3 = chainer.initializers.Normal(scale = 0.1179, dtype=None)
        w4 = chainer.initializers.Normal(scale = 0.189, dtype=None)
        w5 = chainer.initializers.Normal(scale = 0.0001, dtype=None)
        super(SuperResolution_NN, self).__init__()

        # 全ての層を定義する
        with self.init_scope():
            # YUVのYのみ、1chのデータを入力と出力に使う
            # 16 x 16の画像データを入力すると、出力データは40 x 40になる
            self.c1 = L.Convolution2D( 1, 56, ksize = 5, stride = 1, pad = 0, initialW = w1)
            self.l1 = L.PReLU()
            self.c2 = L.Convolution2D( 56, 12, ksize = 1, stride = 1, pad = 0, initialW = w2)
            self.l2 = L.PReLU()
            self.c3 = L.Convolution2D( 12, 12, ksize = 3, stride = 1, pad = 1, initialW = w3)
            self.l3 = L.PReLU()
            self.c4 = L.Convolution2D( 12, 12, ksize = 3, stride = 1, pad = 1, initialW = w3)
            self.l4 = L.PReLU()
            self.c5 = L.Convolution2D( 12, 12, ksize = 3, stride = 1, pad = 1, initialW = w3)
            self.l5 = L.PReLU()
            self.c6 = L.Convolution2D( 12, 12, ksize = 3, stride = 1, pad = 1, initialW = w3)
            self.l6 = L.PReLU()
            self.c7 = L.Convolution2D( 12, 56, ksize = 1, stride = 1, pad = 1, initialW = w4)
            self.l7 = L.PReLU()
            self.c8 = L.Deconvolution2D( 56, 1, ksize = 9, stride = 3, pad = 4, initialW = w5)


    def __call__(self, x, t=None, train=True):
        h1 = self.l1(self.c1(x))
        h2 = self.l2(self.c2(h1))
        h3 = self.l3(self.c3(h2))
        h4 = self.l4(self.c4(h3))
        h5 = self.l5(self.c5(h4))
        h6 = self.l6(self.c6(h5))
        h7 = self.l7(self.c7(h6))
        h8 = self.c8(h7)

        # 損失か結果を返す
        return F.mean_squared_error(h8, t) if train else h8


# カスタムUpdaterのクラス
class SRUpdater(training.StandardUpdater):

    def __init__(self, train_iter, optimizer, device):
        super(SRUpdater, self).__init__(
            train_iter,
            optimizer,
            device=device
        )

    def update_core(self):
        # データを1パッチ分取得
        batch = self.get_iterator('main').next()
        # Optimizerを取得
        optimizer = self.get_optimizer('main')

        # パッチ分のデータを作る
        x_batch = [] # 入力データ
        y_batch = [] # 正解データ
        for img in batch:
            # 高解像度データ
            hpix = np.array(img, dtype=np.float32) / 255.0
            y_batch.append([hpix[:, :, 0]]) # YUVのYのみの1chデータ
            # 低解像度データ
            low = img.resize(( 16, 16), Image.NEAREST)
            lpix = np.array(low, dtype=np.float32) / 255.0
            x_batch.append([lpix[:, :, 0]]) # YUVのYのみの1chデータ


        # numpy or cupy配列にする
        x = cp.array(x_batch, dtype=cp.float32)
        y = cp.array(y_batch, dtype=cp.float32)

        # ニューラルネットワークを学習させる
        optimizer.update(optimizer.target, x, y)

# ニューラルネットワークを作成
model = SuperResolution_NN()

if uses_device >= 0:
    # GPUを使う
    chainer.cuda.get_device_from_id(0).use()
    chainer.cuda.check_cuda_available()
    # GPU用データ形式に変換
    model.to_gpu()


images = []

# すべてのファイル
fs = os.listdir('train')
for fn in fs:
    # スクレイピングした320 x 320画像を読み込み
    img = Image.open('train/' + fn).resize((320, 320)).convert('YCbCr')
    cur_x = 0
    while cur_x <= 320 - 40:
        cur_y = 0
        while cur_y <= 320 - 40:
            # オリジナル画像から40 x 40画像の切り出し
            rect = ( cur_x, cur_y, cur_x+40, cur_y+40)
            cropimg = img.crop(rect).copy()
            # 配列に追加
            images.append(cropimg)
            # 次の切り出し場所へ
            cur_y += 20
        cur_x += 20


# 繰り返し条件を作成する
train_iter = iterators.SerialIterator(images, batch_size, shuffle=True)

# 誤差逆伝播法アルゴリズムを選択する
optimizer = optimizers.Adam()
optimizer.setup(model)

# デバイスを選択してTrainerを作成する
updater = SRUpdater(train_iter, optimizer, device=uses_device)
trainer = training.Trainer(updater, (10000, 'epoch'), out="result")

# 学習の進展を表示するようにする
trainer.extend(extensions.ProgressBar())

# 中間結果を保存する
n_save = 0
# 1000 epochごとに割り込みをかける
@chainer.training.make_extension(trigger=(1000, 'epoch'))

def save_model(trainer):
    # NNのデータを保存
    global n_save
    n_save = n_save + 1
    chainer.serializers.save_hdf5( 'chapt03-'+str(n_save)+'.hdf5', model)

trainer.extend(save_model)

# 機械学習を実行する
trainer.run()

# 学習結果を保存する
chainer.serializers.save_hdf5( 'chapt03.hdf5', model)

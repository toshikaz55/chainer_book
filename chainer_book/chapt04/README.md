# Chainerによる画像の自動生成
## GANとDCGAN
GAN( Generative Adversarial Nets )
データを生成するタイプのニューラルネットワークを学習させるための手法
https://papers.nips.cc/paper/5423-generative-adversarial-nets

D-NetworkとG-Network、２つのニューラルネットワークを同時に学習させて、それぞれがお互いに相手の結果を学び合う
D-Networkはデータが本物（教師データと同じ）か、偽物（G-Networkによって生成されたデータ）かを判定する鑑定家
G-NetworkはD-Networkが本物と認識するようなデータを生成する贋作者

正しく学習が進むパラメータセットを発見して、GANの手法を適用できるようにしたDCGANがある
https://arxiv.org/abs/1511.06434


## chap04-1.py
### 鑑定家側D-Networkニューラルネットワークの作成

#### Batch Normalization
https://arxiv.org/abs/1502.03167

### 贋作者側G-Networkニューラルネットワークの作成


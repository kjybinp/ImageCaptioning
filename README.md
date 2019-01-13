# ImageCaptioning
chainer/exampleのchainerを勉強用にシンプル化したもの。マシーンスペックが足りず、batch sizeを確保できないかためか精度が出ない。

## 準備
- chainer, pycocotoolsのインストール
- [COCO](http://cocodataset.org/#download)から、「2014 Train images」「2014 Val images」「Train/Val annotations」をダウンロードして、main.py の"root_dir"でファイルのあるフォルダを指定。

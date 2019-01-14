import numpy as np

import chainer
import chainer.links as L
from chainer import training
from chainer.training import extensions
from chainer.datasets import TransformDataset

import dataset
import model

class EMBED_DATASET(chainer.dataset.DatasetMixin):
    def __init__(self, dataset, embed_image_file):
        #Ssuper(self, EMBED_DATASET).__init__()
        self.dataset = dataset
        self.embed_img = (np.load(embed_image_file)).astype('float32')
        self.size = self.embed_img.shape[0]
    def __len__(self):
        return self.size
    def get_example(self, i):
        image, caption = self.dataset.get_example(i)
        return self.embed_img[i], caption

def main():

    root_dir = 'dataset directory'
    EMBED = 1
    train, val = dataset.get_mscoco(root_dir)
    vocab_size = len(train.vocab)

    #imageを先にembedする場合

    def transform(x):
        image, caption = x
        image = L.model.vision.vgg.prepare(image)
        return image, caption

    train = TransformDataset(train, transform=transform)
    val = TransformDataset(val, transform=transform)

    if EMBED:
        train = EMBED_DATASET(train, 'train.npy')
        val = EMBED_DATASET(val, 'val.npy')
        net = model.LSTMLanguageModel(vocab_size)
    else:
        net = model.ImageCaptionModel(vocab_size)
    net.to_gpu()

    train_iter = chainer.iterators.MultiprocessIterator(train, batch_size=100, shared_mem=700000)
    val_iter = chainer.iterators.MultiprocessIterator(val, batch_size=20,repeat=False, shuffle=False, shared_mem=700000)

    optimizer = chainer.optimizers.Adam()
    optimizer.setup(net)

    def converter(batch, device):
        max_caption_length = 30
        return dataset.converter(batch, device, max_caption_length=max_caption_length)

    updater = training.updater.StandardUpdater(train_iter, optimizer=optimizer, device=0, converter=converter)

    trainer = training.Trainer(updater, out='result', stop_trigger=(50000, 'iteration'))
    trainer.extend(extensions.Evaluator(val_iter, target=net, converter=converter, device=0), trigger=(100, 'iteration'))
    trainer.extend(extensions.LogReport(['main/loss', 'validation/main/loss'],trigger=(10, 'iteration')))
    trainer.extend(extensions.PlotReport(['main/loss', 'validation/main/loss'],trigger=(10, 'iteration')))
    trainer.extend(extensions.PrintReport(
            ['elapsed_time', 'epoch', 'iteration', 'main/loss','validation/main/loss']), trigger=(10, 'iteration')
    )
    trainer.extend(extensions.snapshot_object(net, 'net_{.updater.iteration}'),trigger=(1000, 'iteration'))
    trainer.extend(extensions.ProgressBar())

    trainer.run()

if __name__ == '__main__':
    main()
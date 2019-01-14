import numpy as np
import math

import chainer
import chainer.links as L
from chainer.datasets import TransformDataset

import dataset
import model


def encode(dataset, output, batch_size, vggenc):
    dataset_input = np.zeros([len(dataset), 4096])
    print('make ' + output + ' start!!')
    print('dataset number: ' + str(int(len(dataset))))
    xp = vggenc.xp
    for i in range(math.ceil(len(dataset) / batch_size)):
        b = min(batch_size, len(dataset) - (i) * batch_size)
        x = np.zeros([b, 3, 224, 224], dtype='float32')
        for j in range(b):
            image, caption = dataset.get_example(i * batch_size + j)
            x[j, :, :, :] = image
        x = chainer.Variable(xp.array(x, dtype='float32'))
        x = vggenc(x)
        dataset_input[i * batch_size:i * batch_size + b, :] = chainer.cuda.to_cpu(x.data)
        if i % batch_size == 0 and i > 0:
            print(i * batch_size)
    np.save(output, dataset_input)


def main():
    root_dir = '/home/yawata/Desktop/work/data'
    train, val = dataset.get_mscoco(root_dir)
    vocab_size = len(train.vocab)
    batch_size = 10

    vggenc = model.VGGEncoder()
    vggenc.to_gpu()

    def transform(x):
        image, caption = x
        image = L.model.vision.vgg.prepare(image)
        return image, caption

    train = TransformDataset(train, transform=transform)
    val = TransformDataset(val, transform=transform)

    encode(val, 'val.npy', batch_size, vggenc)
    encode(train, 'train.npy', batch_size, vggenc)

if __name__ == '__main__':
    main()
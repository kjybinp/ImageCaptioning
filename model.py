import chainer
import chainer.links as L
import chainer.functions as F

class ImageCaptionModel(chainer.Chain):
    def __init__(self, vocab_size, hidden_size=512, dropout_ratio=0.5, ignore_label=-1):
        super(ImageCaptionModel, self).__init__()
        with self.init_scope():
            self.vggenc = VGGEncoder()
            self.lstm = LSTMLanguageModel(vocab_size, hidden_size, dropout_ratio, ignore_label)

    def forward(self, image, caption):
        image = chainer.Variable(image)
        img_feats = self.vggenc(image)
        loss = self.lstm(img_feats, caption)
        chainer.reporter.report({'loss': loss}, self)
        return loss

    def predict(self, imgs, bos, eos, max_caption_length):
        imgs = chainer.Variable(imgs)
        img_feats = self.feat_extractor(imgs)
        captions = self.lang_model.predict(
            img_feats, bos=bos, eos=eos, max_caption_length=max_caption_length)
        return captions

class VGGEncoder(chainer.Chain):
    def __init__(self):
        super(VGGEncoder, self).__init__()
        with self.init_scope():
            self.vgg = L.VGG16Layers()


    def forward(self, x):
        h = self.vgg(x, layers=['fc7'])['fc7']
        return h

class LSTMLanguageModel(chainer.Chain):
    def __init__(self, vocab_size, hidden_size=512, dropout_ratio=0.5, ignore_label=-1):
        super(LSTMLanguageModel, self).__init__()
        with self.init_scope():
            self.embed_word = L.EmbedID(
                vocab_size,
                hidden_size,
                initialW=chainer.initializers.Normal(1.0),
                ignore_label=ignore_label
            )
            self.embed_img = L.Linear(
                hidden_size,
                initialW=chainer.initializers.Normal(0.01)
            )
            self.lstm = L.LSTM(hidden_size, hidden_size)
            self.out_word = L.Linear(
                hidden_size,
                vocab_size,
                initialW=chainer.initializers.Normal(0.01)
            )

        self.dropout_ratio = dropout_ratio

    def forward(self, img_feats, captions):
        self.reset(img_feats)

        loss = 0
        size = 0
        caption_length = captions.shape[1]
        for i in range(caption_length - 1):
            x = chainer.Variable(self.xp.asarray(captions[:, i]))
            t = chainer.Variable(self.xp.asarray(captions[:, i + 1]))
            if (t.array == self.embed_word.ignore_label).all():
                break
            y = self.step(x)
            loss += F.softmax_cross_entropy(
                y, t, ignore_label=self.embed_word.ignore_label)
            size += 1
        loss = loss / max(size, 1)
        chainer.reporter.report({'loss': loss}, self)
        return loss

    def reset(self, img_feats):
        self.lstm.reset_state()
        h = self.embed_img(img_feats)
        h = self.lstm(F.dropout(h, ratio=self.dropout_ratio))
        return h

    def step(self, x):
        h = self.embed_word(x)
        h = self.lstm(F.dropout(h, ratio=self.dropout_ratio))
        h = self.out_word(F.dropout(h, ratio=self.dropout_ratio))
        return h

    def predict(self, img_feats, bos, eos, max_caption_length):
        self.reset(img_feats)

        captions = self.xp.full((img_feats.shape[0], 1), bos, dtype=np.int32)
        for _ in range(max_caption_length):
            x = chainer.Variable(captions[:, -1])  # Previous word token as input
            y = self.step(x)
            pred = y.array.argmax(axis=1).astype(np.int32)
            captions = self.xp.hstack((captions, pred[:, None]))
            if (pred == eos).all():
                break
        return captions
import chainer
import chainer.links as L
import chainer.functions as F

import grl


class AlexDANN(chainer.Chain):
    """ input_size:(batch, 3ch, 227, 227) """

    iter_count = 0

    def __init__(self, class_num=31):
        super(AlexDANN, self).__init__()
        with self.init_scope():
            # --- Feature Extractor --- #
            self.conv1 = L.Convolution2D(3, 96, 11, stride=4)
            self.conv2 = L.Convolution2D(96, 256, 5, pad=2)
            self.conv3 = L.Convolution2D(256, 384, 3, pad=1)
            self.conv4 = L.Convolution2D(384, 384, 3, pad=1)
            self.conv5 = L.Convolution2D(384, 256, 3, pad=1)

            # --- Label Predictor --- #
            self.lp6 = L.Linear(256 * 6 * 6, 4096)
            self.lp7 = L.Linear(4096, 4096)
            self.lp8 = L.Linear(4096, class_num)

            # --- Domain Classifier --- #
            self.dc6 = L.Linear(256 * 6 * 6, 1024)
            self.dc7 = L.Linear(1024, 1024)
            self.dc8 = L.Linear(1024, 1)

    # Forward for Inference Classification (fe + lp)
    def __call__(self, x):
        with chainer.using_config('train', False), chainer.no_backprop_mode():
            return self.forward_lp(self.forward_fe(x))

    # Forward Feature Extractor (Convolution Layers)
    def forward_fe(self, x):
        h1 = F.local_response_normalization(F.relu(self.conv1(x)))
        h1 = F.max_pooling_2d(h1, 3, stride=2)
        h2 = F.local_response_normalization(F.relu(self.conv2(h1)))
        h2 = F.max_pooling_2d(h2, 3, stride=2)
        h3 = F.relu(self.conv3(h2))
        h4 = F.relu(self.conv4(h3))
        h5 = F.max_pooling_2d(F.relu(self.conv5(h4)), 3, stride=2)
        return h5

    # Forward Label Prediction (Full Connect Layers)
    def forward_lp(self, h):
        h6 = F.dropout(F.relu(self.lp6(h)))
        h7 = F.dropout(F.relu(self.lp7(h6)))
        h8 = self.lp8(h7)
        return h8

    # Forward Domain Classification with GRL (Full Connect Layers)
    def forward_dc(self, h):
        self.iter_count += 1
        h5 = grl.flip_grad(h, self.iter_count)
        h6 = F.dropout(F.relu(self.dc6(h5)))
        h7 = F.dropout(F.relu(self.dc7(h6)))
        h8 = self.dc8(h7)
        return h8

    # Forward (fe + lp) and (fe + dc)
    def forward_training(self, src, tgt):
        x = F.concat([src, tgt], axis=0)
        h = self.forward_fe(x)
        h_src, h_tgt = F.split_axis(x=h, indices_or_sections=2, axis=0)
        h_lp = self.forward_lp(h_src)
        h_dc = self.forward_dc(h)
        return h_lp, h_dc


class Mnist2MnistM(chainer.Chain):
    """ input_size:(batch, 3ch, 32, 32) """

    iter_count = 0

    def __init__(self, class_num=10):
        super(Mnist2MnistM, self).__init__()
        with self.init_scope():
            # --- Feature Extractor --- #
            self.conv1 = L.Convolution2D(in_channels=3, out_channels=32, ksize=5)
            self.conv2 = L.Convolution2D(in_channels=32, out_channels=48, ksize=5)

            # --- Label Predictor --- #
            self.lp3 = L.Linear(in_size=48 * 5 * 5, out_size=100)
            self.lp4 = L.Linear(in_size=100, out_size=100)
            self.lp5 = L.Linear(in_size=100, out_size=class_num)

            # --- Domain Classifier --- #
            self.dc3 = L.Linear(in_size=48 * 5 * 5, out_size=100)
            self.dc4 = L.Linear(in_size=100, out_size=1)

    # Forward for Inference Classification (fe + lp)
    def __call__(self, x):
        with chainer.using_config('train', False), chainer.no_backprop_mode():
            return self.forward_lp(self.forward_fe(x))

    # Forward Feature Extractor (Convolution Layers)
    def forward_fe(self, x):
        h = F.max_pooling_2d(F.relu(self.conv1(x)), ksize=2, stride=2)
        h = F.max_pooling_2d(F.relu(self.conv2(h)), ksize=2, stride=2)
        return h

    # Forward Label Prediction (Full Connect Layers)
    def forward_lp(self, h):
        h = F.dropout(F.relu(self.lp3(h)))
        h = F.dropout(F.relu(self.lp4(h)))
        h = F.relu(self.lp5(h))
        return h

    # Forward Domain Classification with GRL (Full Connect Layers)
    def forward_dc(self, h):
        self.iter_count += 1
        h = grl.flip_grad(h, self.iter_count)
        h = F.dropout(F.relu(self.dc3(h)))
        h = self.dc4(h)
        return h

    # Forward (fe + lp) and (fe + dc)
    def forward_training(self, src, tgt):
        x = F.concat([src, tgt], axis=0)
        h = self.forward_fe(x)
        h_src, h_tgt = F.split_axis(x=h, indices_or_sections=2, axis=0)
        h_lp = self.forward_lp(h_src)
        h_dc = self.forward_dc(h)
        return h_lp, h_dc

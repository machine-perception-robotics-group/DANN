import numpy as np

import chainer
from chainer import Variable
import chainer.functions as F


class DANNUpdater(chainer.training.StandardUpdater):
    def __init__(self, source_iterator, target_iterator, model, optimizer, device=-1):
        self.device_id = device
        self.model = model

        iterators = {'main': source_iterator, 'target': target_iterator}
        opt = {'main': optimizer}
        super().__init__(iterators, opt, device=self.device_id)

    def get_source_batch(self):
        batch = self.get_iterator('main').next()
        batch, labels = chainer.dataset.concat_examples(batch, device=self.device_id)
        return Variable(batch), Variable(labels)

    def get_target_batch(self):
        batch = self.get_iterator('target').next()
        batch, labels = chainer.dataset.concat_examples(batch, device=self.device_id)
        return Variable(batch), Variable(labels)

    def domain_label(self, batch, dc_label_id):
        d = np.ones(len(batch), dtype=np.int32) * dc_label_id
        d = chainer.dataset.concat_examples(d, device=self.device_id)
        return Variable(d)

    # Overrides from chainer.training.StandardUpdater
    def update_core(self):
        # Fetch Optimizer
        optim = self.get_optimizer('main')

        # Get Batch and LP Class Label
        batch_src, label_lp_src = self.get_source_batch()
        batch_tgt, label_lp_tgt = self.get_target_batch()

        # Set DC Class Label
        label_dc_src = self.domain_label(batch_src, 0)
        label_dc_tgt = self.domain_label(batch_tgt, 1)
        label_dc = F.concat([label_dc_src, label_dc_tgt], axis=0)

        # Forward Network
        h_lp, h_dc = self.model.forward_training(batch_src, batch_tgt)

        # Calc loss
        loss_lp = F.softmax_cross_entropy(h_lp, label_lp_src)
        h_dc = F.reshape(h_dc, (-1,))  # Adjust shape
        loss_dc = F.sigmoid_cross_entropy(h_dc, label_dc)
        loss = loss_lp + loss_dc

        # Update Network
        self.model.cleargrads()
        loss.backward()
        optim.update()

        # Report training data
        chainer.reporter.report(
            {"train/loss/LP": loss_lp,
             "train/loss/DC": loss_dc,
             "train/loss/total": loss,
             "train/accuracy": F.accuracy(F.softmax(h_lp), label_lp_src)
             })

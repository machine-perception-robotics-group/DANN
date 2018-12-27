# coding:utf-8
# Unsupervised Domain Adaptation by Backpropagation (in ICML'15)
# Project Page : http://sites.skoltech.ru/compvision/projects/grl/

import os
from glob import glob

import chainer
from chainer import iterators, optimizers
from chainer.training import extensions as ex

import model
from updater import DANNUpdater
from data import DataSet

if __name__ == '__main__':
    # --- Prepare Learning --- #
    size = (32, 32)
    class_num = 10
    batch = 64
    lr = 0.01
    gpu_id = 0
    epochs = 300
    result_dir = 'Result'

    data_root = 'DownloadedPlace/Mnist2MnistM'
    source_path = os.path.join(data_root, 'mnist/train/[0-9]/*.png')
    target_path = os.path.join(data_root, 'mnistM/train/[0-9]/*.png')
    valid_path = os.path.join(data_root, 'mnistM/test/[0-9]/*.png')

    # --- Set Path List --- #
    source_path_list = list(map(lambda f: f.replace(os.sep, '/'), glob(source_path)))
    target_path_list = list(map(lambda f: f.replace(os.sep, '/'), glob(target_path)))
    valid_path_list = list(map(lambda f: f.replace(os.sep, '/'), glob(valid_path)))
    print('source_path_list length :{:>6}'.format(len(source_path_list)))
    print('target_path_list length :{:>6}'.format(len(target_path_list)))
    print('valid_path_list length  :{:>6}'.format(len(valid_path_list)))

    # --- Create Data Object --- #
    source_dataset = DataSet(source_path_list, size)
    target_dataset = DataSet(target_path_list, size)
    valid_dataset = DataSet(valid_path_list, size)

    # --- Create iterators for Training and Validation --- #
    # Select MultiprocessIterator or SerialIterator
    source = iterators.SerialIterator(source_dataset, batch, repeat=True, shuffle=True)
    target = iterators.SerialIterator(target_dataset, batch, repeat=True, shuffle=True)
    valid = iterators.SerialIterator(valid_dataset, batch, repeat=False, shuffle=False)

    # --- Create DANN model --- #
    model = model.Mnist2MnistM(class_num=class_num)

    # --- Set Optimizer --- #
    opt = optimizers.MomentumSGD(lr=lr)
    opt.setup(model)

    # --- Set Trainer --- #
    updater = DANNUpdater(source, target, model, opt, gpu_id)
    trainer = chainer.training.Trainer(updater, (epochs, 'epoch'), out=result_dir)

    # --- Set Extensions --- #
    # Setting for Extensions
    print_list = ['epoch', 'train/loss/LP', 'train/loss/DC', 'train/loss/total', 'validation/main/loss',
                  'GRL/lmd', 'lr', 'train/accuracy', 'validation/main/accuracy', 'elapsed_time']
    loss_list = ['train/loss/LP', 'train/loss/DC', 'train/loss/total', 'validation/main/loss']
    accuracy_list = ['train/accuracy', 'validation/main/accuracy']
    eval_model = chainer.links.Classifier(model)

    # Set Extensions
    trainer.extend(ex.Evaluator(valid, eval_model, device=0))
    trainer.extend(ex.dump_graph(root_name='train/loss/total', out_name='cg.dot'))  # calc graph
    trainer.extend(ex.observe_lr())
    trainer.extend(ex.LogReport())
    trainer.extend(ex.LinearShift('lr', (lr, lr / 100.0), (30000, 100000)))
    trainer.extend(ex.PrintReport(print_list))
    trainer.extend(ex.ProgressBar(update_interval=1))
    trainer.extend(ex.PlotReport(loss_list, 'epoch', file_name='loss.png'))
    trainer.extend(ex.PlotReport(accuracy_list, 'epoch', file_name='accuracy.png'))

    # --- Start Training! --- #
    trainer.run()

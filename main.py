from models import unet
import torch
import torchvision

from torch.utils.data.dataloader import DataLoader

import sys
import os
import logging
import random
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import gflags
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset

from trainer import Trainer, TrainerConfig


FLAGS = gflags.FLAGS

gflags.DEFINE_boolean('train', True, '')
gflags.DEFINE_boolean('sample', False, '')
gflags.DEFINE_string('ckpt_path', 'cifar10_model.pt',
                     'The checkpoint to save to during training, or the one to use during sampling.')


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main():
    set_seed(33)

    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    root = './'
    train_dataset = torchvision.datasets.CIFAR10(
        root,
        train=True,
        transform=transform,
        target_transform=None,
        download=True)
    test_dataset = torchvision.datasets.CIFAR10(
        root,
        train=False,
        transform=transform,
        target_transform=None,
        download=True)
    print('Loaded the data.')

    # TODO: save and load the model and optimizer

    print('Building model...')
    model = unet.UNet()

    # initialize a trainer instance and kick off training
    tconf = TrainerConfig(max_epochs=20, batch_size=64, learning_rate=3e-3,
                          betas=(0.9, 0.95), weight_decay=0,
                          lr_decay=False, ckpt_path=FLAGS.ckpt_path, num_workers=8)
    trainer = Trainer(model, train_dataset, test_dataset, tconf)

    if FLAGS.train:
        trainer.train()

    if FLAGS.sample:
        trainer.sample()


if __name__ == '__main__':
    FLAGS(sys.argv)
    main()

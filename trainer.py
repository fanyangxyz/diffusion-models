"""
Simple training loop; Boilerplate that could apply to any arbitrary neural network.
"""

import math
import logging
import os

from tqdm import tqdm
import numpy as np

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data.dataloader import DataLoader

logger = logging.getLogger(__name__)


class TrainerConfig:
    # sampling settings
    T = 1000
    # optimization parameters
    max_epochs = 10
    batch_size = 2  # 64
    learning_rate = 3e-4
    betas = (0.9, 0.95)
    grad_norm_clip = 1.0
    weight_decay = 0.1
    lr_decay = False
    # checkpoint settings
    ckpt_path = None
    num_workers = 0  # for DataLoader

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


class Trainer:

    def __init__(self, model, train_dataset, test_dataset, config):
        self.model = model
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.config = config
        raw_model = model.module if hasattr(model, "module") else model
        self.optimizer = raw_model.configure_optimizers(config)

        # TODO: register these values
        self.beta = torch.linspace(1e-4, 0.02, self.config.T)
        self.alpha = 1 - self.beta
        self.alpha_bar = torch.cumprod(self.alpha, 0)

        self.train_loader = DataLoader(
            train_dataset,
            shuffle=True,
            pin_memory=True,
            batch_size=config.batch_size,
            num_workers=config.num_workers
        )
        if test_dataset is not None:
            self.test_loader = DataLoader(
                self.test_dataset,
                shuffle=True,
                pin_memory=True,
                batch_size=config.batch_size,
                num_workers=config.num_workers
            )
        else:
            self.test_loader = None

        # take over whatever gpus are on the system
        self.device = 'cpu'
        if torch.cuda.is_available():
            self.device = torch.cuda.current_device()
            self.model = torch.nn.DataParallel(self.model).to(self.device)

    def save_checkpoint(self):
        # DataParallel wrappers keep raw model object in .module attribute
        raw_model = self.model.module if hasattr(
            self.model, "module") else self.model
        logger.info("saving %s", self.config.ckpt_path)
        torch.save(raw_model.state_dict(), self.config.ckpt_path)

    def train(self):
        best_loss = float('inf')
        for epoch in range(self.config.max_epochs):
            self.run_epoch(epoch, self.train_loader, is_train=True)
            if self.test_dataset is not None:
                test_loss = run_epoch(self.test_loader, is_train=False)

            # supports early stopping based on the test loss, or just save
            # always if no test set is provided
            good_model = self.test_dataset is None or test_loss < best_loss
            if self.config.ckpt_path is not None and good_model:
                best_loss = test_loss
                self.save_checkpoint()nn.MSELoss

    def run_epoch(self, epoch, loader, is_train):
        self.model.train(is_train)

        losses = []
        pbar = tqdm(enumerate(loader), total=len(loader)
                    ) if is_train else enumerate(loader)
        for it, (x, _) in pbar:

            # sample the noise and compute the noisy image
            batch_size = x.shape[0]
            y = torch.randn_like(x)
            t = torch.randint(low=0, high=self.config.T, size=(batch_size,))
            alpha_bar = torch.gather(
                self.alpha_bar, 0, t).view(
                (batch_size, 1, 1, 1))
            x = torch.sqrt(alpha_bar) * x + torch.sqrt(1 - alpha_bar) * y

            # place data on the correct device
            x = x.to(self.device)
            y = y.to(self.device)

            # forward the model
            with torch.set_grad_enabled(is_train):
                loss = self.model(x, y)
                loss = loss.mean()  # collapse all losses if they are scattered on multiple gpus
                losses.append(loss.item())

            if is_train:
                # backprop and update the parameters
                self.model.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config.grad_norm_clip)
                self.optimizer.step()

                # decay the learning rate based on our progress
                if self.config.lr_decay:
                    raise NotImplementedError
                else:
                    lr = self.config.learning_rate

                # report progress
                pbar.set_description(
                    f"epoch {epoch+1} iter {it}: train loss {loss.item():.5f}. lr {lr:e}")

        if not is_train:
            test_loss = float(np.mean(losses))
            logger.info("test loss: %f", test_loss)
            return test_loss

    def sample(self):
        raise NotImplementedError

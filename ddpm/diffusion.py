import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from functools import partial
from copy import deepcopy

from .ema import EMA
from .utils import extract


class GaussianDiffusion(nn.Module):
    __doc__ = r"""Gaussian Diffusion model. Forwarding through the module returns diffusion reversal scalar loss tensor.

    Input:
        x: tensor of shape (N, img_channels, *img_size)
        y: tensor of shape (N)
    Output:
        scalar loss tensor
    Args:
        model (nn.Module): model which estimates diffusion noise
        img_size (tuple): image size tuple (H, W)
        img_channels (int): number of image channels
        betas (np.ndarray): numpy array of diffusion betas
        loss_type (string): loss type, "l1" or "l2"
        ema_decay (float): model weights exponential moving average decay
        ema_start (int): number of steps before EMA
        ema_update_rate (int): number of steps before each EMA update
    """

    def __init__(
        self,
        model,
        img_size,
        img_channels,
        num_classes,
        betas,
        loss_type="l2",
        ema_decay=0.9999,
        ema_start=5000,
        ema_update_rate=1,
    ):
        super().__init__()

        self.model = model
        self.ema_model = deepcopy(model)

        self.ema = EMA(ema_decay)
        self.ema_decay = ema_decay
        self.ema_start = ema_start
        self.ema_update_rate = ema_update_rate
        self.step = 0

        self.img_size = img_size
        self.img_channels = img_channels
        self.num_classes = num_classes

        if loss_type not in ["l1", "l2"]:
            raise ValueError("__init__() got unknown loss type")

        self.loss_type = loss_type
        self.num_timesteps = len(betas)

        alphas = 1.0 - betas
        alphas_cumprod = np.cumprod(alphas)
        alphas_cumprod_prev = np.append(1.0, alphas_cumprod[:-1])
        sqrt_recip_alphas_cumprod = np.sqrt(1.0 / alphas_cumprod)
        sqrt_recipm1_alphas_cumprod = np.sqrt(1.0 / alphas_cumprod - 1)

        to_torch = partial(torch.tensor, dtype=torch.float32)

        self.register_buffer("betas", to_torch(betas))
        self.register_buffer("alphas", to_torch(alphas))
        self.register_buffer("alphas_cumprod", to_torch(alphas_cumprod))
        self.register_buffer(
            "alphas_cumprod_prev",
            to_torch(alphas_cumprod_prev))
        self.register_buffer(
            "sqrt_recip_alphas_cumprod",
            to_torch(sqrt_recip_alphas_cumprod))
        self.register_buffer(
            "sqrt_recipm1_alphas_cumprod",
            to_torch(sqrt_recipm1_alphas_cumprod))

        self.register_buffer("sqrt_alphas_cumprod",
                             to_torch(np.sqrt(alphas_cumprod)))
        self.register_buffer("sqrt_one_minus_alphas_cumprod",
                             to_torch(np.sqrt(1 - alphas_cumprod)))
        self.register_buffer("reciprocal_sqrt_alphas",
                             to_torch(np.sqrt(1 / alphas)))

        self.register_buffer("remove_noise_coeff", to_torch(
            betas / np.sqrt(1 - alphas_cumprod)))
        self.register_buffer("sigma", to_torch(np.sqrt(betas)))

    def update_ema(self):
        raw_model = self.model.module if hasattr(
            self.model, "module") else self.model
        raw_ema_model = self.ema_model.module if hasattr(
            self.ema_model, "module") else self.ema_model
        self.step += 1
        if self.step % self.ema_update_rate == 0:
            if self.step < self.ema_start:
                raw_ema_model.load_state_dict(raw_model.state_dict())
            else:
                self.ema.update_model_average(raw_ema_model, raw_model)

    @torch.no_grad()
    def remove_noise(self, x, t, y, use_ema=True):
        model = self.ema_model if use_ema else self.model
        return ((x - extract(self.remove_noise_coeff, t, x.shape) *
                model(x, t, y)) * extract(self.reciprocal_sqrt_alphas, t, x.shape))

    @torch.no_grad()
    def _predict_xstart_from_eps(self, x_t, t, eps):
        assert x_t.shape == eps.shape
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * eps
        )

    @torch.no_grad()
    def ddim_sample(self, x, t, y, use_ema=True, eta=0):
        model = self.ema_model if use_ema else self.model
        eps = model(x, t, y)
        pred_xstart = self._predict_xstart_from_eps(x, t, eps)
        alpha_bar = extract(self.alphas_cumprod, t, x.shape)
        alpha_bar_prev = extract(self.alphas_cumprod_prev, t, x.shape)
        sigma = (
            eta
            * torch.sqrt((1 - alpha_bar_prev) / (1 - alpha_bar))
            * torch.sqrt(1 - alpha_bar / alpha_bar_prev)
        )
        # Equation 12.
        noise = torch.randn_like(x)
        mean_pred = (
            pred_xstart * torch.sqrt(alpha_bar_prev)
            + torch.sqrt(1 - alpha_bar_prev - sigma ** 2) * eps
        )
        nonzero_mask = (
            (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
        )  # no noise when t == 0
        sample = mean_pred + nonzero_mask * sigma * noise
        return sample

    @torch.no_grad()
    def sample(self, batch_size, device, y=None,
               use_ema=True, use_ddim=False, return_sequence=False):
        if y is not None and batch_size != len(y):
            raise ValueError(
                "sample batch size different from length of given y")

        x = torch.randn(batch_size, self.img_channels,
                        *self.img_size, device=device)
        diffusion_sequence = [x.cpu().detach()]

        # DDIM is not working yet. Need spaced.
        num_timesteps = self.num_timesteps  # if not use_ddim else 250

        for t in range(num_timesteps - 1, -1, -1):
            t_batch = torch.tensor([t], device=device).repeat(batch_size)
            if not use_ddim:
                x = self.remove_noise(x, t_batch, y, use_ema)
                if t > 0:
                    x += extract(self.sigma, t_batch, x.shape) * \
                        torch.randn_like(x)
            else:
                x = self.ddim_sample(x, t_batch, y, use_ema)

            if return_sequence:
                diffusion_sequence.append(x.cpu().detach())

        if return_sequence:
            return diffusion_sequence

        return x.cpu().detach()

    def perturb_x(self, x, t, noise):
        return (
            extract(self.sqrt_alphas_cumprod, t, x.shape) * x +
            extract(self.sqrt_one_minus_alphas_cumprod, t, x.shape) * noise
        )

    def get_losses(self, x, t, y):
        noise = torch.randn_like(x)

        perturbed_x = self.perturb_x(x, t, noise)
        estimated_noise = self.model(perturbed_x, t, y)

        if self.loss_type == "l1":
            loss = F.l1_loss(estimated_noise, noise)
        elif self.loss_type == "l2":
            loss = F.mse_loss(estimated_noise, noise)

        return loss

    def forward(self, x, y=None):
        b, c, h, w = x.shape
        device = x.device

        if h != self.img_size[0]:
            raise ValueError(
                "image height does not match diffusion parameters")
        if w != self.img_size[0]:
            raise ValueError("image width does not match diffusion parameters")

        t = torch.randint(0, self.num_timesteps, (b,), device=device)
        return self.get_losses(x, t, y)


def generate_cosine_schedule(T, s=0.008):
    def f(t, T):
        return (np.cos((t / T + s) / (1 + s) * np.pi / 2)) ** 2

    alphas = []
    f0 = f(0, T)

    for t in range(T + 1):
        alphas.append(f(t, T) / f0)

    betas = []

    for t in range(1, T + 1):
        betas.append(min(1 - alphas[t] / alphas[t - 1], 0.999))

    return np.array(betas)


def generate_linear_schedule(T, low, high):
    return np.linspace(low, high, T)

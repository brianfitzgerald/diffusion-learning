from inspect import isfunction
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import (
    Compose,
    ToTensor,
    Lambda,
    ToPILImage,
    CenterCrop,
    Resize,
    RandomHorizontalFlip,
    Normalize
)
import numpy as np
from tqdm import tqdm
from dataclasses import dataclass


def cosine_beta_schedule(timesteps, s=0.008):
    """
    cosine schedule as proposed in https://arxiv.org/abs/2102.09672
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)

def linear_beta_schedule(timesteps):
    beta_start = 0.0001
    beta_end = 0.02
    return torch.linspace(beta_start, beta_end, timesteps)

def quadratic_beta_schedule(timesteps):
    beta_start = 0.0001
    beta_end = 0.02
    return torch.linspace(beta_start**0.5, beta_end**0.5, timesteps) ** 2

def sigmoid_beta_schedule(timesteps):
    beta_start = 0.0001
    beta_end = 0.02
    betas = torch.linspace(-6, 6, timesteps)
    return torch.sigmoid(betas) * (beta_end - beta_start) + beta_start


# extract data at t indices from a
def extract(a, t, x_shape):
    batch_size = t.shape[0]
    out = a.gather(-1, t.cpu())
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)


@dataclass
class Scheduler:
    sqrt_recip_alphas: float
    posterior_variance: float
    sqrt_one_minus_alphas_cumprod: float
    sqrt_alphas_cumprod: float
    timesteps: float
    betas: float


def get_scheduler(timesteps: int = 100):
    # define beta schedule
    # define beta schedule
    betas = linear_beta_schedule(timesteps)

    # define alphas
    alphas = 1. - betas
    alphas_cumprod = torch.cumprod(alphas, axis=0)
    alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
    sqrt_recip_alphas = torch.sqrt(1.0 / alphas)

    # calculations for diffusion q(x_t | x_{t-1}) and others
    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)

    # calculations for posterior q(x_{t-1} | x_t, x_0)
    posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
    return Scheduler(
        sqrt_recip_alphas,
        posterior_variance,
        sqrt_one_minus_alphas_cumprod,
        sqrt_alphas_cumprod,
        timesteps,
        betas
    )


def get_transforms(image_size: int = 128):
    inference_transform = Compose(
        [
            Resize(image_size),
            CenterCrop(image_size),
            ToTensor(),  # turn into torch Tensor of shape CHW, divide by 255
            Lambda(lambda t: (t * 2) - 1),
        ]
    )

    reverse_transform = Compose(
        [
            Lambda(lambda t: (t + 1) / 2),
            Lambda(lambda t: t.permute(1, 2, 0)),  # CHW to HWC
            Lambda(lambda t: t * 255.0),
            Lambda(lambda t: t.numpy().astype(np.uint8)),
            ToPILImage(),
        ]
    )

    # define image transformations (e.g. using torchvision)
    dataset_transform = Compose(
        [
            RandomHorizontalFlip(),
            ToTensor(),
            Normalize([0.5], [0.5])
        ]
    )
    return inference_transform, reverse_transform, dataset_transform


@torch.no_grad()
def p_sample(model, scheduler: Scheduler, x, t, t_index):
    betas_t = extract(scheduler.betas, t, x.shape)
    sqrt_one_minus_alphas_cumprod_t = extract(
        scheduler.sqrt_one_minus_alphas_cumprod, t, x.shape
    )
    sqrt_recip_alphas_t = extract(scheduler.sqrt_recip_alphas, t, x.shape)

    # Equation 11 in the paper
    # Use our model (noise predictor) to predict the mean
    model_mean = sqrt_recip_alphas_t * (
        x - betas_t * model(x, t) / sqrt_one_minus_alphas_cumprod_t
    )

    if t_index == 0:
        return model_mean
    else:
        posterior_variance_t = extract(scheduler.posterior_variance, t, x.shape)
        noise = torch.randn_like(x)
        # Algorithm 2 line 4:
        return model_mean + torch.sqrt(posterior_variance_t) * noise


# Algorithm 2 (including returning all images)
# Sample all timesteps in a loop - used for sampling at inference
@torch.no_grad()
def p_sample_loop(model, scheduler: Scheduler, shape, timesteps: int):
    device = next(model.parameters()).device

    b = shape[0]
    # start from pure noise (for each example in the batch)
    img = torch.randn(shape, device=device)
    imgs = []

    for i in tqdm(
        reversed(range(0, timesteps)), desc="sampling loop time step", total=timesteps
    ):
        img = p_sample(
            model,
            scheduler,
            img,
            torch.full((b,), i, device=device, dtype=torch.long),
            i,
        )
        imgs.append(img.cpu())
    return imgs


@torch.no_grad()
def sample(model, scheduler: Scheduler, image_size, batch_size=16, channels=3):
    res = p_sample_loop(
        model,
        scheduler,
        (batch_size, channels, image_size, image_size),
        scheduler.timesteps,
    )
    return res[0]


def num_to_groups(num, divisor):
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr


def q_sample(scheduler: Scheduler, x_start, t, noise=None):
    if noise is None:
        noise = torch.randn_like(x_start)

    sqrt_alphas_cumprod_t = extract(scheduler.sqrt_alphas_cumprod, t, x_start.shape)
    sqrt_one_minus_alphas_cumprod_t = extract(
        scheduler.sqrt_one_minus_alphas_cumprod, t, x_start.shape
    )

    return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise


def p_losses(
    denoise_model, scheduler: Scheduler, x_start, t, noise=None, loss_type="l1"
):
    if noise is None:
        noise = torch.randn_like(x_start)

    x_noisy = q_sample(scheduler, x_start, t, noise)
    predicted_noise = denoise_model(x_noisy, t)

    if loss_type == "l1":
        loss = F.l1_loss(noise, predicted_noise)
    elif loss_type == "l2":
        loss = F.mse_loss(noise, predicted_noise)
    elif loss_type == "huber":
        loss = F.smooth_l1_loss(noise, predicted_noise)
    else:
        raise NotImplementedError()

    return loss

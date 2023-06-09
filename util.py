import os
import imageio
import numpy as np
from typing import Union
import logging
import torch
import torchvision
from tqdm import tqdm
from einops import rearrange
import datetime
import pprint

def available_devices(threshold=5000,n_devices=None):
    """
    search for available GPU devices
    Args:
        threshold: the devices with larger memory than threshold is available
        n_devices: the number of devices
    Returns:
        device: the id for available GPU devices
    """
    memory = list(os.popen('nvidia-smi --query-gpu=memory.free --format=csv,nounits,noheader'))
    mem = [int(x.strip()) for x in memory]
    devices = []
    for i in range(len(mem)):
        if mem[i] > threshold:
            devices.append(i)
    device = devices if n_devices is None else devices[:n_devices]
    return device

def format_devices(devices):
    if isinstance(devices, list):
        return ','.join(map(str,devices))

def backup_profile(profile: dict, path):
    """
    backup args profile
    Args:
        profile: the args profile need to backup code
        path: the path for saving args profile
    """
    os.makedirs(path, exist_ok=True)
    path = os.path.join(path, "profile_{}.txt".format(datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")))
    s = pprint.pformat(profile)
    with open(path, 'w') as f:
        f.write(s)

def set_logger(path, file_path=None):
    os.makedirs(path,exist_ok=True)
    #logger to print information
    logger = logging.getLogger()
    logger.setLevel(level=logging.INFO)
    handler1 = logging.StreamHandler()
    if file_path is not None:
        handler2 = logging.FileHandler(os.path.join(path,file_path), mode='w')
    else:
        handler2 = logging.FileHandler(os.path.join(path, "logs.txt"), mode='w')
    formatter = logging.Formatter("%(asctime)s - %(message)s")
    handler1.setFormatter(formatter)
    handler2.setFormatter(formatter)
    logger.addHandler(handler1)
    logger.addHandler(handler2)

def save_tensor_images_folder(videos: torch.Tensor, path: str, rescale=False, n_rows=4):
    os.makedirs(path, exist_ok=True)
    videos = rearrange(videos, "b c t h w -> t b c h w")
    outputs = []
    for i, x in enumerate(videos):
        x = torchvision.utils.make_grid(x, nrow=n_rows)
        x = x.transpose(0, 1).transpose(1, 2).squeeze(-1)
        if rescale:
            x = (x + 1.0) / 2.0  # -1,1 -> 0,1
        x = (x * 255).numpy().astype(np.uint8)
        save_path = os.path.join(path, f"{i}.png")
        imageio.imsave(save_path, x)
        outputs.append(x)

def save_videos_grid(videos: torch.Tensor, path: str, rescale=False, n_rows=4, fps=8):
    videos = rearrange(videos, "b c t h w -> t b c h w")
    outputs = []
    for x in videos:
        x = torchvision.utils.make_grid(x, nrow=n_rows)
        x = x.transpose(0, 1).transpose(1, 2).squeeze(-1)
        if rescale:
            x = (x + 1.0) / 2.0  # -1,1 -> 0,1
        x = (x * 255).numpy().astype(np.uint8)
        outputs.append(x)

    os.makedirs(os.path.dirname(path), exist_ok=True)
    imageio.mimsave(path, outputs, fps=fps)

# DDIM Inversion
@torch.no_grad()
def init_prompt(prompt, pipeline):
    uncond_input = pipeline.tokenizer(
        [""], padding="max_length", max_length=pipeline.tokenizer.model_max_length,
        return_tensors="pt"
    )
    uncond_embeddings = pipeline.text_encoder(uncond_input.input_ids.to(pipeline.device))[0]
    text_input = pipeline.tokenizer(
        [prompt],
        padding="max_length",
        max_length=pipeline.tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    text_embeddings = pipeline.text_encoder(text_input.input_ids.to(pipeline.device))[0]
    context = torch.cat([uncond_embeddings, text_embeddings])
    return context

def next_step(model_output: Union[torch.FloatTensor, np.ndarray], timestep: int,
              sample: Union[torch.FloatTensor, np.ndarray], ddim_scheduler):
    timestep, next_timestep = min(
        timestep - ddim_scheduler.config.num_train_timesteps // ddim_scheduler.num_inference_steps, 999), timestep
    alpha_prod_t = ddim_scheduler.alphas_cumprod[timestep] if timestep >= 0 else ddim_scheduler.final_alpha_cumprod
    alpha_prod_t_next = ddim_scheduler.alphas_cumprod[next_timestep]
    beta_prod_t = 1 - alpha_prod_t
    next_original_sample = (sample - beta_prod_t ** 0.5 * model_output) / alpha_prod_t ** 0.5
    next_sample_direction = (1 - alpha_prod_t_next) ** 0.5 * model_output
    next_sample = alpha_prod_t_next ** 0.5 * next_original_sample + next_sample_direction
    return next_sample

def get_multicontrol_pre_single(latents, t, context, unet, controlnets, controls, control_scale):
    for i, (image, scale, controlnet) in enumerate(zip(controls, control_scale, controlnets)):
        down_samples, mid_sample = controlnet(
            latents,
            t,
            encoder_hidden_states=context,
            controlnet_cond=image,
            return_dict=False,
        )
        down_samples = [
            down_samples * scale
            for down_samples in down_samples
        ]
        mid_sample *= scale

        # merge samples
        if i == 0:
            down_block_res_samples, mid_block_res_sample = down_samples, mid_sample
        else:
            down_block_res_samples = [
                samples_prev + samples_curr
                for samples_prev, samples_curr in zip(down_block_res_samples, down_samples)
            ]
            mid_block_res_sample += mid_sample

    noise_pred = unet(
        latents,
        t,
        encoder_hidden_states=context,
        down_block_additional_residuals=down_block_res_samples,
        mid_block_additional_residual=mid_block_res_sample,
    )["sample"]
    return noise_pred

def get_noise_pred_single(latents, t, context, unet, controlnet, controls, controlnet_conditioning_scale=1.0):
    down_block_res_samples, mid_block_res_sample = controlnet(
        latents,
        t,
        encoder_hidden_states=context,
        controlnet_cond=controls,
        return_dict=False,
    )
    down_block_res_samples = [
        down_block_res_sample * controlnet_conditioning_scale
        for down_block_res_sample in down_block_res_samples
    ]
    mid_block_res_sample *= controlnet_conditioning_scale
    noise_pred = unet(
        latents,
        t,
        encoder_hidden_states=context,
        down_block_additional_residuals=down_block_res_samples,
        mid_block_additional_residual=mid_block_res_sample,
    )["sample"]
    return noise_pred


@torch.no_grad()
def ddim_loop(pipeline, ddim_scheduler, latent, num_inv_steps, prompt,controls, controlnet_conditioning_scale):
    context = init_prompt(prompt, pipeline)
    uncond_embeddings, cond_embeddings = context.chunk(2)
    all_latent = [latent]
    latent = latent.clone().detach()
    for i in tqdm(range(num_inv_steps)):
        t = ddim_scheduler.timesteps[len(ddim_scheduler.timesteps) - i - 1]
        noise_pred = get_noise_pred_single(latent, t, cond_embeddings, pipeline.unet, pipeline.controlnet, controls, controlnet_conditioning_scale)
        latent = next_step(noise_pred, t, latent, ddim_scheduler)
        all_latent.append(latent)
    return all_latent


@torch.no_grad()
def ddim_inversion(pipeline, ddim_scheduler, video_latent, num_inv_steps, prompt="", controls=None, controlnet_conditioning_scale=1.0):
    ddim_latents = ddim_loop(pipeline, ddim_scheduler, video_latent, num_inv_steps, prompt, controls, controlnet_conditioning_scale)
    return ddim_latents

import inspect
import os
from typing import Dict, Optional, Tuple
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import diffusers
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from diffusers import AutoencoderKL, DDPMScheduler, DDIMScheduler
from diffusers.optimization import get_scheduler
from diffusers.utils.import_utils import is_xformers_available
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer
from util import backup_profile, save_videos_grid, set_logger, save_tensor_images_folder
from dataset import VideoDataset
from libs.piplines import VideoControlNetPipeline
from einops import rearrange
import numpy as np
from annotator.util import get_control, HWC3
import logging
from sampling import VideoGen
from libs.unet import UNet3DConditionModel
from libs.controlnet3d import ControlNetModel

logger = get_logger(__name__)

def main(
    pretrained_model_path: str,
    output_dir: str,
    pretrained_controlnet_path: str,
    train_data: Dict,
    validation_data: Dict,
    control_config: Dict,
    validation_steps: int = 100,
    trainable_modules: Tuple[str] = (
        "attn1.to_q",
    ),
    train_batch_size: int = 1,
    max_train_steps: int = 500,
    learning_rate: float = 3e-5,
    scale_lr: bool = False,
    lr_scheduler: str = "constant",
    lr_warmup_steps: int = 0,
    adam_beta1: float = 0.9,
    adam_beta2: float = 0.999,
    adam_weight_decay: float = 1e-2,
    adam_epsilon: float = 1e-08,
    max_grad_norm: float = 1.0,
    gradient_accumulation_steps: int = 1,
    gradient_checkpointing: bool = True,
    mixed_precision: Optional[str] = "fp16",
    enable_xformers_memory_efficient_attention: bool = True,
    seed: Optional[int] = None
):
    if seed is not None:
        set_seed(seed)

    # set logging file
    output_dir_log = output_dir
    os.makedirs(output_dir_log, exist_ok=True)

    *_, config = inspect.getargvalues(inspect.currentframe())

    backup_profile(config, output_dir_log)
    set_logger(output_dir_log)
    logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s - %(message)s", datefmt="%m/%d/%Y %H:%M:%S",
                        level=logging.INFO)
    logging.info(output_dir_log)

    # set accelerator
    accelerator = Accelerator(
        gradient_accumulation_steps=gradient_accumulation_steps,
        mixed_precision=mixed_precision,
    )

    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # prepare models
    noise_scheduler = DDPMScheduler.from_pretrained(pretrained_model_path, subfolder="scheduler")
    tokenizer = CLIPTokenizer.from_pretrained(pretrained_model_path, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(pretrained_model_path, subfolder="text_encoder")
    vae = AutoencoderKL.from_pretrained(pretrained_model_path, subfolder="vae")
    unet = UNet3DConditionModel.from_pretrained_2d(pretrained_model_path, subfolder="unet")
    controlnet = ControlNetModel.from_pretrained_2d(pretrained_controlnet_path)
    apply_control = get_control(control_config.type)

    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)

    if enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            unet.enable_xformers_memory_efficient_attention()
            controlnet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available. Make sure it is installed correctly")

    if gradient_checkpointing:
        unet.enable_gradient_checkpointing()
        controlnet.enable_gradient_checkpointing()

    if scale_lr:
        learning_rate = (
            learning_rate * gradient_accumulation_steps * train_batch_size * accelerator.num_processes
        )

    optimizer_cls = torch.optim.AdamW

    # set the params need to optimize
    # do not set unet.requires_grad_(False) because of the bug in gradient checkpointing in torch, where if the all the inputs don't need grad, the module in gradient checkpointing will not compute grad.
    optimize_params = []
    params_len = 0
    for name, module in unet.named_modules():
        if name.endswith(tuple(trainable_modules)):
            optimize_params += list(module.parameters())
            for params in module.parameters():
                params_len += len(params.reshape(-1, ))

    for name, module in controlnet.named_modules():
        if name.endswith(tuple(trainable_modules)):
            optimize_params += list(module.parameters())
            for params in module.parameters():
                params_len += len(params.reshape(-1, ))

    logger.info(f"trainable params: {params_len / (1024 * 1024):.2f} M")

    optimizer = optimizer_cls(
        optimize_params,
        lr=learning_rate,
        betas=(adam_beta1, adam_beta2),
        weight_decay=adam_weight_decay,
        eps=adam_epsilon,
    )

    # prepare dataloader
    train_dataset = VideoDataset(**train_data)
    train_dataset.prompt_ids = tokenizer(
        train_dataset.prompt, max_length=tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
    ).input_ids[0]
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=train_batch_size
    )

    # prepare VideoControlNetPipeline
    validation_pipeline = VideoControlNetPipeline(
        vae=vae, text_encoder=text_encoder, tokenizer=tokenizer, unet=unet, controlnet=controlnet,
        scheduler=DDIMScheduler.from_pretrained(pretrained_model_path, subfolder="scheduler")
    )
    validation_pipeline.enable_vae_slicing()

    ddim_inv_scheduler = DDIMScheduler.from_pretrained(pretrained_model_path, subfolder='scheduler')
    ddim_inv_scheduler.set_timesteps(validation_data.num_steps)

    lr_scheduler = get_scheduler(
        lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=lr_warmup_steps * gradient_accumulation_steps,
        num_training_steps=max_train_steps * gradient_accumulation_steps,
    )

    controlnet, unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        controlnet, unet, optimizer, train_dataloader, lr_scheduler
    )

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    text_encoder.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device, dtype=weight_dtype)

    if accelerator.is_main_process:
        accelerator.init_trackers("text2video-fine-tune")

    # show the progress bar
    global_step = 0
    progress_bar = tqdm(range(global_step, max_train_steps), disable=not accelerator.is_local_main_process)
    progress_bar.set_description("Steps")

    # Since we only have one video, we first compute its control before training to save time
    for step, batch in enumerate(train_dataloader):

        #only support 1 batchsize
        assert batch["control"].shape[0] == 1

        control_ = batch["control"].squeeze() #[f h w c] in {0,1,……,255}
        control = []

        # compute control for each frame
        for i in control_:
            if control_config.type == 'canny':
                detected_map = apply_control(i.cpu().numpy(), control_config.low_threshold, control_config.high_threshold)
            elif control_config.type == 'openpose' or control_config.type == 'depth':
                detected_map, _ = apply_control(i.cpu().numpy())
            elif control_config.type == 'hed' or control_config.type == 'seg':
                detected_map = apply_control(i.cpu().numpy())
            elif control_config.type == 'scribble':
                i = i.cpu().numpy()
                detected_map = np.zeros_like(i, dtype=np.uint8)
                detected_map[np.min(i, axis=2) < control_config.value] = 255
            elif control_config.type == 'normal':
                _, detected_map = apply_control(i.cpu().numpy(), bg_th=control_config.bg_threshold)
            elif control_config.type == 'mlsd':
                detected_map = apply_control(i.cpu().numpy(), control_config.value_threshold, control_config.distance_threshold)
            else:
                raise ValueError(control_config.type)
            control.append(HWC3(detected_map))

        # stack control with all frames with shape [b c f h w]
        control = np.stack(control)
        control = np.array(control).astype(np.float32) / 255.0
        control = torch.from_numpy(control).to(accelerator.device)
        control = control.unsqueeze(0) #[f h w c] -> [b f h w c ]
        control = rearrange(control, "b f h w c -> b c f h w")
        control = control.to(weight_dtype)

        pixel_values = batch["pixel_values"].to(weight_dtype)

        # for save original video
        x0 = rearrange(pixel_values, "b f c h w -> b c f h w")
        x0 = (x0 + 1.0) / 2.0  # -1,1 -> 0,1

        # prepare latents with shape [b c f h w]
        video_length = pixel_values.shape[1]
        pixel_values = rearrange(pixel_values, "b f c h w -> (b f) c h w")
        latents = vae.encode(pixel_values).latent_dist.sample()
        latents = rearrange(latents, "(b f) c h w -> b c f h w", f=video_length)
        latents = latents * 0.18215

        # prepare text embedding
        encoder_hidden_states = text_encoder(batch["prompt_ids"])[0]

        while global_step <= max_train_steps:
            unet.train()
            controlnet.train()

            train_loss = 0.0

            # add noise
            noise = torch.randn_like(latents)
            bsz = latents.shape[0]
            timesteps = torch.randint(0, noise_scheduler.num_train_timesteps, (bsz,), device=latents.device)
            timesteps = timesteps.long()
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

            # noise prediction
            down_block_res_samples, mid_block_res_sample = controlnet(
                noisy_latents,
                timesteps,
                encoder_hidden_states=encoder_hidden_states,
                controlnet_cond=control,
                return_dict=False,
            )
            down_block_res_samples = [
                down_block_res_sample * control_config.control_scale
                for down_block_res_sample in down_block_res_samples
            ]
            mid_block_res_sample *= control_config.control_scale
            model_pred = unet(
                noisy_latents,
                timesteps,
                encoder_hidden_states=encoder_hidden_states,
                down_block_additional_residuals=down_block_res_samples,
                mid_block_additional_residual=mid_block_res_sample,
            ).sample

            # compute loss
            loss = F.mse_loss(model_pred.float(), noise.float(), reduction="mean")
            avg_loss = accelerator.gather(loss.repeat(train_batch_size)).mean()
            train_loss += avg_loss.item() / gradient_accumulation_steps

            accelerator.backward(loss)

            if accelerator.sync_gradients:
                params_to_clip = list(unet.parameters()) + list(controlnet.parameters())
                accelerator.clip_grad_norm_(params_to_clip, max_grad_norm)

            optimizer.step()
            optimizer.zero_grad()

            # for saving outputs
            origin_save = x0.cpu().float()
            control_save = control.cpu().float()

            if accelerator.sync_gradients:
                accelerator.log({"train_loss": train_loss}, step=global_step)
                progress_bar.update(1)
                global_step += 1

                if global_step % validation_steps == 0:
                    if accelerator.is_main_process:
                        unet.eval()
                        controlnet.eval()

                        samples = [x0.cpu().float(), control.cpu().float()]

                        generator = torch.Generator(device=latents.device)
                        generator.manual_seed(seed)

                        samples = VideoGen(validation_data, generator, latents, validation_pipeline, ddim_inv_scheduler, train_data, control, weight_dtype, control_config.control_scale, samples)
                        sample_save = samples[-1]
                        samples = torch.concat(samples)
                        save_path = f"{output_dir_log}/{global_step}.mp4"
                        save_videos_grid(samples, save_path)

                logs = {"step_loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
                progress_bar.set_postfix(**logs)

        print("save origin")
        save_path = f"{output_dir}/results/origin"
        save_tensor_images_folder(origin_save, save_path)

        print("save control")
        save_path = f"{output_dir}/results/control"
        save_tensor_images_folder(control_save, save_path)

        print("save translated video")
        save_path = f"{output_dir}/results/controlvideo"
        save_tensor_images_folder(sample_save, save_path)

    accelerator.end_training()



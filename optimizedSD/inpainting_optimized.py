import argparse
import os
import re
import time
from contextlib import nullcontext
from itertools import islice
from random import randint

import numpy as np
import torch
from PIL import Image
from einops import rearrange, repeat
from omegaconf import OmegaConf
from pytorch_lightning import seed_everything
from torch import autocast
from torchvision.utils import make_grid
from tqdm import tqdm, trange
from transformers import logging

from ldm.util import instantiate_from_config
from optimUtils import split_weighted_subprompts

logging.set_verbosity_error()

def chunk(it, size):
    it = iter(it)
    return iter(lambda: tuple(islice(it, size)), ())


def load_model_from_config(ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    return sd

def load_img(path, h0, w0):
   
    image = Image.open(path).convert("RGB")
    w, h = image.size

    print(f"loaded input image of size ({w}, {h}) from {path}")   
    if(h0 is not None and w0 is not None):
        h, w = h0, w0
    
    w, h = map(lambda x: x - x % 64, (w, h))  # resize to integer multiple of 32

    print(f"New image size ({w}, {h})")
    image = image.resize((w, h), resample = Image.LANCZOS)
    image = np.array(image).astype(np.float32) / 255.0
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)
    return 2.*image - 1.

def load_mask(mask, h0, w0, newH, newW, invert=False):
    image = Image.open(mask).convert("L")
    image = np.array(image)
    if invert:
        image = np.clip(image,254,255)+1    
    else:
        image = np.clip(image+1,0,1)-1
    image = Image.fromarray(image).convert("RGB")
    w, h = image.size
    print(f"loaded input mask of size ({w}, {h})")
    if h0 is not None and w0 is not None:
        h, w = h0, w0

    w, h = map(lambda x: x - x % 64, (w, h))  # resize to integer multiple of 32

    print(f"New mask size ({w}, {h})")
    image = image.resize((newW, newH), resample=Image.LANCZOS)
    # image = image.resize((64, 64), resample=Image.LANCZOS)
    image = np.array(image)

    # if invert:
    #     print("inverted")
    #     where_0, where_1 = np.where(image == 0), np.where(image == 255)
    #     image[where_0], image[where_1] = 255, 0
    image = image.astype(np.float32) / 255.0
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)
    return image

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--prompt",
        type=str,
        nargs="?",
        default="a painting of a virus monster playing guitar",
        help="the prompt to render"
    )
    parser.add_argument(
        "--outdir",
        type=str,
        nargs="?",
        help="dir to write results to",
        default="outputs/img2img-samples"
    )

    parser.add_argument(
        "--init_image",
        type=str,
        nargs="?",
        help="path to the input image"
    )
    parser.add_argument(
        "--mask",
        type=str,
        nargs="?",
        help="path to the input mask"
    )
    parser.add_argument(
        "--skip_grid",
        action='store_true',
        help="do not save a grid, only individual samples. Helpful when evaluating lots of samples",
    )
    parser.add_argument(
        "--skip_save",
        action='store_true',
        help="do not save individual samples. For speed measurements.",
    )
    parser.add_argument(
        "--ddim_steps",
        type=int,
        default=50,
        help="number of ddim sampling steps",
    )

    parser.add_argument(
        "--ddim_eta",
        type=float,
        default=0.0,
        help="ddim eta (eta=0.0 corresponds to deterministic sampling",
    )
    parser.add_argument(
        "--n_iter",
        type=int,
        default=1,
        help="sample this often",
    )
    parser.add_argument(
        "--H",
        type=int,
        default=None,
        help="image height, in pixel space",
    )
    parser.add_argument(
        "--W",
        type=int,
        default=None,
        help="image width, in pixel space",
    )
    parser.add_argument(
        "--strength",
        type=float,
        default=0.75,
        help="strength for noising/unnoising. 1.0 corresponds to full destruction of information in init image",
    )
    parser.add_argument(
        "--C",
        type=int,
        default=4,
        help="latent channels",
    )
    parser.add_argument(
        "--f",
        type=int,
        default=8,
        help="downsampling factor",
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=1,
        help="how many samples to produce for each given prompt. A.k.a. batch size",
    )
    parser.add_argument(
        "--n_rows",
        type=int,
        default=0,
        help="rows in the grid (default: n_samples)",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=7.5,
        help="unconditional guidance scale: eps = eps(x, empty) + scale * (eps(x, cond) - eps(x, empty))",
    )
    parser.add_argument(
        "--from-file",
        type=str,
        help="if specified, load prompts from this file",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="the seed (for reproducible sampling)",
    )
    parser.add_argument(
        "--small_batch",
        action='store_true',
        help="Reduce inference time when generate a smaller batch of images",
    )
    parser.add_argument(
        "--precision",
        type=str,
        help="evaluate at this precision",
        choices=["full", "autocast"],
        default="autocast"
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        help="path to checkpoint of model",
    )
    parser.add_argument(
        "--turbo",
        action="store_true",
        help="Reduces inference time on the expense of 1GB VRAM",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="CPU or GPU (cuda/cuda:0/cuda:1/...)",
    )
    parser.add_argument(
        "--unet_bs",
        type=int,
        default=1,
        help="Slightly reduces inference time at the expense of high VRAM (value > 1 not recommended )",
    )
    parser.add_argument(
        "--sampler",
        type=str,
        help="sampler",
        choices=["ddim","plms"],
        default="ddim",
    )
    parser.add_argument(
        "--superfast",
        action="store_true",
        help="Reduces inference time on the expense of 1GB VRAM",
    )
    parser.add_argument(
        "--invert",
        action="store_true",
        help="Invert mask",
    )
    opt = parser.parse_args()
    if opt.superfast:
        config = "optimizedSD/v1-inference.yaml"
    else:
        config = "optimizedSD/v1-inference_lowvram.yaml"    
    ckpt = opt.ckpt
    device = opt.device
    
    sd = load_model_from_config(f"{ckpt}")
    li, lo = [], []
    for key, v_ in sd.items():
        sp = key.split(".")
        if (sp[0]) == "model":
            if "input_blocks" in sp:
                li.append(key)
            elif "middle_block" in sp:
                li.append(key)
            elif "time_embed" in sp:
                li.append(key)
            else:
                lo.append(key)
    for key in li:
        sd["model1." + key[6:]] = sd.pop(key)
    for key in lo:
        sd["model2." + key[6:]] = sd.pop(key)

    config = OmegaConf.load(f"{config}")

    model = instantiate_from_config(config.modelUNet)
    _, _ = model.load_state_dict(sd, strict=False)
    model.eval()

    modelCS = instantiate_from_config(config.modelCondStage)
    _, _ = modelCS.load_state_dict(sd, strict=False)
    modelCS.eval()

    modelFS = instantiate_from_config(config.modelFirstStage)
    _, _ = modelFS.load_state_dict(sd, strict=False)
    modelFS.eval()
    del sd

    seed_everything(opt.seed)
    sampler = "ddim"

    init_image = load_img(opt.init_image, opt.H, opt.W).to(device)

    model.unet_bs = opt.unet_bs
    model.turbo = opt.turbo
    model.cdevice = device
    modelCS.cond_stage_model.device = device

    if device != "cpu" and opt.precision != "full":
        model.half()
        modelCS.half()
        modelFS.half()
        init_image = init_image.half()
        # mask.half()

    tic = time.time()

    # n_rows = opt.n_rows if opt.n_rows > 0 else opt.n_samples
    batch_size = opt.n_samples
    n_rows = opt.n_rows if opt.n_rows > 0 else batch_size
    if not opt.from_file:
        prompt = opt.prompt
        print("Prompt:",opt.prompt)
        assert prompt is not None
        data = [batch_size * [prompt]]
    else:
        with open(opt.from_file+"prompt.txt", "r") as f:
            opt.prompt = f.read().splitlines()[0]
            print("Prompt:",opt.prompt)
            # data = list(chunk(opt.prompt, batch_size))
            data = [batch_size * [opt.prompt]]
        try:
            with open(opt.from_file+"negative_prompt.txt", "r") as f:
                negative_prompt = f.read().splitlines()[0]
                print("Negative Prompt:",negative_prompt)
                negative_prompt_data = [batch_size * negative_prompt]
        except:
            negative_prompt_data = [batch_size * ""]

    os.makedirs(opt.outdir, exist_ok=True)
    outpath = opt.outdir

    sample_path = os.path.join(outpath,re.sub(r'\W+', '',"_".join(opt.prompt.split())))[:150]
    os.makedirs(sample_path, exist_ok=True)
    base_count = len(os.listdir(sample_path))
    grid_count = len(os.listdir(outpath)) - 1

    modelFS.to(device)

    init_latent = modelFS.get_first_stage_encoding(modelFS.encode_first_stage(init_image))  # move to latent space
    init_latent = repeat(init_latent, "1 ... -> b ...", b=opt.n_samples)

    mask = load_mask(opt.mask, opt.H, opt.W, init_latent.shape[2], init_latent.shape[3], opt.invert).to(device)
    mask = mask[0][0].unsqueeze(0).repeat(4, 1, 1).unsqueeze(0)
    mask = repeat(mask, '1 ... -> b ...', b=opt.n_samples)

    if device != "cpu":
        mem = torch.cuda.memory_allocated() / 1e6
        modelFS.to("cpu")
        while torch.cuda.memory_allocated() / 1e6 >= mem:
            time.sleep(1)

    if opt.strength == 1:
        print("strength should be less than 1, setting it to 0.999")
        strength = 0.999
    assert 0.0 <= opt.strength < 1.0, "can only work with strength in [0.0, 1.0]"
    t_enc = int(opt.strength * opt.ddim_steps)
    print(f"target t_enc is {t_enc} steps")

    if opt.precision != "full" and device != "cpu":
        precision_scope = autocast
    else:
        precision_scope = nullcontext

    all_samples = []
    with torch.no_grad():
        all_samples = list()
        for _ in trange(opt.n_iter, desc="Sampling"):
            for prompts in tqdm(data, desc="data"):
                with precision_scope("cuda"):
                    modelCS.to(device)
                    uc = None
                    if opt.scale != 1.0:
                        uc = modelCS.get_learned_conditioning(negative_prompt_data)
                    if isinstance(prompts, tuple):
                        prompts = list(prompts)

                    subprompts, weights = split_weighted_subprompts(prompts[0])
                    if len(subprompts) > 1:
                        c = torch.zeros_like(uc)
                        totalWeight = sum(weights)
                        # normalize each "sub prompt" and add it
                        for i in range(len(subprompts)):
                            weight = weights[i]
                            # if not skip_normalize:
                            weight = weight / totalWeight
                            c = torch.add(c, modelCS.get_learned_conditioning(subprompts[i]), alpha=weight)
                    else:
                        c = modelCS.get_learned_conditioning(prompts)

                    if device != "cpu":
                        mem = torch.cuda.memory_allocated() / 1e6
                        modelCS.to("cpu")
                        while torch.cuda.memory_allocated() / 1e6 >= mem:
                            time.sleep(1)

                    # encode (scaled latent)
                    z_enc = model.stochastic_encode(
                        init_latent, torch.tensor([t_enc] * opt.n_samples).to(device),
                        opt.seed, opt.ddim_eta, opt.ddim_steps)

                    # decode it
                    samples_ddim = model.sample(
                        t_enc,
                        c,
                        z_enc,
                        unconditional_guidance_scale=opt.scale,
                        unconditional_conditioning=uc,
                        mask=mask,
                        x_T=init_latent,
                        sampler=sampler,
                    )

                    modelFS.to(device)
                    print("saving images")
                    for i in range(opt.n_samples):
                        x_samples_ddim = modelFS.decode_first_stage(samples_ddim[i].unsqueeze(0))
                        x_sample = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)
                        all_samples.append(x_sample.to("cpu"))
                        x_sample = 255.0 * rearrange(x_sample[0].cpu().numpy(), "c h w -> h w c")
                        Image.fromarray(x_sample.astype(np.uint8)).save(
                            os.path.join(sample_path, "seed_" + str(opt.seed) + "_" + f"{base_count:05}.png")
                        )
                        opt.seed += 1
                        base_count += 1

                    if device != "cpu":
                        mem = torch.cuda.memory_allocated() / 1e6
                        modelFS.to("cpu")
                        while torch.cuda.memory_allocated() / 1e6 >= mem:
                            time.sleep(1)

                    del samples_ddim
                    del x_sample
                    del x_samples_ddim
                    print("memory_final = ", torch.cuda.memory_allocated() / 1e6)

    toc = time.time()

    time_taken = (toc - tic) / 60.0
    grid = torch.cat(all_samples, 0)
    grid = make_grid(grid, nrow=opt.n_iter)
    grid = 255.0 * rearrange(grid, "c h w -> h w c").cpu().numpy()

    txt = (
            "Samples finished in "
            + str(round(time_taken, 3))
            + " minutes and exported to \n"
            + sample_path
    )
    # return Image.fromarray(grid.astype(np.uint8)), image['mask'], txt


import argparse, os, sys, glob
import time
import torch
import numpy as np

from PIL import Image
from torch import autocast
from itertools import islice
from einops import rearrange
from tqdm import tqdm, trange
from omegaconf import OmegaConf
from torchvision.utils import make_grid
from pytorch_lightning import seed_everything
from contextlib import contextmanager, nullcontext
from transformers import logging
logging.set_verbosity_error()

import torch.nn as nn
import k_diffusion as K
import re
import traceback
from handle_errs import process_error_trace

from ldm.util import instantiate_from_config
from optimUtils import split_weighted_subprompts

def chunk(it, size):
    it = iter(it)
    return iter(lambda: tuple(islice(it, size)), ())

def load_model_from_config(config, ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    _, _ = model.load_state_dict(sd, strict=False)
    model.cuda()
    model.eval()
    return model

class CFGDenoiser(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.inner_model = model

    def forward(self, x, sigma, uncond, cond, cond_scale):
        x_in = torch.cat([x] * 2)
        sigma_in = torch.cat([sigma] * 2)
        cond_in = torch.cat([uncond, cond])
        uncond, cond = self.inner_model(x_in, sigma_in, cond=cond_in).chunk(2)
        return uncond + (cond - uncond) * cond_scale

def create_random_tensors(shape, seeds, device):
    xs = []
    for seed in seeds:
        torch.manual_seed(seed)
        xs.append(torch.randn(shape, device=device))
    x = torch.stack(xs, 0)
    return x

def main():
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
        default="outputs/txt2img-samples"
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
        "--fixed_code",
        action='store_true',
        help="if enabled, uses the same starting code across samples ",
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
        default=2,
        help="sample this often",
    )
    parser.add_argument(
        "--H",
        type=int,
        default=512,
        help="image height, in pixel space",
    )
    parser.add_argument(
        "--W",
        type=int,
        default=512,
        help="image width, in pixel space",
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
        default=3,
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
        "--dyn",
        type=float,
        help="dynamic thresholding from Imagen, in latent space (TODO: try in pixel space with intermediate decode)",
    )
    parser.add_argument(
        "--from-file",
        type=str,
        help="if specified, load prompts from this file",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/stable-diffusion/v1-inference.yaml",
        help="path to config which constructs model",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default="models/ldm/stable-diffusion-v1/model.ckpt",
        help="path to checkpoint of model",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="the seed (for reproducible sampling)",
    )
    parser.add_argument(
        "--sampler",
        type=str,
        help="Choose the sampler used",
        choices=["lms", "euler", "euler_a", "dpm", "dpm_a", "heun"],
        default="lms"
    )
    parser.add_argument(
        "--precision",
        type=str,
        help="evaluate at this precision",
        choices=["full", "autocast"],
        default="autocast"
    )
    parser.add_argument(
        "--turbo",
        action="store_true",
        help="Reduces inference time on the expense of 1GB VRAM",
    )
    opt = parser.parse_args()

    if opt.seed is None:
        opt.seed = hash(opt)

    config = OmegaConf.load(f"{opt.config}")
    model = load_model_from_config(config, f"{opt.ckpt}")
    model.turbo = opt.turbo

    if opt.precision == "autocast":
        model = model.half()
        torch.set_default_tensor_type(torch.HalfTensor)
    
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    #model = model.to(device)
    ksamplers = {'lms': K.sampling.sample_lms, 
    'euler': K.sampling.sample_euler, 
    'euler_a': K.sampling.sample_euler_ancestral, 
    'dpm': K.sampling.sample_dpm_2, 
    'dpm_a': K.sampling.sample_dpm_2_ancestral,
    'heun': K.sampling.sample_heun }
    
    model_wrap = K.external.CompVisDenoiser(model)
    sampler = ksamplers[opt.sampler]

    os.makedirs(opt.outdir, exist_ok=True)
    outpath = opt.outdir

    batch_size = opt.n_samples
    n_rows = opt.n_rows if opt.n_rows > 0 else batch_size
    if not opt.from_file:
        prompt = opt.prompt
        print("Prompt:",opt.prompt)
        assert prompt is not None
        data = [batch_size * prompt]
    else:
        with open(opt.from_file+"prompt.txt", "r") as f:
            opt.prompt = f.read().splitlines()[0]
            print("Prompt:",opt.prompt)
            # data = list(chunk(opt.prompt, batch_size))
            data = [batch_size * opt.prompt]
        try:
            with open(opt.from_file+"negative_prompt.txt", "r") as f:
                negative_prompt = f.read().splitlines()[0]
                print("Negative Prompt:",negative_prompt)
                negative_prompt_data = [batch_size * negative_prompt]
        except:
            negative_prompt_data = [batch_size * ""]

    try:
        sample_path = os.path.join(outpath,re.sub(r'\W+', '',"_".join(opt.prompt.split())))[:150]
        os.makedirs(sample_path, exist_ok=True)
        base_count = len(os.listdir(sample_path))
        grid_count = len(os.listdir(outpath)) - 1

        precision_scope = autocast if opt.precision=="autocast" else nullcontext
        with torch.no_grad(), precision_scope("cuda"), model.ema_scope():
            all_samples = list()
            for n in trange(opt.n_iter, desc="Sampling"):
                seeds = list(opt.seed + n*batch_size + i for i in range(batch_size))
                uc = None
                if opt.scale != 1.0:
                    uc = model.get_learned_conditioning(negative_prompt_data)
                if isinstance(data, tuple):
                    data = list(data)
                subprompts, weights = split_weighted_subprompts(data[0])
                if len(subprompts) > 1:
                    c = torch.zeros_like(uc)
                    totalWeight = sum(weights)
                    # normalize each "sub prompt" and add it
                    for i in range(len(subprompts)):
                        weight = weights[i]
                        # if not skip_normalize:
                        weight = weight / totalWeight
                        c = torch.add(c, model.get_learned_conditioning(subprompts[i]), alpha=weight)
                else:
                    c = model.get_learned_conditioning(data)

                shape = [opt.C, opt.H // opt.f, opt.W // opt.f]

                sigmas = model_wrap.get_sigmas(opt.ddim_steps)
                x = create_random_tensors(shape, seeds, device=device) * sigmas[0]
                model_wrap_cfg = CFGDenoiser(model_wrap)
                extra_args = {'cond': c, 'uncond': uc, 'cond_scale': opt.scale}

                samples_ddim = sampler(model_wrap_cfg, x, sigmas, extra_args=extra_args, disable=False)
                x_samples_ddim = model.decode_first_stage(samples_ddim)
                x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)

                if  not opt.skip_save:
                    for x_sample in x_samples_ddim:
                        x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
                        Image.fromarray(x_sample.astype(np.uint8)).save(
                            os.path.join(sample_path, "seed_" + str(opt.seed) + "_" + f"{base_count:05}.png"))
                        Image.fromarray(x_sample.astype(np.uint8)).save(
                            os.path.join(sample_path, "latest.png"))    
                        base_count += 1
                        opt.seed +=1 
                if  not opt.skip_grid:
                    all_samples.append(x_samples_ddim)

            if not opt.skip_grid:
                # additionally, save as grid
                grid = torch.stack(all_samples, 0)
                grid = rearrange(grid, 'n b c h w -> (n b) c h w')
                grid = make_grid(grid, nrow=n_rows)

                # to image
                grid = 255. * rearrange(grid, 'c h w -> h w c').cpu().numpy()
                Image.fromarray(grid.astype(np.uint8)).save(os.path.join(outpath, f'grid-{grid_count:04}.png'))
                grid_count += 1
    except Exception as err:
        print(opt.from_file)
        process_error_trace(traceback.format_exc(), err, opt.from_file, outpath)

    print(f"Your samples are ready and waiting for you here: \n{outpath} \n"
          f" \nEnjoy.")

if __name__ == "__main__":
    main()

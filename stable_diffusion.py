import os
import sys
import json
from unittest import load_tests
sys.path.append("stable-diffusion/optimizedSD/")

import numpy as np
import time
import re
import gc
import torch
from PIL import Image
from contextlib import nullcontext
from einops import rearrange, repeat
from itertools import islice
from omegaconf import OmegaConf
from pytorch_lightning import seed_everything
from torch import autocast
from tqdm import tqdm, trange
from transformers import logging
import base64
from io import BytesIO
import random

from ldm.util import instantiate_from_config
logging.set_verbosity_error()
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 

def chunk(it, size):
    it = iter(it)
    return iter(lambda: tuple(islice(it, size)), ())

def load_model_from_config(ckpt, verbose=False):
    # print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    # if "global_step" in pl_sd:
    #     print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    return sd

def load_img(image, h0, w0):
    w, h = image.size
    if(h0 != 0 and w0 != 0):
        h, w = h0, w0
    
    w, h = map(lambda x: x - x % 64, (w, h))  # resize to integer multiple of 32

    print(f"New image size ({w}, {h})")
    image = image.resize((w, h), resample = Image.LANCZOS)
    image = np.array(image).astype(np.float32) / 255.0
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)
    return 2.*image - 1.

def load_mask(mask, h0, w0, newH, newW, invert=False):
    image = np.array(mask)
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

def image_to_b64(image):
    image_file = BytesIO()
    image.save(image_file, format='JPEG')
    im_bytes = image_file.getvalue()  # im_bytes: image in binary format.
    imgb64 = base64.b64encode(im_bytes)
    return 'data:image/jpeg;base64,'+str(imgb64)[2:-1]

def b64_to_image(b64):
    image_data = re.sub('^data:image/.+;base64,', '', b64)
    return Image.open(BytesIO(base64.b64decode(image_data))).convert('RGB')

def image_grid(imgs, rows, cols, path):
    assert len(imgs) <= rows*cols
    imgs = [Image.fromarray(img) for img in imgs]
    w, h = imgs[0].size
    grid = Image.new('RGB', size=(cols*w, rows*h))
    
    for i, img in enumerate(imgs):
        grid.paste(img, box=(i%cols*w, i//cols*h))
    grid.save(path)

class StableDiffusion():
    def __init__(self):
        self.current_num = 0
        self.total_num = 0
        self.loading_model = False

        self.artroom_path = None
        self.latest_images_part1 = []
        self.latest_images_part2 = []
        self.latest_images_id = 0

        self.model = None
        self.modelCS = None 
        self.modelFS = None 

        self.ckpt = ''
        self.image_save_path = os.environ['USERPROFILE']+'/Desktop/'
        self.device = "cuda"
        self.precision = "autocast"
        self.speed = "High"

    def set_artroom_path(self,path):
        self.artroom_path = path
        #First load of ckpt
        loaded = False
        if os.path.exists(f"{self.artroom_path}/artroom/settings/sd_settings.json"):
            sd_settings = json.load(open(f"{self.artroom_path}/artroom/settings/sd_settings.json"))
            model_ckpt = sd_settings['ckpt']
            speed = sd_settings['speed']
            precision = sd_settings['precision']
            if os.path.exists(model_ckpt):
                loaded = self.load_ckpt(model_ckpt,speed,precision)

        if not loaded:
            if os.path.exists(f"{self.artroom_path}/artroom/model_weights/model.ckpt"):
                loaded = self.load_ckpt(f"{self.artroom_path}/artroom/model_weights/model.ckpt",self.speed,self.precision)
                

    def get_steps(self):
        if self.model:
            return self.current_num, self.total_num, self.model.current_step, self.model.total_steps
        else:
            return 0,0,0,0

    def get_latest_images(self):
        return self.latest_images_part1 + self.latest_images_part2

    def get_latest_image(self):
        latest_images = self.get_latest_images()
        if len(latest_images) > 0:
            return image_to_b64(Image.open(latest_images[-1]).convert('RGB'))
        else:
            return ''
    
    def loaded_models(self):
        return self.model != None

    def load_ckpt(self,ckpt,speed,precision):
        assert ckpt != '', 'Checkpoint cannot be empty'
        if self.ckpt != ckpt or self.speed != speed or self.precision != precision:
            try:
                self.set_up_models(ckpt,speed,precision)
                return True
            except:
                self.loading_model = False 
                self.model = None 
                self.modelCS = None 
                self.modelFS = None 
                return False

    def set_up_models(self, ckpt, speed, precision):
        self.loading_model = True
        if speed == 'Low':
            self.config = 'stable-diffusion/optimizedSD/v1-inference_lowvram.yaml'
        elif speed == 'Medium':
            self.config = 'stable-diffusion/optimizedSD/v1-inference_lowvram.yaml'
        elif speed == 'High':
            self.config = 'stable-diffusion/optimizedSD/v1-inference.yaml'
        elif speed == 'Max':
            self.config = 'stable-diffusion/optimizedSD/v1-inference_xformer.yaml'
        sd = load_model_from_config(f"{ckpt}")
        li = []
        lo = []
        for key, value in sd.items():
            sp = key.split('.')
            if(sp[0]) == 'model':
                if('input_blocks' in sp):
                    li.append(key)
                elif('middle_block' in sp):
                    li.append(key)
                elif('time_embed' in sp):
                    li.append(key)
                else:
                    lo.append(key)
        for key in li:
            sd['model1.' + key[6:]] = sd.pop(key)  
        for key in lo:
            sd['model2.' + key[6:]] = sd.pop(key)

        config = OmegaConf.load(f"{self.config}")
        self.model = instantiate_from_config(config.modelUNet)
        _, _ = self.model.load_state_dict(sd, strict=False)
        self.model.eval()
        self.model.cdevice = self.device
        self.model.unet_bs = 1 #unet_bs=1

        self.model.turbo = (speed != 'Low')
            
        self.modelCS = instantiate_from_config(config.modelCondStage)
        _, _ = self.modelCS.load_state_dict(sd, strict=False)
        self.modelCS.eval()
        self.modelCS.cond_stage_model.device = self.device

        self.modelFS = instantiate_from_config(config.modelFirstStage) 
        _, _ = self.modelFS.load_state_dict(sd, strict=False)
        self.modelFS.eval()
        del sd
        if self.device != "cpu" and precision == "autocast":
            self.model.half()
            self.modelCS.half()
            self.modelFS.half()
            torch.set_default_tensor_type(torch.HalfTensor)        
        
        self.ckpt = ckpt
        self.speed = speed 
        self.precision= precision
        self.loading_model = False

    def generate(self, text_prompts="",negative_prompts="",batch_name="",init_image="",mask="",invert=False,steps=50,H=512,W=512,strength=0.75,cfg_scale=7.5,seed=-1,sampler="ddim",C=4,ddim_eta=0.0,f=8,n_iter=4,batch_size=1,ckpt="", image_save_path = "", speed="High", device = 'cuda', precision = 'autocast', skip_grid = False):
        self.latest_images_part1 = self.latest_images_part2
        self.latest_images_part2 = []
        
        torch.cuda.empty_cache()
        gc.collect()
        seed_everything(seed)

        if len(init_image) > 0: #and 'dpm' not in sampler:
            sampler = 'ddim'

        self.image_save_path = image_save_path
        ddim_steps = steps 

        # self.device = device

        print("Setting up models...")
        self.load_ckpt(ckpt,speed,precision)
        if not self.model:
            return 'Failure'
        
        print("Generating...")
        outdir = self.image_save_path + batch_name
        os.makedirs(outdir, exist_ok=True)

        if len(init_image) > 0:
            if init_image[:4] == 'data':
                print("Loading image from b64")
                image = b64_to_image(init_image).convert('RGB')
            else:
                image = Image.open(init_image).convert('RGB')
            init_image = load_img(image, H, W).to(self.device)
            _, _, H, W = init_image.shape
            if self.device != "cpu" and self.precision == "autocast":
                init_image = init_image.half()
        else:
            init_image = None

        print("Prompt:",text_prompts)
        data = [batch_size * text_prompts]
        print("Negative Prompt:",negative_prompts)
        negative_prompts_data = [batch_size * negative_prompts]

        sample_path = os.path.join(outdir,re.sub(r'\W+', '',"_".join(text_prompts.split())))[:150]
        # sample_path = outdir
        os.makedirs(sample_path, exist_ok=True)
        base_count = len(os.listdir(sample_path))

        if init_image is not None:
            self.modelFS.to(self.device)
            init_image = repeat(init_image, '1 ... -> b ...', b=batch_size)
            init_latent = self.modelFS.get_first_stage_encoding(self.modelFS.encode_first_stage(init_image)).to(self.device)  # move to latent space
            steps = int(strength * steps)
            if steps <= 0:
                steps = 1
            if self.device != "cpu":
                mem = torch.cuda.memory_allocated(device=self.device) / 1e6
                self.modelFS.to("cpu")
                while(torch.cuda.memory_allocated(device=self.device)/1e6 >= mem):
                    time.sleep(1)
        if len(mask) > 0:
            if mask[:4] == 'data':
                print("Loading mask from b64")
                mask = b64_to_image(mask).convert('L')
            else:
                mask = Image.open(mask).convert("L")
            mask = load_mask(mask, H, W, init_latent.shape[2], init_latent.shape[3], invert).to(self.device)
            mask = mask[0][0].unsqueeze(0).repeat(4, 1, 1).unsqueeze(0)
            mask = repeat(mask, '1 ... -> b ...', b=batch_size)
            x_T = init_latent
        else:
            mask = None
            x_T = None 

        if self.precision == "autocast" and self.device != "cpu":
            precision_scope = autocast
        else:
            precision_scope = nullcontext

        self.total_num = n_iter
        
        with torch.no_grad():
            all_samples = list()
            for n in trange(n_iter, desc="Sampling"):
                self.current_num = n
                self.model.current_step = 0
                self.model.total_steps = steps
                for prompts in tqdm(data, desc="data"):
                    with precision_scope("cuda"):
                        self.modelCS.to(self.device)
                        uc = None
                        if cfg_scale != 1.0:
                            uc = self.modelCS.get_learned_conditioning(negative_prompts_data)
                        if isinstance(prompts, tuple):
                            prompts = list(prompts)
                        c = self.modelCS.get_learned_conditioning(prompts)

                        shape = [batch_size, C, H // f, W // f]

                        if init_image is not None:
                            x0 = self.model.stochastic_encode(
                                init_latent,
                                torch.tensor([steps] * batch_size).to(self.device),
                                seed,
                                ddim_eta,
                                ddim_steps,
                            )
                        else:
                            x0 = None
                        # decode it
                        
                        print("Sampler",sampler)
                        samples_ddim = self.model.sample(
                            S=steps,
                            conditioning=c,
                            x0=x0,
                            unconditional_guidance_scale=cfg_scale,
                            unconditional_conditioning=uc,
                            eta = ddim_eta,
                            sampler=sampler,
                            shape=shape,
                            batch_size=batch_size,
                            seed=seed,
                            mask=mask,
                            x_T=x_T
                        )
                        self.modelFS.to(self.device)
                        for i in range(batch_size):
                            x_samples_ddim = self.modelFS.decode_first_stage(samples_ddim[i].unsqueeze(0))
                            x_sample = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)
                            x_sample = 255. * rearrange(x_sample[0].cpu().numpy(), 'c h w -> h w c')
                            Image.fromarray(x_sample.astype(np.uint8)).save(
                                os.path.join(sample_path, "seed_" + str(seed) + "_" + f"{base_count:05}.png"))
                            self.latest_images_part2.append(Image.fromarray(x_sample.astype(np.uint8)))
                            self.latest_images_id = random.randint(1,922337203685)

                            base_count += 1
                            seed += 1
                            if not skip_grid:
                                all_samples.append(x_sample.astype(np.uint8))

                        if self.device != "cpu":
                            mem = torch.cuda.memory_allocated() / 1e6
                            self.modelFS.to("cpu")
                            while(torch.cuda.memory_allocated()/1e6 >= mem):
                                time.sleep(1)

                        del samples_ddim
            if not skip_grid:
                rows = int(np.sqrt(len(all_samples)))
                cols = int(np.ceil(len(all_samples)/rows))
                os.makedirs(sample_path+"/grids",exist_ok=True)
                image_grid(all_samples, rows, cols, path = os.path.join(sample_path+"/grids", f'grid-{len(os.listdir(sample_path+"/grids")):04}.png'))

            if not skip_grid:
                # additionally, save as grid
                rows = int(np.sqrt(len(all_samples)))
                cols = int(np.ceil(len(all_samples)/rows))
                os.makedirs(sample_path+"/grids",exist_ok=True)
                image_grid(all_samples, rows, cols, path = os.path.join(sample_path+"/grids", f'grid-{len(os.listdir(sample_path+"/grids")):04}.png'))

        self.total_num = 0
        self.current_num = 0
        if self.model:
            self.model.current_step = 0
            self.model.total_steps = 0
        
import subprocess
import json
import os
import random
import time
import subprocess
import sys
import re
from glob import glob

userprofile = os.environ["USERPROFILE"]
sd_json = json.load(open(f"{userprofile}/artroom/settings/sd_settings.json"))
sd_settings = sd_json['Settings']
sd_config = sd_json['Config']

os.chdir('stable-diffusion')

if sd_config['image_save_path'][-1] == "/" or sd_config['image_save_path'][-1] == "\\":
    sd_config['image_save_path'] = sd_config['image_save_path'][:-1]

outdir = sd_config['image_save_path'] + "/" + sd_settings['batch_name']
if "%UserProfile%" in outdir:
    outdir = outdir.replace("%UserProfile%",userprofile)

if sd_settings['use_random_seed']:
    seed = random.randint(1, 2703686851)
    sd_settings['seed'] = seed
    # sd_settings['use_random_seed'] = False
else:
    seed = sd_settings['seed']

if sd_settings['init_image'] == "":
    #txt2img
    if sd_config['use_optimized_version']:
        runner_script = "optimizedSD/optimized_txt2img.py"
        if sd_settings['sampler'] != "ddim" and sd_settings['sampler'] != "plms":
            runner_script = "optimizedSD_K/optimized_txt2img_k.py"
    else:
        runner_script = "scripts/txt2img.py"
        if sd_settings['sampler'] != "ddim" and sd_settings['sampler'] != "plms":
            runner_script = "scripts/txt2img_k.py"
else:
    #img2img
    if sd_config['use_optimized_version']:
        runner_script = "optimizedSD/optimized_img2img.py"
        # Samplers not included in img2img yet
        if sd_settings['sampler'] != "ddim" and sd_settings['sampler'] != "plms":
            runner_script = "optimizedSD_K/optimized_img2img_k.py"
    else:
        runner_script = "scripts/img2img.py"
        # Samplers not included in img2img yet
        if sd_settings['sampler'] != "ddim" and sd_settings['sampler'] != "plms":
            runner_script = "scripts/img2img_k.py"

prompt_file_path = f"{userprofile}/artroom/settings/prompt.txt"
with open(prompt_file_path, "w") as f:
    f.write(sd_settings['text_prompts'])

script_command = [ f"{userprofile}\\artroom\\miniconda3\\condabin\\activate.bat","&&",
                    f"conda", "run","--no-capture-output", "-p", f"{userprofile}/artroom/miniconda3/envs/artroom-ldm",
                    "python", f"{runner_script}",
                    "--scale", str(sd_settings['cfg_scale']),
                    "--outdir", outdir,
                    "--n_samples", str(sd_settings['n_samples']),
                    "--ddim_steps", str(sd_settings['steps']),
                    "--seed", str(seed),
                    "--ckpt", f"{userprofile}/artroom/model_weights/model.ckpt",
                    "--n_iter", str(sd_settings['n_iter']),
                    "--from-file", prompt_file_path,
                    "--skip_grid"
                    ]    

if sd_config['use_optimized_version']:
    if sd_config['use_turbo']:
        script_command += ["--turbo"]
    if sd_config['use_superfast']:
        script_command += ["--superfast"]

if sd_config['use_full_precision']:
    script_command += ["--precision","full"]

if sd_settings['aspect_ratio'] != "Init Image" or len(sd_settings['init_image']) == 0:
    script_command += ["--W", str(sd_settings['width'])]
    script_command += ["--H", str(sd_settings['height'])]

if sd_settings['init_image'] != "":
    script_command += ["--init-img",sd_settings['init_image']]
    script_command += ["--strength",str(sd_settings['strength'])]
else:    
    if not sd_config['use_optimized_version']:
        if sd_settings['sampler'] == "plms":
            script_command += ["--plms"]
    else:
        if sd_settings['sampler'] == "plms" or sd_settings['sampler'] == "ddim":
            script_command += ["--sampler",sd_settings['sampler']]

    if sd_settings['sampler'] != "plms" and sd_settings['sampler'] != "ddim":
        script_command += ["--sampler", sd_settings['sampler']]

try:
    print("Running....")
    print("If it freezes, please try pressing enter. Doesn't happen often but could happen once in a while")
    image_folder = os.path.join(outdir,re.sub(r'\W+', '',"_".join(sd_settings['text_prompts'].split())))[:150]
    # print(image_folder)
    os.makedirs(image_folder,exist_ok=True)
    sd_json = {"Settings": sd_settings, "Config": sd_config}
    sd_settings_count = len(glob(image_folder+"/*.json"))
    with open(f"{image_folder}/sd_settings_{seed}_{sd_settings_count}.json", "w") as outfile:
        json.dump(sd_json, outfile, indent=4)
    time.sleep(1)
    process = subprocess.run(script_command)
    print("Finished!")
    time.sleep(3)

except Exception as e:
    print(f"ERROR: {e}")
    time.sleep(10)

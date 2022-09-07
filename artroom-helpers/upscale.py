import subprocess
import json
import os
import time
import subprocess
from glob import glob
import shutil
import sys
import re

userprofile = os.environ["USERPROFILE"]
upscale_json = json.load(open(f"{userprofile}/artroom/settings/upscale_settings.json"))

upscale_folder = upscale_json["upscale_folder"]
upscaler = upscale_json["upscaler"]
upscale_factor = upscale_json["upscale_factor"]
upscale_dest = upscale_json["upscale_dest"]
if upscale_dest == "":
    upscale_dest = upscale_folder

images = glob(f"{upscale_folder}/*.png") + glob(f"{upscale_folder}/*.jpg") + glob(f"{upscale_folder}/*.jpeg")
upscale_queue_path = f"{userprofile}/artroom/settings/upscale_queue"
os.makedirs(upscale_queue_path,exist_ok=True)
for image in images:    
    shutil.copy(image,upscale_queue_path)

if upscaler == "GFPGAN":
    os.chdir('stable-diffusion/src/gfpgan')
    script_command = [ f"{userprofile}\\artroom\\miniconda3\\condabin\\activate.bat","&&",
                    f"conda", "run","--no-capture-output", "-p", f"{userprofile}/artroom/miniconda3/envs/artroom-ldm",
                    "python", "inference_gfpgan.py","-i",f"{upscale_queue_path}","-o",f"{upscale_dest}","-v","1.3","-s",f"{upscale_factor}","--bg_upsampler","realesrgan","--suffix","_upscaled"
                    ]    
    print("Running....")
    print("If it freezes, please try pressing enter. Doesn't happen often but could happen once in a while")
    process = subprocess.run(script_command)
    print("Finished!")
    time.sleep(3)
elif upscaler == "RealESRGAN":
    os.chdir('stable-diffusion/src/realesrgan')
    script_command = [ f"{userprofile}\\artroom\\miniconda3\\condabin\\activate.bat","&&",
                    f"conda", "run","--no-capture-output", "-p", f"{userprofile}/artroom/miniconda3/envs/artroom-ldm",
                    "python", "inference_realesrgan.py","-i",f"{upscale_queue_path}","-o",f"{upscale_dest}/upscaled","-s",f"{upscale_factor}","--suffix","_upscaled","--tile","400"
                    ]    
    print("Running....")
    print("If it freezes, please try pressing enter. Doesn't happen often but could happen once in a while")
    process = subprocess.run(script_command)
    print("Finished!")
    time.sleep(3)
elif upscaler == "RealESRGAN-Anime":
    os.chdir('stable-diffusion/src/realesrgan')
    script_command = [ f"{userprofile}\\artroom\\miniconda3\\condabin\\activate.bat","&&",
                f"conda", "run","--no-capture-output", "-p", f"{userprofile}/artroom/miniconda3/envs/artroom-ldm",
                "python", "inference_realesrgan.py","-i",f"{upscale_queue_path}","-o",f"{upscale_folder}/upscaled-anime","-s",f"{upscale_factor}","--suffix","_upscaled","--tile","400",
                "--model", "RealESRGAN_x4plus_anime_6B"
                ]    
    print("Running....")
    print("If it freezes, please try pressing enter. Doesn't happen often but could happen once in a while")
    process = subprocess.run(script_command)
    print("Finished!")
    time.sleep(3)   
else:
    print("FAILURE")
    time.sleep(10)

#Clean up
shutil.rmtree(upscale_queue_path)
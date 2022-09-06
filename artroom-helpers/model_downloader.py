import os
import shutil
from tqdm import tqdm
import json
import time
#import gdown
import requests
import shutil

#Set up settings
userprofile = os.environ["USERPROFILE"]
os.makedirs(f"{userprofile}/artroom/settings/",exist_ok=True)

if not os.path.exists(f"{userprofile}/artroom/settings/upscale_settings.json"):
    shutil.copy("upscale_settings.json",f"{userprofile}/artroom/settings/")
else:
    os.remove(f"{userprofile}/artroom/settings/upscale_settings.json")
    shutil.copy("upscale_settings.json",f"{userprofile}/artroom/settings/")

if not os.path.exists(f"{userprofile}/artroom/settings/error_mode.json"):
    shutil.copy("error_mode.json",f"{userprofile}/artroom/settings/")
else:
    os.remove(f"{userprofile}/artroom/settings/error_mode.json")
    shutil.copy("error_mode.json",f"{userprofile}/artroom/settings/")

if not os.path.exists(f"{userprofile}/artroom/settings/sd_settings.json"):
    shutil.copy("sd_settings.json",f"{userprofile}/artroom/settings/")
else:
    original_json = json.load(open(f"{userprofile}/artroom/settings/sd_settings.json"))
    original_settings = original_json['Settings']
    original_config = original_json['Config']
    new_json = json.load(open("sd_settings.json"))
    new_settings = new_json['Settings']
    new_config = new_json['Config']
    for key in new_settings:
        if key not in original_settings:
            original_settings[key] = new_settings[key]
    for key in new_config:
        if key not in original_config:
            original_config[key] = new_config[key]
    update_original = {
        "Settings": original_settings,
        "Config": original_config
    }
    with open(f"{userprofile}/artroom/settings/sd_settings.json", "w") as outfile:
        json.dump(update_original, outfile, indent=4)

upscale_models = {
    "RealESRGAN": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
    "RealESRGAN-anime": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth",
    "GFPGAN": "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth"
}

print("Downloading upscalers...")
url = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth"
model_name = os.path.basename(url)
model_dest = "stable-diffusion/src/realesrgan/realesrgan/weights/"
#Remove broken download if failed during install
if os.path.exists(model_name):
    os.remove(model_name)
if not os.path.exists(model_dest+model_name):
    print(f"Downloading {model_name}...")
    response = requests.get(url, stream=True)
    total_size_in_bytes= int(response.headers.get('content-length', 0))
    block_size = 1024 #1 Kibibyte
    progress_bar = tqdm(total=total_size_in_bytes, unit='iB', unit_scale=True)
    with open(model_name, 'wb') as file:
        for data in response.iter_content(block_size):
            progress_bar.update(len(data))
            file.write(data)
    progress_bar.close()
    if total_size_in_bytes != 0 and progress_bar.n != total_size_in_bytes:
        print("ERROR, something went wrong")            
    shutil.move(model_name,model_dest)

for model in upscale_models:
    url = upscale_models[model]
    model_name = os.path.basename(url)
    if "RealESRGAN" in model_name:
        model_dest = "stable-diffusion/src/realesrgan/experiments/pretrained_models/"
    elif "GFPGAN" in model:
        model_dest = "stable-diffusion/src/gfpgan/experiments/pretrained_models/"
    #Remove broken download if failed during install
    if os.path.exists(model_name):
        os.remove(model_name)
    if not os.path.exists(model_dest+model_name):
        print(f"Downloading {model_name}...")
        response = requests.get(url, stream=True)
        total_size_in_bytes= int(response.headers.get('content-length', 0))
        block_size = 1024 #1 Kibibyte
        progress_bar = tqdm(total=total_size_in_bytes, unit='iB', unit_scale=True)
        with open(model_name, 'wb') as file:
            for data in response.iter_content(block_size):
                progress_bar.update(len(data))
                file.write(data)
        progress_bar.close()
        if total_size_in_bytes != 0 and progress_bar.n != total_size_in_bytes:
            print("ERROR, something went wrong")            
        shutil.move(model_name,model_dest)

model_dl_json = json.load(open("model_downloader.json"))

possible_sources = model_dl_json["possible_sources"]
sd_path = model_dl_json["sd_path"]
model_path = model_dl_json["model_path"]
model_name = model_dl_json["model_name"]
userprofile = os.environ["USERPROFILE"]

if "%UserProfile%" in model_path:
    model_path = model_path.replace("%UserProfile%",userprofile)

#Remove broken download if failed during install
if os.path.exists("model.ckpt"):
    os.remove("model.ckpt")
os.makedirs(model_path, exist_ok=True)  

try: 
    if os.path.exists(sd_path):
        if os.path.exists(model_path+"/model.ckpt"):
            print("Model found")
        else:
            print("Model not found")
            print("Downloading Model weights..... (if it freezes, try pressing enter)")
            # url = 'https://drive.google.com/uc?id=1W6nSS8at4kjO4ekAGJeFRwnRKIaLfDwS'
            # gdown.download(url, model_name, quiet=False) 
            # shutil.move(model_name,model_path)

            for url in possible_sources:
                try:
                    response = requests.get(url, stream=True)
                    total_size_in_bytes= int(response.headers.get('content-length', 0))
                    block_size = 1024 #1 Kibibyte
                    progress_bar = tqdm(total=total_size_in_bytes, unit='iB', unit_scale=True)
                    with open(model_name, 'wb') as file:
                        for data in response.iter_content(block_size):
                            progress_bar.update(len(data))
                            file.write(data)
                    progress_bar.close()
                    if total_size_in_bytes != 0 and progress_bar.n != total_size_in_bytes:
                        print("ERROR, something went wrong")              
                    shutil.move(model_name,model_path)
                except:
                    pass
                
except Exception as e:
    print(f"Failed downloading model: {e}")
    if "access" in str(e):
        print("Please try running as admin for access")
    time.sleep(5)

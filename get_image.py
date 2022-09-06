from email.mime import image
import json
from glob import glob
import re
import os
import base64
userprofile = os.environ["USERPROFILE"]

sd_json = json.load(open(f"{userprofile}/artroom/settings/sd_settings.json"))
sd_settings = sd_json['Settings']
sd_config = sd_json['Config']

outdir = sd_config['image_save_path'] + "/" + sd_settings['batch_name']
if "%UserProfile%" in outdir:
    userprofile = os.environ["USERPROFILE"]
    outdir = outdir.replace("%UserProfile%",userprofile)

# For grids
# image_folder = outdir
image_folder = os.path.join(outdir,re.sub(r'\W+', '',"_".join(sd_settings['text_prompts'].split())))[:150]

if os.path.exists(image_folder):
    images = glob(image_folder+"/latest.png")
    with open(images[0], "rb") as img_file:
        b64_string = base64.b64encode(img_file.read())
    b64_string = "data:image/png;base64,"+str(b64_string)[2:-1]
    print(b64_string)
else:
    print("None")
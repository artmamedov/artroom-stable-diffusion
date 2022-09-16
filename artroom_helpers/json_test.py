import json
import os
import shutil

userprofile = os.environ["USERPROFILE"]

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
    json.dump(update_original, outfile)
import sys
import json
import os
userprofile = os.environ["USERPROFILE"]

data = json.loads(sys.argv[1])
original = json.load(open(f"{userprofile}/artroom/settings/sd_settings.json"))

data['image_save_path'] = data['image_save_path'].strip()

json_settings = {"Settings": original['Settings'], 
"Config": data
}

with open(f"{userprofile}/artroom/settings/sd_settings.json", "w") as outfile:
    json.dump(json_settings, outfile, indent = 4)

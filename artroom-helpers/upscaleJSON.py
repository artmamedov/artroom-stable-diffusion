import sys
import json
import os
userprofile = os.environ["USERPROFILE"]
data = json.loads(sys.argv[1])

data['upscale_folder'] = data['upscale_folder'].strip()

with open(f"{userprofile}/artroom/settings/upscale_settings.json", "w") as outfile:
    json.dump(data, outfile, indent=4)

print("Done")
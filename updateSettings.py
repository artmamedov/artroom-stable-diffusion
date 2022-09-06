import sys
import json
import os 
userprofile = os.environ["USERPROFILE"]

data = json.loads(sys.argv[1])
original = json.load(open(f"{userprofile}/artroom/settings/sd_settings.json"))

data['batch_name'] = data['batch_name'].strip()

json_settings = {"Settings": data, 
"Config": original['Config']
}

with open(f"{userprofile}/artroom/settings/sd_settings.json", "w") as outfile:
    json.dump(json_settings, outfile, indent = 4)

print("Done!")
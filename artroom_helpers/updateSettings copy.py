import sys
import json
import os 
userprofile = os.environ["USERPROFILE"]

data = json.loads(sys.argv[1])
original = json.load(open(f"{userprofile}/artroom/settings/sd_settings.json"))

data['batch_name'] = data['batch_name'].strip()

#Temporary fixes
if original['Config']['use_optimized_version'] and (original['Config']['use_turbo'] or original['Config']['use_superfast']) and (data['sampler'] != 'ddim'):
    data['sampler']['sampler'] = 'plms'

#Legacy
sampler_format_mapping = {
    "euler": "k_euler",
    "euler_a": "k_euler_ancestral",
    "dpm": "k_dpm_2",
    "dpm_a": "k_dpm_2_ancestral",
    "lms": "k_lms",
    "heun": "k_heun"
}

if data['sampler'] in sampler_format_mapping:
    data['sampler'] = sampler_format_mapping[data['sampler']]                

json_settings = {"Settings": data, 
"Config": original['Config']
}

with open(f"{userprofile}/artroom/settings/sd_settings.json", "w") as outfile:
    json.dump(json_settings, outfile, indent = 4)

print("Done!")
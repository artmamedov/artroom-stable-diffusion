import json
import time
import random
from glob import glob 
import os
import re

def return_error(status, status_message = '', content=''):
    if not status_message and status == 'Failure':
        status_message = 'Unknown Error'
    return {'status': status, 'status_message': status_message, 'content': content}

class QueueManager():
    def __init__(self, SD, artroom_path=None):
        self.SD = SD
        self.artroom_path = artroom_path
        self.queue = []
        self.running = False
        self.delay = 5
        self.thread = None

        self.error = None

        self.read_queue_json()

    def parse_errors(self,error):
        self.error = return_error('Error')
        return self.error 

    def clear_queue(self):
        self.queue = []

    def remove_from_queue(self,id):
        for i, item in enumerate(self.queue):
            if item['id'] == id:
                self.queue.pop(i)

    def update_delay(self,new_delay):
        self.delay = new_delay

    def write_queue_json(self):
        queue_json = {'queue': self.queue}
        with open(f'{self.artroom_path}/artroom/settings/queue.json', 'w') as outfile:
            json.dump(queue_json, outfile, indent=4)     

    def read_queue_json(self):
        if os.path.exists(f'{self.artroom_path}/artroom/settings/queue.json'):
            queue_json = json.load(open(f'{self.artroom_path}/artroom/settings/queue.json'))
            self.queue = queue_json['queue']
        else:
            self.queue = []

        if os.path.exists(f'{self.artroom_path}/artroom/settings/s.json'):
            queue_json = json.load(open(f'{self.artroom_path}/artroom/settings/queue.json'))
            self.queue = queue_json['queue']

    def set_artroom_path(self,path):
        self.artroom_path = path

    def add_to_queue(self,data):
        data['id'] = random.randint(1,922337203685)
        sampler_format_mapping = {
            'k_euler': 'euler',
            'k_euler_ancestral': 'euler_a',
            'k_dpm_2': 'dpm',
            'k_dpm_2_ancestral': 'dpm_a',
            'k_lms': 'lms',
            'k_heun': 'heun'
        }

        if data['sampler'] in sampler_format_mapping:
            data['sampler'] = sampler_format_mapping[data['sampler']]                

        if len(data['image_save_path']) > 0 and data['image_save_path'][-1] != '/':
            data['image_save_path'] += '/'

        if data['mask'] == '':
            data['invert'] = False
        else:
            data['invert'] = data['reverse_mask']

        if data['use_random_seed']:
            data['seed'] = random.randint(1, 4294967295)
        else:
            data['seed'] = int(data['seed'])
        data['steps'] = int(data['steps'])
        data['n_iter'] = int(data['n_iter'])
        data['cfg_scale'] = float(data['cfg_scale'])

        if len(data['init_image']) > 0:
            if 'strength' in data:
                data['strength'] = float(data['strength'])
            else:
                data['strength'] = 0.75
        else:
            data['strength'] = 0.75

        data['device'] = 'cuda'
        data['precision'] = 'full' if data['use_full_precision'] else 'autocast'

        if '%UserProfile%' in data['image_save_path']:
            data['image_save_path'] = data['image_save_path'].replace('%UserProfile%',os.environ["USERPROFILE"])
        if '%UserProfile%' in data['ckpt']:
            data['ckpt'] = data['ckpt'].replace('%UserProfile%',os.environ["USERPROFILE"])
        if '%InstallPath%' in data['ckpt']:
            data['ckpt'] = data['ckpt'].replace('%InstallPath%',self.artroom_path)

        if '%UserProfile%' in data['ckpt_dir']:
            data['ckpt_dir'] = data['ckpt_dir'].replace('%UserProfile%',os.environ["USERPROFILE"])
        if '%InstallPath%' in data['ckpt_dir']:
            data['ckpt_dir'] = data['ckpt_dir'].replace('%InstallPath%',self.artroom_path)

        if data['aspect_ratio'] == 'Init Image':
            #Load image sets it to be equal to init_image dimensions
            data['width'] = 0
            data['height'] = 0

        self.queue.append(data)
        self.save_settings_cache(data)
        self.write_queue_json()

    def save_to_settings_folder(self,data):
        image_folder = os.path.join(self.SD.image_save_path+data['batch_name'],re.sub(r'\W+', '','_'.join(data['text_prompts'].split())))[:150]
        os.makedirs(image_folder,exist_ok=True)
        os.makedirs(image_folder+'/settings',exist_ok=True)
        sd_settings_count = len(glob(image_folder+'/settings/*.json'))
        with open(f"{image_folder}/settings/sd_settings_{data['seed']}_{sd_settings_count}.json", 'w') as outfile:
            json.dump(data, outfile, indent=4)

    def save_settings_cache(self,data):
        with open(f'{self.artroom_path}/artroom/settings/sd_settings.json', 'w') as outfile:
            json.dump(data, outfile, indent=4)

    def generate(self,next_gen):
        mask_b64 = next_gen['mask']
        next_gen['mask'] = ''
        init_image = next_gen['init_image']
        if next_gen['init_image'][:4] == 'data':
            next_gen['init_image'] = 'upload'
        self.save_to_settings_folder(next_gen)
        try:
            self.SD.generate(
            text_prompts=next_gen['text_prompts'],
            negative_prompts=next_gen['negative_prompts'],
            batch_name=next_gen['batch_name'],
            init_image=init_image,
            strength=next_gen['strength'],
            mask = mask_b64,
            invert = next_gen['invert'],
            n_iter=int(next_gen['n_iter']),
            steps=int(next_gen['steps']),
            H=int(next_gen['height']),
            W=int(next_gen['width']),
            seed=int(next_gen['seed']),
            sampler=next_gen['sampler'],
            cfg_scale=float(next_gen['cfg_scale']),
            ckpt=next_gen['ckpt'],
            image_save_path = next_gen['image_save_path'], 
            speed=next_gen['speed'], 
            device = next_gen['device'], 
            precision = next_gen['precision'], 
            skip_grid = not next_gen['save_grid'],
            )
        except Exception as e:
            print(f'Failure: {e}')
            self.parse_errors(e)
            self.running = False
            
    def run_queue(self):
        if not self.running:
            self.running = True
            while(self.running):
                if len(self.queue) > 0 and not self.SD.loading_model:
                    queue_item = self.queue[0]
                    self.generate(queue_item)
                    try:
                        self.remove_from_queue(queue_item['id'])
                    except:
                        pass
                    self.write_queue_json()
                time.sleep(self.delay)
                if len(self.queue) == 0:
                    self.running = False
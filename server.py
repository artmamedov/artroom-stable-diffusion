import os
from flask import Flask, request, jsonify 
from PIL import Image 
import json
import base64
from io import BytesIO
import threading
import re
import ctypes


kernel32 = ctypes.windll.kernel32
kernel32.SetConsoleMode(kernel32.GetStdHandle(-10), 128)

import logging
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)
print("Running in Debug Mode. Please keep CMD window open")

from stable_diffusion import StableDiffusion
from queue_manager import QueueManager
from upscale import Upscaler

def return_output(status, status_message = '', content=''):
    if not status_message and status == 'Failure':
        status_message = 'Unknown Error'
    return jsonify({'status': status, 'status_message': status_message, 'content': content})

def image_to_b64(image):
    image_file = BytesIO()
    image.save(image_file, format='JPEG')
    im_bytes = image_file.getvalue()  # im_bytes: image in binary format.
    imgb64 = base64.b64encode(im_bytes)
    return 'data:image/jpeg;base64,'+str(imgb64)[2:-1]

def b64_to_image(b64):
    image_data = re.sub('^data:image/.+;base64,', '', b64)
    return Image.open(BytesIO(base64.b64decode(image_data))).convert('RGB')

app = Flask(__name__)
SD = StableDiffusion()
UP = Upscaler()

def set_artroom_paths(artroom_path):
    QM.set_artroom_path(artroom_path)
    UP.set_artroom_path(artroom_path)
    SD.set_artroom_path(artroom_path)

user_profile = os.environ['USERPROFILE']
artroom_install_log = f'{user_profile}/AppData/Local/artroom_install.log'
if os.path.exists(artroom_install_log):
    #artroom_path = f'{user_profile}/AppData/Local/artroom_install.log'
    f = open(artroom_install_log,"r")
    artroom_path_raw = f.readline()
    f.close()
    artroom_path = artroom_path_raw[:-1]
else:
    artroom_path = os.environ['USERPROFILE']
QM = QueueManager(SD, artroom_path)
threading.Thread(target=set_artroom_paths,args=[artroom_path], daemon=True).start()

@app.route('/upscale', methods=['POST'])
def upscale():
    data = json.loads(request.data)
    if UP.running:
        return return_output('Failure', 'Upscale is already running')
    if len(data['upscale_images']) == 0:
        return return_output('Failure', 'Please select an image')
    #returns (status,status_message)
    upscale_status = UP.upscale(data['upscale_images'], data['upscaler'], data['upscale_factor'], data['upscale_dest'])
    return return_output(upscale_status[0],upscale_status[1])
        
@app.route('/get_images',methods=['GET'])
def get_images():
    path = request.args.get('path')
    id = int(request.args.get('id'))
    if id == SD.latest_images_id:
        return return_output('Hold','No new updates on images',content={'latest_images_id':SD.latest_images_id, 'latest_images':[]})
    try:
        if path == 'latest':
            imageB64 = [image_to_b64(image) for image in SD.get_latest_images()]
        else:
            image = Image.open(path).convert('RGB')
            imageB64 = image_to_b64(image)
        return return_output('Success',content={'latest_images_id':SD.latest_images_id, 'latest_images':imageB64})
    except:
        return return_output('Failure','Failed to get image',{'latest_images_id':-1,'imageB64':''})
  
@app.route('/get_server_status',methods=['GET'])
def get_server_status():
    running_tasks = []
    user_id = request.args.get('user_id')
    if user_id in running_tasks:
        #running_tasks.status = ['in queue', 'running', 'completed]
        #running_tasks.content = imgb64 or None
        return {'status':running_tasks.status,'status_message':running_tasks.status_message, 'content':running_tasks.content}
    else:
        return {'status':'Not Found','status_message':'Requested image ID not found','content':None}
  

@app.route('/get_progress', methods=['GET'])
def get_progress():
    # try:
        current_num, total_num, current_step, total_step = SD.get_steps()
        if total_step*total_num > 0:
            percentage = (current_num*total_step+current_step)/(total_num*total_step)
        else:
            percentage = 0
        return return_output('Success',content={'current_name': current_num, 'total_num': total_num, 'current_step': current_step, 'total_step': total_step,
        'percentage': int(percentage*100), 'loading_model': SD.loading_model})
    # except:
    #     return return_output('Failure',content={'current_name': 0, 'total_num': 0, 
    #                     'current_step': 0, 'total_step': 0,
    #                     'percentage': 0, 'loading_model': False})

@app.route('/update_settings', methods=['POST'])
def update_settings():
    data = json.loads(request.data)   
    if not SD.artroom_path:
        return return_output('Failure','Artroom Path not found')
    if not os.path.exists(f'{SD.artroom_path}/artroom/settings/sd_settings.json'):
        return return_output('Failure','sd_settings.json not found')
    
    if 'delay' in data:
        QM.set_delay(data['delay'])
    sd_settings = json.load(open(f'{SD.artroom_path}/artroom/settings/sd_settings.json'))
    for key in data:
        value = data[key]
        if type(value) == str and '%UserProfile%' in value:
            value = value.replace('%UserProfile%',os.environ["USERPROFILE"])
        if type(value) == str and '%InstallPath%' in value:
            value = value.replace('%InstallPath%',SD.artroom_path)
        sd_settings[key] = value
    print("Updated Settings")
    with open(f'{SD.artroom_path}/artroom/settings/sd_settings.json', 'w') as outfile:
        json.dump(sd_settings, outfile, indent=4)     
    # SD.load_from_settings_json()
    return return_output('Success')

@app.route('/get_settings', methods=['GET'])
def get_settings():
    if not SD.artroom_path:
        return return_output('Failure','Artroom Path not found')
    if not os.path.exists(f'{SD.artroom_path}/artroom/settings/sd_settings.json'):
        return return_output('Failure','sd_settings.json not found')
    sd_settings = json.load(open(f'{SD.artroom_path}/artroom/settings/sd_settings.json'))
    return return_output('Success',content={'status': QM.queue, 'settings': sd_settings})

@app.route('/get_queue', methods=['GET'])
def get_queue():
    return return_output('Success',content={'queue': QM.queue})

@app.route('/start_queue', methods=['GET'])
def start_queue():
    if not QM.running:
        run_sd()
        print("Running...")
        return return_output('Success')
    else:
        print("Failure...")
        return return_output('Failure')  

@app.route('/stop_queue', methods=['GET'])
def stop_queue():
    if QM.running:
        QM.running = False
        return return_output('Success')
    else:
        return return_output('Failure')  

@app.route('/clear_queue', methods=['POST'])
def clear_queue():
    QM.clear_queue()
    return return_output('Success')

@app.route('/remove_from_queue', methods=['POST'])
def remove_from_queue():
    data = json.loads(request.data)
    QM.remove_from_queue(data['id'])
    return return_output('Success',content={'queue':QM.queue})

@app.route('/add_to_queue', methods=['POST'])
def add_to_queue():
    data = json.loads(request.data)
    if data['ckpt'] == '':
        return return_output('Failure','Model Checkpoint cannot be blank. Please go to Settings and set a model ckpt.')

    QM.add_to_queue(data)
    if not QM.running:
        run_sd()
    return return_output('Success',content={'queue':QM.queue})

@app.route('/start', methods=['POST'])
def run_sd():
    if not QM.running:
        QM.read_queue_json()
        QM.thread = threading.Thread(target=QM.run_queue, daemon=True)
        QM.thread.start()
        return return_output('Success','Starting Artroo m')
    else:
        return return_output('Failure','Already running')


if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5300)
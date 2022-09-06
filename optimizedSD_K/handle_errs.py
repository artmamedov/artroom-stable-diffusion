import os
import json

def process_error_trace(trace, err, prompt_file):
    errorOutPath = os.path.join(os.path.dirname(prompt_file), "error_output.log")
    errorJsonPath = os.path.join(os.path.dirname(prompt_file), "error_mode.json")

    with open(errorOutPath, 'w') as f:
        f.write(str(err))
        f.write(trace)
    
    if "RuntimeError: CUDA out of memory." in trace:     
        update_error_json('CUDA', errorJsonPath)
        raise ValueError('CUDA')
    else:
        update_error_json('UNKNOWN', errorJsonPath)
        raise ValueError('UNKNOWN')

def update_error_json(error_type, errorJsonPath):
    if os.path.exists(errorJsonPath):
        err_json = json.load(open(errorJsonPath))
        err_json['error_code'] = error_type
    else:
        err_json = {
            'error_code' : error_type
        }
    with open(errorJsonPath, "w") as f:
        json.dump(err_json, f)


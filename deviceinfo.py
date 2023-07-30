import os
import os.path as osp
import psutil
import torch
from utils.dictutils import writejson

if __name__ == "__main__":
    ncpu = os.cpu_count()
    gpus = {'total':0, 'detail':{}}
    if torch.cuda.is_available():
        gpus['total'] = torch.cuda.device_count()
        for i in range(gpus['total']):
            devi = torch.cuda.device(f'cuda:{i}')
            namei = torch.cuda.get_device_name(devi)
            gpus['detail'][i] = namei
    I = {
            'cpus':ncpu, 
            'Ram(GiB)':psutil.virtual_memory().total//(1024**3),
            'gpu':gpus,
        }
    writejson(I,osp.join("device.json"))
    print(I)
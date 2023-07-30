import torch

def fix_torch_condition(seed=0):
    
    """
    For experiment reproducibility
    """
    print(f"seed : {seed}")
    torch.manual_seed(seed) 
    torch.cuda.manual_seed_all(seed) 
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_one_torch_device(gpuid:int|str=0)->torch.device:
    dev = torch.device('cpu')
    if torch.cuda.is_available():
        assert int(gpuid) < torch.cuda.device_count()
        dev = torch.device(f'cuda:{gpuid}')
    return dev

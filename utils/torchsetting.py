import torch

def fix_torch_seeds(seed=0):
    
    """
    For experiment reproducable
    """
    print(f"seed : {seed}")
    torch.manual_seed(seed) 
    # sets the seed for generating random numbers.
    torch.cuda.manual_seed(seed) 
    # Sets the seed for generating random numbers for the current GPU. It’s safe to call this function if CUDA is not available; in that case, it is silently ignored.

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_one_torch_device(gpuid:int|str=0)->torch.device:
    dev = torch.device('cpu')
    if torch.cuda.is_available():
        assert int(gpuid) < torch.cuda.device_count()
        dev = torch.device(f'cuda:{gpuid}')
    return dev

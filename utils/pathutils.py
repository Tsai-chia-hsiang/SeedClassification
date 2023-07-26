import os
import shutil

def makedir(d:os.PathLike, rmold=False):
    
    if os.path.exists(d):
        if rmold :
            shutil.rmtree(d)
        else:
            return d
        
    os.mkdir(d)
    return d
    
def check_dir(root:os.PathLike, necessary_subfolers_files = [])->os.PathLike:
    
    for i in necessary_subfolers_files:
        assert os.path.exists(os.path.join(root, i))
        
    return root


def get_newest_index_subdir(root:os.PathLike,based_idx:int, prefix="")->int:
    bidx = based_idx
    while(os.path.exists(os.path.join(root, f"{prefix}{bidx}"))):
        bidx += 1
    return bidx

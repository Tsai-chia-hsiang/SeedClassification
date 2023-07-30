import os
import os.path as osp
import shutil

def makedir(d:os.PathLike, rmold=False):
    
    if osp.exists(d):
        if rmold :
            shutil.rmtree(d)
        else:
            return d
        
    os.mkdir(d)
    return d
    
def check_dir(root:os.PathLike, necessary_subfolers_files:list = None)->os.PathLike:
    nsf = [] if necessary_subfolers_files is None else necessary_subfolers_files
    for i in nsf:
        assert osp.exists(osp.join(root, i))
        
    return root


def get_newest_index_subdir(root:os.PathLike,based_idx:int, prefix="")->int:
    bidx = based_idx
    while(osp.exists(osp.join(root, f"{prefix}{bidx}"))):
        bidx += 1
    return bidx

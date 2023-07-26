from torch.utils.data import Dataset, Subset
from collections import Counter
from .dictutils import revers_dict

def counting_label(dset:Dataset|Subset)->list:
        
    counting,idx_to_class = None,None

    if isinstance(dset, Subset):
        
        counting =  dict(Counter([
            dset.dataset.targets[i] for i in dset.indices
        ]))
        idx_to_class = revers_dict(dset.dataset.class_to_idx)
        #print(idx_to_class)
        
    elif isinstance(dset, Dataset):
        counting = dict(Counter(dset.targets))
        idx_to_class = revers_dict(dset.class_to_idx)

    counting =  {
        idx_to_class[idx]:counting[idx] 
        for idx in range(len(idx_to_class))
    }

    return sorted(counting.items())

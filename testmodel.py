import torch
from torch.utils.data import DataLoader
import os
import os.path as osp
from tqdm import tqdm
import pandas as pd
from utils.parse_argv import parsing_argv
from utils.dictutils import readjson
from utils.torchsetting import get_one_torch_device
from utils.modelutils import TransferResNet50
from utils.dataset import TestIMG

def main(modeldir:os.PathLike, dev:torch.device):

    print(f"using model saved at : {modeldir}")
    print(f"{dev} : {torch.cuda.get_device_name(dev)}")
    
    fc = readjson(osp.join(modeldir,"fc.json"))
    id2class = readjson(osp.join(modeldir,"id2class.json"))
    transrn50 = TransferResNet50(FC=fc)
    transrn50.load_state_dict(
        torch.load(
            osp.join(modeldir,"transferRN50.pth"), 
            map_location='cpu'
        ),
    )
    transrn50.to(device=dev)
    testloader = DataLoader(
        dataset=TestIMG(testdir=osp.join("data","test")),
        batch_size=128, num_workers=os.cpu_count()//2
    )
    prediction = {'file':[],'species':[]}

    for testimg, names in tqdm(testloader):
        pred = transrn50(testimg.to(device=dev))
        _, pred_label = torch.max(pred, 1)

        pred_label = pred_label.cpu().tolist()
        species = list(id2class[f"{pli}"] for pli in pred_label)

        prediction['file'] += names
        prediction['species'] += species
    
    prediction = pd.DataFrame(prediction)
    prediction.to_csv(osp.join("submission.csv"), index=False)

    

if __name__ == "__main__":

    argmap = parsing_argv()
    
    main(
        modeldir = osp.join(
            "model",f"transferRN50_{argmap['modelid']}"
        ),
        dev=get_one_torch_device(
            gpuid = argmap['gpuid'] if 'gpuid' in argmap else 0
        )
    )
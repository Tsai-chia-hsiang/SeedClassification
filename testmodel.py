import torch
from torch.utils.data import DataLoader
import os
import pandas as pd
from utils.dictutils import readjson
from utils.modelutils import TransferResNet50
from utils.dataset import TestIMG
from tqdm import tqdm

def main(modeldir:os.PathLike):
    dev = torch.device('cuda:0')

    fc = readjson(os.path.join(modeldir,"fc.json"))
    id2class = readjson(os.path.join(modeldir,"id2class.json"))
    transrn50 = TransferResNet50(FC=fc)
    transrn50.load_state_dict(
        torch.load(
            os.path.join(modeldir,"transferRN50.pth"), 
            map_location='cpu'
        ),
    )
    transrn50.to(device=dev)
    testloader = DataLoader(
        dataset=TestIMG(testdir=os.path.join("data","test")),
        batch_size=128, num_workers=os.cpu_count()//2
    )
    prediction = {
        'file':[],'species':[]
    }
    for testimg, names in tqdm(testloader):
        pred = transrn50.forward(testimg.to(device=dev))
        _, pred_label = torch.max(pred, 1)

        pred_label = pred_label.cpu().tolist()
        species = list(id2class[f"{pli}"] for pli in pred_label )
        prediction['file'] += names
        prediction['species'] += species
    
    prediction = pd.DataFrame(prediction)
    prediction.to_csv(os.path.join("submission.csv"), index=False)

    

if __name__ == "__main__":
    bestversion = 0
    bestmodeldir = os.path.join("model",f"transferRN50_{bestversion}")
    main(modeldir = bestmodeldir)
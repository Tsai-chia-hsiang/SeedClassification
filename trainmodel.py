import os
import torch
import torch.nn as nn
from utils.dictutils import readjson, writejson
from utils.pathutils import check_dir, makedir, get_newest_index_subdir
from utils.modelutils import transfer_rn50_model
from utils.train import train_model, plotting_loss_and_acc


def main(loader_dir,modelsavingroot, dev:torch.device, hypparas:list):
    
    _ = check_dir(loader_dir, ["tvloader.pth","id2class.json"])
    
    train_val_loader = torch.load(os.path.join(loader_dir, "tvloader.pth"))
    
    idx_to_classes = readjson(os.path.join(loader_dir, "id2class.json"))

    base_saving_idx = get_newest_index_subdir(
        modelsavingroot, based_idx=0,prefix="transferRN50_"
    )

    for i, hyppara in enumerate(hypparas):
        
        modelsavingdir = makedir(
            os.path.join(
                modelsavingroot,f"transferRN50_{i+base_saving_idx}"), 
                rmold=True
        )
        
        FC = hyppara['fc'] + [len(idx_to_classes.keys())]
        
        transfermodel = transfer_rn50_model(
            pretrained_path=os.path.join(modelsavingroot,"rn50pretrained.pth"),
            FC = FC
        )
        print(transfermodel.rn50.fc)
        
        writejson(FC, os.path.join(modelsavingdir,"fc.json"))
        writejson(idx_to_classes,os.path.join(modelsavingdir,"id2class.json"))

        transfermodel = transfermodel.to(device=dev)
            
        lossfunction = nn.CrossEntropyLoss()
        optr = torch.optim.Adam(
            transfermodel.parameters(),lr = hyppara['lr']
        )
            
        history = train_model(
            model=transfermodel, criteria=lossfunction, optr=optr,
            loader=train_val_loader, 
            epochs=hyppara['epochs'], ondevice=dev, 
            modelsavepath = os.path.join(
                modelsavingdir, "transferRN50.pth"
            ) 
        )
        
        writejson(history, os.path.join(modelsavingdir,"traininghist.json"))
        plotting_loss_and_acc(history=history, savedir=modelsavingdir)


if __name__ == "__main__":
    dev = torch.device('cpu')
    if torch.cuda.is_available():
        gpuid = 0
        assert gpuid < torch.cuda.device_count()
        dev = torch.device(f'cuda:{gpuid}')
    
    main(
        loader_dir=os.path.join("data","trainvalloader"), 
        dev=dev,
        modelsavingroot=makedir(os.path.join("model")), 
        hypparas=readjson("hyppara.json")
    )

import os
import os.path as osp
import torch
import torch.nn as nn
from utils.parse_argv import parsing_argv
from utils.torchsetting import fix_torch_condition, get_one_torch_device
from utils.dictutils import readjson, writejson
from utils.pathutils import check_dir, makedir, get_newest_index_subdir
from utils.modelutils import transfer_rn50_model
from utils.train import train_model, plotting_loss_and_acc



def main(loader_dir,modelsavingroot, dev:torch.device, hypparas:list):
    
    print(f"{dev} : {torch.cuda.get_device_name(dev)}")
    
    _ = check_dir(loader_dir, ["tvloader.pth","id2class.json"])
    
    train_val_loader = torch.load(osp.join(loader_dir, "tvloader.pth"))
    idx_to_classes = readjson(osp.join(loader_dir, "id2class.json"))

    base_saving_idx = get_newest_index_subdir(
        modelsavingroot, based_idx=0,prefix="transferRN50_"
    )

    for i, hyppara in enumerate(hypparas):
        
        fix_torch_condition(seed=0)
        
        modelsavingdir = makedir(
            osp.join(
                modelsavingroot,f"transferRN50_{i+base_saving_idx}"), 
                rmold=True
        )
        
        FC = hyppara['fc'] + [len(idx_to_classes.keys())]
        
        transfermodel = transfer_rn50_model(
            pretrained_path=osp.join(modelsavingroot,"rn50pretrained.pth"),
            FC = FC
        )
        print(transfermodel.rn50.fc)
        
        writejson(FC, osp.join(modelsavingdir,"fc.json"))
        writejson(idx_to_classes,osp.join(modelsavingdir,"id2class.json"))

        transfermodel = transfermodel.to(device=dev)
            
        lossfunction = nn.CrossEntropyLoss()

        optr = torch.optim.Adam(
            transfermodel.parameters(),
            lr = hyppara['lr'],
            weight_decay=hyppara['weight_decay'] if 'weight_decay' in hyppara else 0.0
        )
            
        history = train_model(
            model=transfermodel, criteria=lossfunction, optr=optr,
            loader=train_val_loader, 
            epochs=hyppara['epochs'], ondevice=dev, 
            modelsavepath = osp.join(
                modelsavingdir, "transferRN50.pth"
            ) 
        )
        
        writejson(history, osp.join(modelsavingdir,"traininghist.json"))
        plotting_loss_and_acc(history=history, savedir=modelsavingdir)
        
if __name__ == "__main__":
    
    argmap = parsing_argv()
    
    main(
        loader_dir=osp.join("data","trainvalloader"), 
        dev=get_one_torch_device(
            gpuid = argmap['gpuid'] if 'gpuid' in argmap else 0
        ),
        modelsavingroot=makedir(osp.join("model")), 
        hypparas=readjson("hyppara.json")
    )


import os
import torch
import torch.nn as nn
import torchvision
from torchvision.models import resnet50, ResNet50_Weights

class TransferResNet50(nn.Module):

    def __init__(self, FC:list=None) -> None:
        super(TransferResNet50,self).__init__()
        self.rn50 = resnet50(weights=None)
        if FC is not None:
            self.modify_fc(fctrans=FC)
        
    def modify_fc(self, fctrans:list):
        fctrans_ = [self.rn50.fc.in_features] + fctrans
        fclayers = []
        #print(fctrans_)
        for i, li in enumerate(fctrans_):
            fclayers.append(nn.Linear(li, fctrans_[i+1]))
            if i < len(fctrans_)-2:
                fclayers.append(nn.ReLU(inplace=True))
            else:
                break
                
        self.rn50.fc = nn.Sequential(*fclayers)
        
    def forward(self, x:torch.Tensor):
        return self.rn50.forward(x)


def dowload_pretrainrn50(savepath):

    rn50 = resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V2)
    torch.save(rn50.state_dict(),f=savepath)

def transfer_rn50_model(pretrained_path:os.PathLike, FC:list)->TransferResNet50:
    
    trn50 = TransferResNet50()
    trn50.rn50.load_state_dict(
        torch.load(f=pretrained_path, map_location="cpu")
    )
    
    trn50.modify_fc(fctrans=FC)
    
    return trn50

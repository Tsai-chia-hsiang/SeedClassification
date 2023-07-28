import os
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torch.utils.data import random_split
from .pathutils import makedir
from .plotutils import plot_label_count
from .dictutils import revers_dict, writejson
from PIL import Image


resnet50_transformers = transforms.Compose(
    [
    transforms.Resize(256),transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    ]
)

def build_img_train_val_loader(
    root:os.PathLike, train_val_rate:list, batch_size=32,
    **kwargs
)->tuple:

    T = transforms.Compose([transforms.ToTensor()])
    if 't' in kwargs:
        T = kwargs['t']
    dataset = ImageFolder(root=root, transform=T)
    train, val = random_split(dataset, lengths=train_val_rate) 
    
    if 'statics' in kwargs:
        print("train val num counting ..")
        plot_label_count(
            data_distributions=[
                {'dataset':train, "label":"train"},
                {'dataset':val, "label":"validation"}
            ],
            title="Train Validation count", 
            showinline=kwargs['statics'][1], 
            saveto=kwargs['statics'][0], 
            savetype=['jpg','pdf']
        )
    
    return dataset, {
        'train':DataLoader(
            dataset=train, batch_size=batch_size,
            shuffle=True, num_workers=os.cpu_count()//2
        ), 
        'val':DataLoader(
            dataset=val, batch_size=batch_size,
            shuffle=True, num_workers=os.cpu_count()//2
        )
    }, revers_dict(d=dataset.class_to_idx)

def train_valid_split_and_save(
    dataroot, savedir, 
    splitrate=[0.8,0.2], bsize=64, showstatics_barchart=False
):
    
    print(savedir)
    s = makedir(savedir, rmold=True)
    
    alldata, tv, id2class = build_img_train_val_loader(
        root = os.path.join(dataroot, "train"), 
        train_val_rate=splitrate,batch_size=bsize, 
        t=resnet50_transformers, 
        statics=(os.path.join(s,"TrainValcount"),showstatics_barchart)
    )
    torch.save(tv, f=os.path.join(s, "tvloader.pth"))
    writejson(id2class, os.path.join(s, "id2class.json"))


class TestIMG(Dataset):

    def __init__(self, testdir:os.PathLike) -> None:
        
        super().__init__()
        self.fp = [
            os.path.join(testdir, i) for i in os.listdir(testdir)
        ]
        self.T=resnet50_transformers

    def __getitem__(self, index):
        fpi = self.fp[index]
        img = None
        with open(fpi, "rb") as f:
            img = Image.open(f)
            img = img.convert("RGB")
            img = self.T(img)
    
        return (img, os.path.split(fpi)[-1])

    def __len__(self):
        return len(self.fp)

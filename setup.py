import os
from utils.pathutils import makedir
from utils.dataset import train_valid_split_and_save
from utils.modelutils import dowload_pretrainrn50


def dowloadmodel():
    dowload_pretrainrn50(
        savepath=os.path.join(makedir("model"),"rn50pretrained.pth")
    )

def splitdata():
    train_valid_split_and_save(
        dataroot=os.path.join("data"),
        savedir=makedir(os.path.join("data","trainvalloader")),
        bsize=128,splitrate=[0.8,0.2], 
        showstatics_barchart=False
    )

def main():
    r"""
    - Download the pytorch pretrained ResNet50

    - Split image under ```data/train/``` into training & validation set and save them in
    torch DataLoader form.

    """
    #dowloadmodel()
    splitdata()
    

if __name__ == "__main__":
    main()
    
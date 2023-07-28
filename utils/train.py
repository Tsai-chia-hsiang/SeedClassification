import os
import torch
import torch.nn as nn
from torch.nn.modules.loss import _Loss
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm
import sys
from .evaluation import accuracy as acc
from .plotutils import plot_change

def forward_one_epoch(
    model:nn.Module, loss:_Loss, loader:DataLoader,dev:torch.device,
    evafunc:dict, optr:torch.optim.Optimizer=None,
    trainable=True, need_bar = True
):
    torch.set_grad_enabled(trainable)
    if trainable:
        model.train()
    else:
        model.eval()

    total_loss = 0
    total_eva_value = 0
    dnum = 0
    pbar = tqdm(loader) if need_bar else loader

    for x, y in pbar:
        
        if trainable:
            optr.zero_grad()
        imgs = x.to(device=dev)
        labels = y.to(device=dev)
        out = model(imgs)

        l = loss(out, labels)
        evai = evafunc['function'](out, labels, *evafunc['args'])

        if trainable:
            l.backward()
            optr.step()
        
        dnum += x.size()[0]
        total_loss += l*(x.size()[0]) 
        total_eva_value += evai

        if need_bar:
            pbar.set_postfix(ordered_dict={"loss":f"{l:.4f}","eva":evai})

    avg_loss = (total_loss.cpu().item()) / dnum
    avg_eva = total_eva_value / dnum
    
    return avg_loss, avg_eva


def train_model(
    model:nn.Module, loader:dict, criteria:_Loss,
    epochs:int, optr:Optimizer, ondevice:torch.device, 
    modelsavepath:os.PathLike, epochpbar=False, batchpbar=True
)->dict:
    
    history = {
        'loss':{k:[] for k in list(loader.keys())},
        'accuracy':{k:[] for k in list(loader.keys())}
    }
    
    e_itrs = range(epochs)
    pbar = tqdm(e_itrs) if epochpbar else e_itrs
    best_acc = 0

    for _ in pbar:

        for phase in list(loader.keys()):
            need_train = True if phase == 'train' else False
            li, acci = forward_one_epoch(
                model=model, loader=loader[phase], 
                loss=criteria, trainable=need_train,
                optr=optr,evafunc={'function':acc,'args':[False]},
                need_bar=batchpbar,dev=ondevice
            )
            history['loss'][phase].append(li)
            history['accuracy'][phase].append(acci)
            print(f"{phase} epochs : {_} loss:{li}, acc:{acci}")
            if phase == "val":
                if acci > best_acc:

                    if epochpbar:
                        pbar.set_postfix_str(f"Better acc : {acci}")
                    else:
                        print(f"Better acc : {acci}")
                    
                    torch.save(model.state_dict(), f = modelsavepath)
                    best_acc = acci

    return history

def plotting_loss_and_acc(history:dict, savedir:os.PathLike):
    
    opt_choose = None
    for k,v in history.items():
        if k == "accuracy":
            opt_choose = max
        elif k == "loss":
            opt_choose = min
                
        plot_change(
            y=v, title=k, showinline=False, 
            saveto=os.path.join(savedir, k),
            optimal_index=opt_choose,
            savetype=["jpg","pdf"]
        )  
      
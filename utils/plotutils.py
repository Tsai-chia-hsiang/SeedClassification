import os
import sys
from .countlabel import counting_label
import numpy as np
import matplotlib.pyplot as plt

def _outputplot(title:str, showinline=False, saveto=None, savetype=["jpg"]):

    plt.title(title)
    plt.tight_layout()
    
    if saveto is not None:
        
        if not os.path.exists(saveto):
            os.mkdir(saveto)
        
        savename = os.path.split(saveto)[1]
        for s in savetype:
            plt.savefig(
                os.path.join(saveto, f"{savename}.{s}"), 
                format=s
            )
    
    if showinline:
        plt.show()
    plt.close()


def plot_label_count(data_distributions:list, title:str, showinline=False, saveto=None, savetype=["jpg"]):
    
    def addlabels(x,y, base):
        for i in range(len(x)):
            plt.text(
                x = y[i]-25+base[i] ,y = i-0.2, s = y[i], ha = 'center'
            )

    base = None
    
    plt.figure(dpi=800)

    for di in data_distributions:

        class_names,class_counts = map(
            list, zip(*counting_label(di['dataset']))
        )
        if base is None:
            base = np.zeros(len(class_counts))
        plt.barh(
            y=class_names, width=class_counts, left=base, 
            label=di['label'],alpha=0.5
        )
        addlabels(
            x=class_names, y=class_counts, base=base.astype(np.int32)
        )
        base += class_counts
    
    plt.legend()
    _outputplot(
        title=title, 
        showinline=showinline, saveto=saveto, 
        savetype=savetype
    )
    


def plot_change(y:dict, title:str, optimal_index=None, showinline=False, saveto=None, savetype=["jpg"]):
    
    plt.figure(dpi=800)
    colortable=['r','b','g']
    for i, (k,v) in enumerate(y.items()):
        xi = np.arange(len(v))
        
        if optimal_index is not None:
            optvalue = optimal_index(v)
            plt.plot(xi, [optvalue]*len(v), alpha=0.5, linestyle='--',c=colortable[i])
            plt.plot(xi, v, label=f"{k},best={optvalue:.3f}",c=colortable[i])
        else:
            plt.plot(xi, v, label=k,c=colortable[i])
    
    plt.legend()
    _outputplot(title=title, showinline=showinline, saveto=saveto, savetype=savetype)

import torch

def accuracy(x:torch.Tensor, y:torch.Tensor, normalization=False)->int|float:

    def Nmatch(x:torch.Tensor, y:torch.Tensor)->int:
        _, match = torch.max(x, 1)
        n_match = torch.sum(match == y)
        return n_match.cpu().item()

    nmatch = Nmatch(x,y)

    if normalization:
        return nmatch/x.size()[0]
    return nmatch
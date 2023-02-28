import torch.nn.functional as F 
import torch

def D(p, z, version='simplified'): # negative cosine similarity, used as SimCLR loss function
    if version == 'original':
        z = z.detach() # stop gradient
        p = F.normalize(p, dim=1) # l2-normalize 
        z = F.normalize(z, dim=1) # l2-normalize 
        return -(p*z).sum(dim=1).mean()

    elif version == 'simplified':# same thing, much faster. Scroll down, speed test in __main__
        return - F.cosine_similarity(p, z.detach(), dim=-1).mean()
    else:
        raise Exception

def tanh_clip(x, clip_val=20.):
    '''
    soft clip values to the range [-clip_val, +clip_val]
    '''
    if clip_val is not None:
        x_clip = clip_val * torch.tanh((1. / clip_val) * x)
    else:
        x_clip = x
    return x_clip


def loss_xent(logits, labels, ignore_index=-1): # use in linear evaluation (Not sure about CIFAR-10. default hyperparameters are taken from BYOL paper)
    '''
    compute multinomial cross-entropy, for e.g. training a classifier.
    '''
    xent = F.cross_entropy(tanh_clip(logits, 10.), labels,
                           ignore_index=ignore_index)
    lgt_reg = 1e-2 * (logits**2.).mean()
    return xent + lgt_reg
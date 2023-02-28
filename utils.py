import torch
import numpy as np
import logging
import copy
import sys
from PIL import ImageFilter
import random
from torchvision.utils import make_grid
# import matplotlib.pyplot as plt
formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')

def setup_logger(name, log_file, level=logging.INFO, console_out = True):
    """To setup as many loggers as you want"""

    handler = logging.FileHandler(log_file, mode='a')
    handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    while logger.hasHandlers():
        logger.removeHandler(logger.handlers[0])
    logger.addHandler(handler)
    if console_out:
        stdout_handler = logging.StreamHandler(sys.stdout)
        logger.addHandler(stdout_handler)
    return logger

def average_weights(w, pool = None):
    """
    Returns the average of the weights.
    """
    w_avg = copy.deepcopy(w[0].state_dict())
    for key in w_avg.keys():
        if pool is None:
            for i in range(1, len(w)):
                w_avg[key] += w[i].state_dict()[key]
            w_avg[key] = torch.true_divide(w_avg[key], len(w))
        else:
            for i in range(1, len(pool)):
                w_avg[key] += w[pool[i]].state_dict()[key]
            w_avg[key] = torch.true_divide(w_avg[key], len(pool))
    return w_avg

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class DatasetSplit(torch.utils.data.Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        images, labels = self.dataset[self.idxs[item]]
        return images, labels

def get_multiclient_trainloader_list(training_data, num_client, shuffle, num_workers, batch_size, noniid_ratio = 1.0, num_class = 10, hetero = False, hetero_string = "0.2_0.8|16|0.8_0.2"):
    #mearning of default hetero_string = "C_D|B" - dividing clients into two groups, stronger group: C clients has D of the data (batch size = B); weaker group: the other (1-C) clients have (1-D) of the data (batch size = 1).
    if num_client == 1:
        training_loader_list = [torch.utils.data.DataLoader(training_data,  batch_size=batch_size, shuffle=shuffle,
                num_workers=num_workers)]
    elif num_client > 1:
        if noniid_ratio < 1.0:
            training_subset_list = noniid_alllabel(training_data, num_client, noniid_ratio, num_class, hetero, hetero_string) # TODO: implement non_iid_hetero version.
        
        training_loader_list = []
        

        if hetero:
            rich_data_ratio = float(hetero_string.split("|")[-1].split("_")[0])
            rich_data_volume = int(rich_data_ratio * len(training_data))
            rich_client_ratio = float(hetero_string.split("|")[0].split("_")[0])
            rich_client = int(rich_client_ratio * num_client)

        for i in range(num_client):
            # print(f"client {i}:")
            if noniid_ratio == 1.0:
                if not hetero:
                    training_subset = torch.utils.data.Subset(training_data, list(range(i * (len(training_data)//num_client), (i+1) * (len(training_data)//num_client))))
                else:
                    if i < rich_client:
                        training_subset = torch.utils.data.Subset(training_data, list(range(i * (rich_data_volume//rich_client), (i+1) * (rich_data_volume//rich_client))))
                    elif i >= rich_client:
                        heteor_list = list(range(rich_data_volume + (i - rich_client) * (len(training_data) - rich_data_volume) // (num_client - rich_client), rich_data_volume + (i - rich_client + 1) * (len(training_data) - rich_data_volume) // (num_client - rich_client)))
                        training_subset = torch.utils.data.Subset(training_data, heteor_list)
                
            else:
                training_subset = DatasetSplit(training_data, training_subset_list[i])
            # print(len(training_subset))
            if not hetero:
                if num_workers > 0:
                    subset_training_loader = torch.utils.data.DataLoader(
                        training_subset, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size, persistent_workers = True)
                else:
                    subset_training_loader = torch.utils.data.DataLoader(
                        training_subset, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size, persistent_workers = False)
            else:
                if i < rich_client:
                    real_batch_size = batch_size * int(hetero_string.split("|")[1])
                elif i >= rich_client:
                    real_batch_size = batch_size
                if num_workers > 0:
                    subset_training_loader = torch.utils.data.DataLoader(
                        training_subset, shuffle=shuffle, num_workers=num_workers, batch_size=real_batch_size, persistent_workers = True)
                else:
                    subset_training_loader = torch.utils.data.DataLoader(
                        training_subset, shuffle=shuffle, num_workers=num_workers, batch_size=real_batch_size, persistent_workers = False)
                # print(f"batch size is {real_batch_size}")
            training_loader_list.append(subset_training_loader)
    
    return training_loader_list


class Subset(torch.utils.data.Dataset):
    r"""
    Subset of a dataset at specified indices.

    Args:
        dataset (Dataset): The whole Dataset
        indices (sequence): Indices in the whole set selected for subset
    """

    def __init__(self, dataset, indices) -> None:
        self.dataset = dataset
        self.indices = indices

    def __getitem__(self, idx):
        if isinstance(idx, list):
            return self.dataset[[self.indices[i] for i in idx]]
        return self.dataset[self.indices[idx]]

    def __len__(self):
        return len(self.indices)

def noniid_unlabel(dataset, num_users, label_rate, noniid_ratio = 0.2, num_class = 10):
    num_class_per_client = int(noniid_ratio * num_class)
    num_shards, num_imgs = num_class_per_client * num_users, int(len(dataset)/num_users/num_class_per_client)
    idx_shard = [i for i in range(num_shards)]
    dict_users_unlabeled = {i: np.array([], dtype='int64') for i in range(num_users)}
    idxs = np.arange(len(dataset))
    labels = np.arange(len(dataset))  
    

    for i in range(len(dataset)):
        labels[i] = dataset[i][1]
        
    dict_users_labeled = set()

    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:,idxs_labels[1,:].argsort()]
    idxs = idxs_labels[0,:]

    # divide and assign
    for i in range(num_users):
        rand_set = set(np.random.choice(idx_shard, num_class_per_client, replace=False))
        idx_shard = list(set(idx_shard) - rand_set)
        for rand in rand_set:
            dict_users_unlabeled[i] = np.concatenate((dict_users_unlabeled[i], idxs[rand*num_imgs:(rand+1)*num_imgs]), axis=0)

    dict_users_labeled = set(np.random.choice(list(idxs), int(len(idxs) * label_rate), replace=False))
    
    for i in range(num_users):

        dict_users_unlabeled[i] = set(dict_users_unlabeled[i])
        dict_users_unlabeled[i] = dict_users_unlabeled[i] - dict_users_labeled


    return dict_users_labeled, dict_users_unlabeled


# def visualize_classification(loader_iter, labelMap = None, nrofItems = 16, pad = 4, save_name = "unknown"):

#   #Iterate through the data loader
#   imgTensor, labels = next(loader_iter)
  
#   # Generate image grid
#   grid = make_grid(imgTensor[:nrofItems], padding = pad, nrow=nrofItems)

#   # Permute the axis as numpy expects image of shape (H x W x C) 
#   grid = grid.permute(1, 2, 0)

#   # Get Labels
#   if labelMap is not None:
#       labels = [labelMap[lbl.item()] for lbl in labels[:nrofItems]]
#   else:
#       labels = [f"unknown" for lbl in labels[:nrofItems]]
#   # Set up plot config
#   plt.figure(figsize=(8, 2), dpi=300)
#   plt.axis('off')

#   # Plot Image Grid
#   plt.imshow(grid)
  
#   # Plot the image titles
#   fact = 1 + (nrofItems)/100
#   rng = np.linspace(1/(fact*nrofItems), 1 - 1/(fact*nrofItems) , num = nrofItems)
#   for idx, val in enumerate(rng):
#     plt.figtext(val, 0.85, labels[idx], fontsize=8)

#   # Show the plot
# #   plt.show()
#   plt.savefig(f"visual{save_name}.png")


class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x


def divergence_plot(path_to_log, freq = 1):
    file1 = open(path_to_log, 'r')
    Lines = file1.readlines()
    
    count = 0
    divergence_mean_list = []
    divergence_std_list = []
    # Strips the newline character
    divergence_mean = 0
    divergence_std = 0
    for line in Lines:
        
        if "divergence mean:" in line:
            count += 1
            divergence_mean += float(line.split("divergence mean: ")[-1].split(", std:")[0])
            divergence_std += float(line.split(", std: ")[-1].split(" and detailed_list:")[0])
            
            if count % freq == 0:
                
                divergence_mean_list.append(divergence_mean/freq)
                divergence_std_list.append(divergence_std/freq)
                divergence_mean = 0
                divergence_std = 0
                count = 0
    
    return divergence_mean_list, divergence_std_list




def noniid_alllabel(dataset, num_users, noniid_ratio = 0.2, num_class = 10, hetero = False, hetero_string = "0.2_0.8|16|0.8_0.2"):
    num_class_per_client = int(noniid_ratio * num_class)
    # 500 clients -> *5 = 2500 clients.
    if hetero:
        num_shards_multiplier = float(hetero_string.split("|")[-1].split("_")[-1]) # 0.2 (last float)
        num_shards = int(num_class_per_client  * num_users  / num_shards_multiplier) # more shards (equivalent to more clients)
        num_imgs = int(len(dataset)/num_users/num_class_per_client * num_shards_multiplier) # less image
        rich_client_ratio = float(hetero_string.split("|")[0].split("_")[0]) # 0.2 (first float)
        rich_client = int(rich_client_ratio * num_users) # 100 clients
        rich_client_gets_shards = int((1-num_shards_multiplier)/num_shards_multiplier) # each get 4 shards
    else:
        num_shards, num_imgs = num_class_per_client * num_users, int(len(dataset)/num_users/num_class_per_client)
    # print(f"num_shards: {num_shards}, num_imgs: {num_imgs}")

    idx_shard = [i for i in range(num_shards)]
    dict_users_labeled = {i: np.array([], dtype='int64') for i in range(num_users)}
    idxs = np.arange(len(dataset))
    labels = np.arange(len(dataset))
    for i in range(len(dataset)):
        if dataset.__class__.__name__ == "Subset":
            labels[i] = dataset.dataset.targets[dataset.indices[i]] #dataset must be a subset
        else:
            labels[i] = dataset[i][1]

    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:,idxs_labels[1,:].argsort()]
    idxs = idxs_labels[0,:]

    # divide and assign
    if not hetero:
        for i in range(num_users):
            rand_set = set(np.random.choice(idx_shard, num_class_per_client, replace=False))
            idx_shard = list(set(idx_shard) - rand_set)
            for rand in rand_set:
                dict_users_labeled[i] = np.concatenate((dict_users_labeled[i], idxs[rand*num_imgs:(rand+1)*num_imgs]), axis=0)
    else:
        virtual_num_user = rich_client * rich_client_gets_shards + num_users - rich_client
        for i in range(virtual_num_user):
            rand_set = set(np.random.choice(idx_shard, num_class_per_client, replace=False))
            idx_shard = list(set(idx_shard) - rand_set)
            if i < rich_client * rich_client_gets_shards: # assign shards for rich clients
                for rand in rand_set:
                    dict_users_labeled[i // rich_client_gets_shards] = np.concatenate((dict_users_labeled[i // rich_client_gets_shards], idxs[rand*num_imgs:(rand+1)*num_imgs]), axis=0)
            else:
                for rand in rand_set:
                    dict_users_labeled[(i - rich_client * rich_client_gets_shards) + rich_client] = np.concatenate((dict_users_labeled[(i - rich_client * rich_client_gets_shards) + rich_client], idxs[rand*num_imgs:(rand+1)*num_imgs]), axis=0)


    for i in range(num_users):
        # print(f"user {i} has {len(dict_users_labeled[i])} images")
        dict_users_labeled[i] = set(dict_users_labeled[i])

    return dict_users_labeled



if __name__ == '__main__':
    #avgfreq
    avg_freq = 1
    cutlayer = 3
    file_name = f'mocosflV2_ResNet18_cifar10_cut{cutlayer}_bnlNone_client5_nonIID0.2_avg_freq_{avg_freq}'
    # file_name = f'mocosflV2_ResNet18_cifar10_cut{cutlayer}_bnlNone_client5_nonIID0.2'
    path_to_log = f'outputs/divergence/{file_name}/output.log'


    file_name = 'mocofl_ResNet18-cifar10_crosssilo_batchsize128_nonIID0.2_client5_subsample_1.0_local_epoch_5'
    path_to_log = f'outputs/{file_name}/output.log'


    divergence_mean_list, divergence_std_list = divergence_plot(path_to_log, avg_freq)
    print(divergence_mean_list)

    #cutlayer
    # avg_freq = 1
    # cutlayer = 4
    # file_name = f'mocosflV2_ResNet18_cifar10_cut{cutlayer}_bnlNone_client5_nonIID0.2'
    # path_to_log = f'outputs/divergence/{file_name}/output.log'
    # divergence_mean_list, divergence_std_list = divergence_plot(path_to_log, avg_freq)
    # print(divergence_mean_list)
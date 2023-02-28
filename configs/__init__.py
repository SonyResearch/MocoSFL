import argparse
import os
import torch
import numpy as np
import torch
import random

def set_deterministic(seed):
    if seed is not None:
        print(f"Deterministic with seed = {seed}")
        random.seed(seed) 
        np.random.seed(seed) 
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True 
        torch.backends.cudnn.benchmark = False 
    else:
        print("Non-deterministic")

def get_sfl_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--debug', action='store_true')

    # training specific args
    parser.add_argument('--arch', type=str, default='ResNet18', help="which architecture to use")
    parser.add_argument('--dataset', type=str, default='cifar10', help='choose from cifar10, cifar100, imagenet')
    parser.add_argument('--aux_data', type=str, default='cifar100', help='used only in expert_target_aware, as auxiliary dataset, choose from cifar10, cifar100, imagenet')
    parser.add_argument('--num_class', type=int, default=10, help="number of classes: N")
    parser.add_argument('--num_epoch', type=int, default=200)
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--data_dir', type=str, default='./data/')
    parser.add_argument('--output_dir', type=str, default='./outputs/')
    parser.add_argument('--lr', type=float, default=0.05, help="server-side model learning rate")
    parser.add_argument('--c_lr', type=float, default=-1.0, help="client-side model learning rate")
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=0.0005)
    parser.add_argument('--resume', action='store_true', default=False, help="resume from checkpoint")
    parser.add_argument('--data_size', type=int, default=32, help="dimension size of input data")
    parser.add_argument('--avg_freq', type=int, default=1, help="frequency to perform fedavg per round")

    # Split Learning Setting (Basic)
    parser.add_argument('--num_client', type=int, default=1)
    parser.add_argument('--cutlayer', type=int, default=1)
    parser.add_argument('--noniid_ratio', type=float, default=1.0, help="Use non-iid data across clients")
    parser.add_argument('--loss_threshold', type=float, default=0.0, help="Use loss_threshold as loss-based communication saving technique, set to zero to disable it")
    parser.add_argument('--ressfl_alpha', type=float, default=0.0, help="ressfl_alpha, set it to zero to disable it")
    parser.add_argument('--ressfl_target_ssim', type=float, default=0.5, help="ressfl_target_ssim, set it properly to force ssim to be lower than this value")
    parser.add_argument('--augmentation_option', action='store_true', default=False, help="augmentation option for normal training purpose, for moco, augmentation is always on (so irrelavent here)")
    parser.add_argument('--adds_bottleneck', action='store_true', default=False, help="adds_bottleneck")
    parser.add_argument('--disable_WS', action='store_true', default=False, help="disable Weight Standardization")
    parser.add_argument('--disable_c_residual', action='store_true', default=False, help="disable client residule connection if client-side model is ResNet")
    parser.add_argument('--enable_ressfl', action='store_true', default=False, help="enable ressfl")
    parser.add_argument('--load_server', action='store_true', default=False, help="load_server")
    
    # Split Learning Setting (Advanced, including optimization)
    parser.add_argument('--data_proportion', type=float, default=1.0, help="Use subset of iid data")
    parser.add_argument('--client_sample_ratio', type=float, default=1.0, help="client_sample_ratio, sample a subset of clients")
    parser.add_argument('--hetero', action='store_true', default=False, help="if heterogeneous")
    parser.add_argument('--hetero_string', type=str, default="0.2_0.8|16|0.8_0.2", help="string, followed in format of A_B|C|D_E, only valid if heterogeneous")
    parser.add_argument('--bottleneck_option', type=str, default="None", help="string, followed in format of A_B|C|D_E, only valid if heterogeneous")
    parser.add_argument('--MIA_arch', type=str, default="custom", help="simulated MIA architecture, the more complex, the better quality")
    parser.add_argument('--attack', action='store_true', default=False, help="set MIA_attack option")
    parser.add_argument('--auto_adjust', action='store_true', default=False, help="auto_adjust some recommended hyperparameters")
    parser.add_argument('--divergence_measure', action='store_true', default=False, help="for each fedavg, measure model divergence")
    parser.add_argument('--disable_feature_sharing', action='store_true', default=False, help="disable_feature_sharing")
    parser.add_argument('--initialze_path', type=str, default="None", help="set relative path to find the initial model, i.e.: ./outputs/expert/xx")

    # Split Learning Non-IID specific Setting (Advanced)
    parser.add_argument('--divergence_aware', action='store_true', default=False, help="set consistency_loss to False")
    parser.add_argument('--div_lambda', type=float, default=1.0, help="divergence_aware_strength, if enable divergence_aware, increase strength means more personalization")
  
    # Moco setting
    parser.add_argument('--moco_version', type=str, default="V2", help="moco_version: V1, smallV2, V2, largeV2")
    parser.add_argument('--pairloader_option', type=str, default="None", help="set a pairloader option (results in augmentation differences), only enable it in contrastive learning, choice: mocov1, mocov2")
    parser.add_argument('--K', type=int, default=6000, help="max number of keys stored in queue")
    parser.add_argument('--symmetric', action='store_true', default=False, help="enable symmetric contrastive loss, can improve accuracy")
    parser.add_argument('--K_dim', type=int, default=128, help="dimension size of key")
    parser.add_argument('--T', type=float, default=0.1, help="Temperature of InfoCE loss")
    
    # Moco-V2 setting
    parser.add_argument('--mlp', action='store_true', default=False, help="apply MLP head")
    parser.add_argument('--aug_plus', action='store_true', default=False, help="apply extra augmentation (Gaussian Blur))")
    parser.add_argument('--cos', action='store_true', default=False, help="use cosannealing LR scheduler")
    parser.add_argument('--CLR_option', type=str, default="multistep", help="set a client LR scheduling option, no need to mannually set, binding with args.moco_version")
    args = parser.parse_args()

    dataset_name_list = ["cifar10", "cifar100", "imagenet", "svhn", "stl10", "tinyimagenet", "imagenet12"]
    if args.dataset not in dataset_name_list:
        raise NotImplementedError

    if args.c_lr == -1.0: # if client_lr is not set explicitly, set it to global lr
        args.c_lr = args.lr

    '''Auto adjust fedavg frequency, batch size and client sampling ratio according to num_client'''
    if args.auto_adjust:
        if args.num_client <= 50:
            args.batch_size = 100 // args.num_client
        else:
            args.batch_size = 1
        
        if args.num_client <= 1000:
            args.avg_freq = 1000 // (args.batch_size * args.num_client) # num_step per epoch.

        if args.num_client >= 200:
            args.client_sample_ratio = 1 / (args.num_client // 100)
        else:
            args.client_sample_ratio = 1.0

    if args.bottleneck_option == "None":
        args.adds_bottleneck = False
    else:
        args.adds_bottleneck = True

    if args.disable_WS:
        args.WS = False
    else:
        args.WS = True

    if args.disable_feature_sharing:
        args.feature_sharing = False
    else:
        args.feature_sharing = True

    if args.disable_c_residual:
        args.c_residual = False
    else:
        args.c_residual = True

    '''Pre-fix moco version settings '''
    if args.moco_version == "V1":
        args.mlp = False
        args.cos = False
        args.K_dim = 128
        args.pairloader_option = "mocov1"
        args.CLR_option = "multistep"
    elif args.moco_version == "smallV2":
        args.mlp = True # use extra MLP head
        args.cos = True # set cos annearling learning rate decay to true
        args.K_dim = 512
        args.pairloader_option = "mocov2"
        args.symmetric = True
        args.CLR_option = "cos"
    elif args.moco_version == "V2":
        args.mlp = True # use extra MLP head
        args.cos = True # set cos annearling learning rate decay to true
        args.K_dim = 1024
        args.pairloader_option = "mocov2"
        args.symmetric = True
        args.CLR_option = "cos"
    elif args.moco_version == "largeV2": #we adopt the baseline's setting (https://arxiv.org/pdf/2204.04385.pdf)
        args.mlp = True # use extra MLP head
        args.cos = True # set cos annearling learning rate decay to true
        args.K_dim = 2048
        args.pairloader_option = "mocov2"
        args.symmetric = True
        args.CLR_option = "cos"
    elif args.moco_version == "largeV2_aggressive":
        args.mlp = True # use extra MLP head
        # args.cos = True # set cos annearling learning rate decay to true
        args.K_dim = 2048
        args.pairloader_option = "mocov2"
        args.symmetric = True
        args.CLR_option = "aggressive"
    elif args.moco_version == "largeV2_highmomen":
        args.mlp = True # use extra MLP head
        # args.cos = True # set cos annearling learning rate decay to true
        args.K_dim = 2048
        args.pairloader_option = "mocov2"
        args.symmetric = True
        args.CLR_option = "highmomen"
    
    if args.client_sample_ratio != 1.0:
        args.num_epoch = args.num_epoch * int(1/args.client_sample_ratio)

    '''Pre-fix hetero strings '''
    if args.hetero:
        if args.num_client == 500:
            args.hetero_string = "0.2_0.8|16|0.8_0.2"
        elif args.num_client == 200:
            args.hetero_string = "0.5_0.5|4|0.8_0.2"
        elif args.num_client == 100:
            args.hetero_string = "0.2_0.8|16|0.8_0.2"

    # so that no need to set num_class
    if args.dataset == "cifar10":
        args.num_class = 10
        args.data_size = 32
    elif args.dataset == "svhn":
        args.num_class = 10
        args.data_size = 32
    elif args.dataset == "stl10":
        args.num_class = 10
        args.data_size = 96
    elif args.dataset == "tinyimagenet":
        args.num_class = 200
        args.data_size = 64
    elif args.dataset == "cifar100":
        args.num_class = 100
        args.data_size = 32
    elif args.dataset == "imagenet":
        args.num_class = 1000
        args.data_size = 224
    elif args.dataset == "imagenet12":
        args.num_class = 12
        args.data_size = 224
    else:
        raise("UNKNOWN DATASET!")

    assert not None in [args.output_dir, args.data_dir]
    os.makedirs(args.output_dir, exist_ok=True)

    return args



def get_fl_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--debug', action='store_true')
    # training specific args
    parser.add_argument('--arch', type=str, default='ResNet18', help="which architecture to use")
    parser.add_argument('--dataset', type=str, default='cifar10', help='choose from cifar10, cifar100, imagenet')
    parser.add_argument('--num_class', type=int, default=10, help="number of classes: N")
    parser.add_argument('--num_epoch', type=int, default=200)
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--local_epoch', type=int, default=5, help="local epoch per client, used in FL")
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--data_dir', type=str, default='./data/')
    parser.add_argument('--output_dir', type=str, default='./outputs/')
    parser.add_argument('--lr', type=float, default=0.05, help="server-side model learning rate")
    parser.add_argument('--c_lr', type=float, default=-1.0, help="client-side model learning rate")
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=0.0005)
    parser.add_argument('--resume', action='store_true', default=False, help="resume from checkpoint")
    parser.add_argument('--data_size', type=int, default=32, help="dimension size of input data")
    parser.add_argument('--avg_freq', type=int, default=1, help="frequency to perform fedavg per round")
    parser.add_argument('--divergence_measure', action='store_true', default=False, help="for each fedavg, measure model divergence")
    
    # Moco setting
    parser.add_argument('--moco_version', type=str, default="V2", help="moco_version: V1, smallV2, V2, largeV2")
    parser.add_argument('--pairloader_option', type=str, default="None", help="set a pairloader option (results in augmentation differences), only enable it in contrastive learning, choice: mocov1, mocov2")
    parser.add_argument('--K', type=int, default=6000, help="max number of keys stored in queue")
    parser.add_argument('--symmetric', action='store_true', default=False, help="enable symmetric contrastive loss, can improve accuracy")
    parser.add_argument('--K_dim', type=int, default=128, help="dimension size of key")
    parser.add_argument('--T', type=float, default=0.1, help="Temperature of InfoCE loss")
    
    # Moco-V2 setting
    parser.add_argument('--mlp', action='store_true', default=False, help="apply MLP head")
    parser.add_argument('--aug_plus', action='store_true', default=False, help="apply extra augmentation (Gaussian Blur))")
    parser.add_argument('--cos', action='store_true', default=False, help="use cosannealing LR scheduler")
    
    # Federated Learning Setting
    parser.add_argument('--num_client', type=int, default=1)
    parser.add_argument('--cutlayer', type=int, default=1)
    parser.add_argument('--data_proportion', type=float, default=1.0, help="Use subset of iid data")
    parser.add_argument('--client_sample_ratio', type=float, default=1.0, help="client_sample_ratio, sample a subset of clients")
    parser.add_argument('--noniid_ratio', type=float, default=1.0, help="Use non-iid data across clients")
    parser.add_argument('--augmentation_option', action='store_true', default=False, help="set consistency_loss to False")
    parser.add_argument('--divergence_aware', action='store_true', default=False, help="set consistency_loss to False")
    parser.add_argument('--div_lambda', type=float, default=1.0, help="divergence_aware_strength, if enable divergence_aware, increase strength means more personalization")
    parser.add_argument('--hetero', action='store_true', default=False, help="if heterogeneous")
    parser.add_argument('--hetero_string', type=str, default="0.2_0.8|16|0.8_0.2", help="string, followed in format of A_B|C|D_E, only valid if heterogeneous")
    
    args = parser.parse_args()

    dataset_name_list = ["cifar10", "cifar100", "imagenet", "svhn", "stl10", "tinyimagenet"]
    if args.dataset not in dataset_name_list:
        raise NotImplementedError

    if args.c_lr == -1.0: # if client_lr is not set explicitly, set it to global lr
        args.c_lr = args.lr

    if args.client_sample_ratio != 1.0:
        args.num_epoch = args.num_epoch * int(1/args.client_sample_ratio)

    '''Pre-fix moco version settings '''
    if args.moco_version == "V1":
        args.mlp = False
        args.cos = False
        args.K_dim = 128
        args.pairloader_option = "mocov1"
        args.CLR_option = "multistep"
    elif args.moco_version == "smallV2":
        args.mlp = True # use extra MLP head
        args.cos = True # set cos annearling learning rate decay to true
        args.K_dim = 512
        args.pairloader_option = "mocov2"
        args.symmetric = True
        args.CLR_option = "cos"
    elif args.moco_version == "V2":
        args.mlp = True # use extra MLP head
        args.cos = True # set cos annearling learning rate decay to true
        args.K_dim = 1024
        args.pairloader_option = "mocov2"
        args.symmetric = True
        args.CLR_option = "cos"
    elif args.moco_version == "largeV2": #we adopt the baseline's setting (https://arxiv.org/pdf/2204.04385.pdf)
        args.mlp = True # use extra MLP head
        args.cos = True # set cos annearling learning rate decay to true
        args.K_dim = 2048
        args.pairloader_option = "mocov2"
        args.symmetric = True
        args.CLR_option = "cos"
    elif args.moco_version == "largeV2_aggressive":
        args.mlp = True # use extra MLP head
        # args.cos = True # set cos annearling learning rate decay to true
        args.K_dim = 2048
        args.pairloader_option = "mocov2"
        args.symmetric = True
        args.CLR_option = "aggressive"
    elif args.moco_version == "largeV2_highmomen":
        args.mlp = True # use extra MLP head
        # args.cos = True # set cos annearling learning rate decay to true
        args.K_dim = 2048
        args.pairloader_option = "mocov2"
        args.symmetric = True
        args.CLR_option = "highmomen"

    '''Pre-fix hetero strings '''
    if args.hetero:
        if args.num_client == 500:
            args.hetero_string = "0.2_0.8|16|0.8_0.2"
        elif args.num_client == 200:
            args.hetero_string = "0.5_0.5|4|0.8_0.2"
        elif args.num_client == 100:
            args.hetero_string = "0.2_0.8|16|0.8_0.2"

    # so that no need to set num_class
    if args.dataset == "cifar10":
        args.num_class = 10
        args.data_size = 32
    elif args.dataset == "svhn":
        args.num_class = 10
        args.data_size = 32
    elif args.dataset == "stl10":
        args.num_class = 10
        args.data_size = 96
    elif args.dataset == "tinyimagenet":
        args.num_class = 200
        args.data_size = 64
    elif args.dataset == "imagenet12":
        args.num_class = 12
        args.data_size = 224
    elif args.dataset == "cifar100":
        args.num_class = 100
        args.data_size = 32
    elif args.dataset == "imagenet":
        args.num_class = 1000
        args.data_size = 224
    else:
        raise("UNKNOWN DATASET!")

    assert not None in [args.output_dir, args.data_dir]
    os.makedirs(args.output_dir, exist_ok=True)
    # assert args.stop_at_epoch <= args.num_epoch

    return args
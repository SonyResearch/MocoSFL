# %%
'''
MocoSFL-taressfl. This script intends to use as the company expert in ResSFL - can be used with 1-client..
together with large batch size / IID to build up resistance.
But this must be used with a different dataset for tranfer-learning purposes. (i.e. CIFAR-10 build-up for CIFAR-100 use)
Assume company already has its own dataset. This can also be used in super-vised learning, please use run_sfl_ressfl_expert.

'''
from cmath import inf
import datasets
from configs import get_sfl_args, set_deterministic
import torch
import torch.nn as nn
import numpy as np
from models import resnet
from models import vgg
from models import mobilenetv2
from models.resnet import init_weights
from functions.sflmoco_functions import sflmoco_simulator
from functions.sfl_functions import client_backward
from functions.attack_functions import MIA_attacker, MIA_simulator
from functions.pytorch_ssim import SSIM
import gc
VERBOSE = False
#get default args
args = get_sfl_args()
set_deterministic(args.seed)

'''Prefix arguments (resSFL expert)'''
# args.num_client = 1
# args.batch_size = 512
# args.num_workers = 4
# args.noniid_ratio = 1.0
args.attack = True

'''Preparing'''
#get data
create_dataset = getattr(datasets, f"get_{args.dataset}")
train_loader, mem_loader, test_loader = create_dataset(batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True, 
                                                        num_client = args.num_client, data_proportion = args.data_proportion, 
                                                        noniid_ratio = args.noniid_ratio, augmentation_option = True, 
                                                        pairloader_option = args.pairloader_option, hetero = args.hetero, hetero_string = args.hetero_string)

create_dataset_aux = getattr(datasets, f"get_{args.aux_data}")
train_loader_aux, _, _ = create_dataset_aux(batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True, 
                                                        num_client = args.num_client, data_proportion = 1.0, 
                                                        noniid_ratio = 1.0, augmentation_option = True, 
                                                        pairloader_option = args.pairloader_option, hetero = args.hetero, hetero_string = args.hetero_string)


# num_batch = len(train_loader[0])
if args.data_proportion > 0.0:
    num_batch = max(len(train_loader_aux[0]), len(train_loader[0]))
else:
    num_batch = len(train_loader_aux[0])


if "ResNet" in args.arch or "resnet" in args.arch:
    if "resnet" in args.arch:
        args.arch = "ResNet" + args.arch.split("resnet")[-1]
    create_arch = getattr(resnet, args.arch)
    output_dim = 512
elif "vgg" in args.arch:
    create_arch =  getattr(vgg, args.arch)
    output_dim = 512
elif "MobileNetV2" in args.arch:
    create_arch =  getattr(mobilenetv2, args.arch)
    output_dim = 1280
#get model - use a larger classifier, as in Zhuang et al. Divergence-aware paper
global_model = create_arch(cutting_layer=args.cutlayer, num_client = args.num_client, num_class=args.K_dim, group_norm=True, input_size= args.data_size,
                             adds_bottleneck=args.adds_bottleneck, bottleneck_option=args.bottleneck_option, c_residual = args.c_residual, WS = args.WS)

if args.mlp:
    if args.moco_version == "largeV2": # This one uses a larger classifier, same as in Zhuang et al. Divergence-aware paper
        classifier_list = [nn.Linear(output_dim * global_model.expansion, 4096),
                        nn.BatchNorm1d(4096),
                        nn.ReLU(True),
                        nn.Linear(4096, args.K_dim)]
    elif "V2" in args.moco_version:
        classifier_list = [nn.Linear(output_dim * global_model.expansion, args.K_dim * global_model.expansion),
                        nn.ReLU(True),
                        nn.Linear(args.K_dim * global_model.expansion, args.K_dim)]
    else:
        raise("Unknown version! Please specify the classifier.")
    global_model.classifier = nn.Sequential(*classifier_list)
    global_model.classifier.apply(init_weights)
global_model.merge_classifier_cloud()

#get loss function
criterion = nn.CrossEntropyLoss().cuda()

#initialize sfl
sfl = sflmoco_simulator(global_model, criterion, train_loader, test_loader, args)
sfl.cuda()

sfl.log(f"Enable ResSFL fine-tuning: arch-res_normN4C64-alpha-{args.ressfl_alpha}-ssim-{args.ressfl_target_ssim}, using {args.aux_data} as auxiliary data")
ressfl = MIA_simulator(sfl.model, args, "res_normN4C64")
ressfl.cuda()
aux_iterator = iter(train_loader_aux[0])
'''Training'''
if not args.resume:
    sfl.log(f"SFL-Moco (ResSFL-expert) Train on {args.dataset} with cutlayer {args.cutlayer}: total epochs: {args.num_epoch}")
    
    sfl.train()
    #Training scripts (SFL-V1 style)
    knn_accu_max = 0.0

    for epoch in range(1, args.num_epoch + 1):
        
        pool = [0]

        avg_loss = 0.0
        avg_accu = 0.0
        avg_gan_train_loss = 0.0
        avg_gan_eval_loss = 0.0
        for batch in range(num_batch):
            sfl.optimizer_zero_grads()

            #client forward
            if args.data_proportion > 0.0:
                query, pkey = sfl.next_data_batch(0)
                query = query.cuda()
                pkey = pkey.cuda()

            # get aux data
            try:
                query_aux, pkey_aux = next(aux_iterator)
                if query_aux.size(0) != args.batch_size:
                    try:
                        next(aux_iterator)
                    except StopIteration:
                        pass
                    aux_iterator = iter(train_loader_aux[0])
                    query_aux, pkey_aux = next(aux_iterator)
            except StopIteration:
                aux_iterator = iter(train_loader_aux[0])
                query_aux, pkey_aux = next(aux_iterator)
            
            query_aux = query_aux.cuda()
            pkey_aux = pkey_aux.cuda()

            if args.data_proportion > 0.0:
                query_all = torch.cat([query, query_aux], dim = 0)
                pkey_all = torch.cat([pkey, pkey_aux], dim = 0)
            else:
                query_all = query_aux
                pkey_all = pkey_aux


            hidden_query = sfl.c_instance_list[0](query_all)# pass to online \

            with torch.no_grad():
                hidden_pkey = sfl.c_instance_list[0].t_model(pkey_all).detach() # pass to target \

            # locally simulate an attacker here, which one is easier to attack?
            # let's use the query to train the AE (train three times)
            gan_train_loss = ressfl.train(0, hidden_query, query_all)

            #client attacker-aware training loss
            gan_eval_loss, gan_grad = ressfl.regularize_grad(0, hidden_query, query_all)
            
            sfl.s_optimizer.zero_grad()
            
            #server compute
            loss, gradient, accu = sfl.s_instance.compute(hidden_query, hidden_pkey)
            
            sfl.s_optimizer.step() # with reduced step, to simulate a large batch size.

            avg_loss += loss
            avg_accu += accu
            avg_gan_train_loss += gan_train_loss
            avg_gan_eval_loss += gan_eval_loss

            #client backward
            if gan_grad is not None:
                # sfl.c_instance_list[0].backward(gradient)
                # sfl.c_instance_list[0].backward(gan_grad)
                sfl.c_instance_list[0].backward(gradient + gan_grad)
            else:
                sfl.c_instance_list[0].backward(gradient)
            sfl.c_optimizer_list[0].step()
            
            gc.collect()
        
        # adjusting learning rate
        ressfl.scheduler_step()
        sfl.c_scheduler_list[0].step()
        sfl.s_scheduler.step()

        avg_accu = avg_accu / num_batch
        avg_loss = avg_loss / num_batch
        avg_gan_train_loss = avg_gan_train_loss / num_batch
        avg_gan_eval_loss = avg_gan_eval_loss / num_batch
        
        knn_val_acc = sfl.knn_eval(memloader=mem_loader)

        if knn_val_acc > knn_accu_max:
            knn_accu_max = knn_val_acc
            sfl.save_model(epoch, is_best = True)
        sfl.log(f"epoch:{epoch}, knn_val_accu: {knn_val_acc:.2f}, contrast_loss: {avg_loss:.2f}, contrast_acc: {avg_accu:.2f}, gan_train_loss: {avg_gan_train_loss:.4f}, gan_eval_loss: {avg_gan_eval_loss:.4f}")
        gc.collect()
    

sfl.load_model() # load model that has the lowest contrastive loss.

if not args.resume:
    '''Testing Accuracy'''
    val_acc = sfl.knn_eval(memloader=mem_loader)
    sfl.log(f"final knn evaluation accuracy is {val_acc:.2f}")
    create_train_dataset = getattr(datasets, f"get_{args.dataset}_trainloader")

    eval_loader = create_train_dataset(128, args.num_workers, False, 1, 1.0, 1.0, False)
    val_acc = sfl.linear_eval(eval_loader, 100)
    sfl.log(f"final linear-probe evaluation accuracy is {val_acc:.2f}")

    eval_loader = create_train_dataset(128, args.num_workers, False, 1, 0.1, 1.0, False)
    val_acc = sfl.semisupervise_eval(eval_loader, 100)
    sfl.log(f"final semi-supervised evaluation accuracy with 10% data is {val_acc:.2f}")

    eval_loader = create_train_dataset(128, args.num_workers, False, 1, 0.01, 1.0, False)
    val_acc = sfl.semisupervise_eval(eval_loader, 100)
    sfl.log(f"final semi-supervised evaluation accuracy with 1% data is {val_acc:.2f}")

if args.attack:
    # sfl.load_model() # load model that has the lowest contrastive loss.
    val_acc = sfl.knn_eval(memloader=mem_loader)
    sfl.log(f"final knn evaluation accuracy is {val_acc:.2f}")
    
    if args.data_proportion > 0.0:
        '''Evaluate Privacy'''
        MIA = MIA_attacker(sfl.model, train_loader, args, "res_normN4C64")
        # MIA = MIA_attacker(sfl.model, train_loader, args, "custom")
        MIA.MIA_attack()

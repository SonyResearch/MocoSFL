# %%
'''
FLmoco micro-batch baseline.

1. (optional) can use server multi-step.
2. Moco-V2 implemented
3. Moco-V3 symmetric loss implemented
4. Support resource/data heterogeneity
'''
from cmath import inf
import datasets
from configs import get_fl_args, set_deterministic
import torch
import torch.nn as nn
import numpy as np
from models import resnet
from models import vgg
from models.resnet import init_weights
from functions.flmoco_functions import flmoco_simulator
import gc
VERBOSE = False
#get default args
args = get_fl_args()
set_deterministic(args.seed)

'''Preparing'''
#get data
create_dataset = getattr(datasets, f"get_{args.dataset}")
train_loader, mem_loader, test_loader = create_dataset(batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True, 
                                                        num_client = args.num_client, data_proportion = args.data_proportion, 
                                                        noniid_ratio = args.noniid_ratio, augmentation_option = True, 
                                                        pairloader_option = args.pairloader_option, hetero = args.hetero, hetero_string = args.hetero_string)
num_batch = len(train_loader[0]) * args.local_epoch

if "ResNet" in args.arch or "resnet" in args.arch:
    if "resnet" in args.arch:
        args.arch = "ResNet" + args.arch.split("resnet")[-1]
    create_arch = getattr(resnet, args.arch)
elif "vgg" in args.arch:
    create_arch =  getattr(vgg, args.arch)

#get model - use a larger classifier, as in Zhuang et al. Divergence-aware paper
global_model = create_arch(cutting_layer=args.cutlayer, num_client = args.num_client, num_class=args.K_dim, group_norm=True, input_size= args.data_size)

# editing the MLP head
if args.mlp:
    if args.moco_version == "largeV2": # This one uses a larger classifier, same as in Zhuang et al. Divergence-aware paper
        classifier_list = [nn.Linear(512 * global_model.expansion, 4096),
                        nn.BatchNorm1d(4096),
                        nn.ReLU(True),
                        nn.Linear(4096, args.K_dim)]
    elif "V2" in args.moco_version:
        classifier_list = [nn.Linear(512 * global_model.expansion, args.K_dim * global_model.expansion),
                        nn.ReLU(True),
                        nn.Linear(args.K_dim * global_model.expansion, args.K_dim)]
    else:
        raise("Unknown version! Please specify the classifier.")
    global_model.classifier = nn.Sequential(*classifier_list)
    global_model.classifier.apply(init_weights)
global_model.merge_classifier_cloud()

#get loss function
criterion = nn.CrossEntropyLoss().cuda()

#initialize fl-moco
fl = flmoco_simulator(global_model, criterion, train_loader, test_loader, args)
fl.cuda()

'''Training'''
if not args.resume:
    fl.log(f"FL-Moco-microbatch (Moco-{args.moco_version}, Hetero: {args.hetero}, Sample_Ratio: {args.client_sample_ratio}) Train on {args.dataset} with cutlayer {args.cutlayer} and {args.num_client} clients with {args.noniid_ratio}-data-distribution: total epochs: {args.num_epoch}, total number of batches for each client is {num_batch}")
    if args.hetero:
        fl.log(f"Hetero setting: {args.hetero_string}")
    
    fl.train()
    #Training scripts (FL style)
    knn_accu_max = 0.0
    
    for epoch in range(1, args.num_epoch + 1):
        
        if args.client_sample_ratio == 1.0:
            pool = range(args.num_client)
        else:
            pool = np.random.choice(range(args.num_client), int(args.client_sample_ratio * args.num_client), replace=False) # 10 out of 1000
        
        avg_loss = 0.0
        avg_accu = 0.0

        for batch in range(num_batch):

            fl.optimizer_zero_grads()

            #client forward
            for i, client_id in enumerate(pool): # if distributed, this can be parallelly done.
                
                query, pkey = fl.next_data_batch(client_id)

                query = query.cuda()
                pkey = pkey.cuda()

                loss, accu = fl.c_instance_list[client_id].compute(query, pkey)

                fl.c_optimizer_list[client_id].step()

                if VERBOSE and (batch% 50 == 0 or batch == num_batch - 1):
                    fl.log(f"epoch {epoch} batch {batch}, loss {loss}")
                avg_loss += loss
                avg_accu += accu
            gc.collect()
            if batch == num_batch - 1 or (batch % (num_batch//args.avg_freq) == (num_batch//args.avg_freq) - 1):
                # sync client-side models
                divergence_list = fl.fedavg(pool, divergence_aware = args.divergence_aware, divergence_measure = args.divergence_measure)
                if divergence_list is not None:
                    fl.log(f"divergence mean: {np.mean(divergence_list)}, std: {np.std(divergence_list)} and detailed_list: {divergence_list}")
                    pass #TODO: implement divergence measure
                    # if i in pool: # if current client is selected.
                    #     weight_divergence = 0.0
                    #     for key in global_weights.keys():
                    #         if "running" in key or "num_batches" in key: # skipping batchnorm running stats
                    #             continue
                    #         weight_divergence += torch.linalg.norm(torch.flatten(self.model.local_list[i].state_dict()[key] - global_weights[key]).float(), dim = -1, ord = 2)
                        
        # adjusting learning rate
        for i in range(args.num_client):
            fl.c_scheduler_list[i].step()

        avg_accu = avg_accu / num_batch / args.local_epoch / len(pool)
        avg_loss = avg_loss / num_batch / args.local_epoch / len(pool)
        knn_val_acc = fl.knn_eval(memloader=mem_loader)
        if knn_val_acc > knn_accu_max:
            knn_accu_max = knn_val_acc
            fl.save_model(epoch, is_best = True)
        fl.log(f"epoch:{epoch}, knn_val_accu: {knn_val_acc:.2f}, contrast_loss: {avg_loss:.2f}, contrast_acc: {avg_accu:.2f}")
        gc.collect()
'''Testing'''
fl.load_model() # load model that has the lowest contrastive loss.

 # finally, do a thorough evaluation.
val_acc = fl.knn_eval(memloader=mem_loader)
fl.log(f"final knn evaluation accuracy is {val_acc:.2f}")

create_train_dataset = getattr(datasets, f"get_{args.dataset}_trainloader")

mem_loader = create_train_dataset(128, args.num_workers, False, 1, 1.0, 1.0, False)
val_acc = fl.linear_eval(mem_loader, 100)
fl.log(f"final linear-probe evaluation accuracy is {val_acc:.2f}")

mem_loader = create_train_dataset(128, args.num_workers, False, 1, 0.1, 1.0, False)
val_acc = fl.semisupervise_eval(mem_loader, 100)
fl.log(f"final semi-supervised evaluation accuracy with 10% data is {val_acc:.2f}")

mem_loader = create_train_dataset(128, args.num_workers, False, 1, 0.01, 1.0, False)
val_acc = fl.semisupervise_eval(mem_loader, 100)
fl.log(f"final semi-supervised evaluation accuracy with 1% data is {val_acc:.2f}")

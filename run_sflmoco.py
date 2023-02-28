# %%
'''
MocoSFL
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
from functions.sfl_functions import client_backward, loss_based_status
from functions.attack_functions import MIA_attacker, MIA_simulator
import gc
VERBOSE = False
#get default args
args = get_sfl_args()
set_deterministic(args.seed)

'''Preparing'''
#get data
create_dataset = getattr(datasets, f"get_{args.dataset}")
train_loader, mem_loader, test_loader = create_dataset(batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True, 
                                                        num_client = args.num_client, data_proportion = args.data_proportion, 
                                                        noniid_ratio = args.noniid_ratio, augmentation_option = True, 
                                                        pairloader_option = args.pairloader_option, hetero = args.hetero, hetero_string = args.hetero_string)
num_batch = len(train_loader[0])

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

'''Initialze with ResSFL resilient model ''' 
if args.initialze_path != "None":
    sfl.log("Load from resilient model, train with client LR of {}".format(args.c_lr))
    sfl.load_model_from_path(args.initialze_path, load_client = True, load_server = args.load_server)
    args.attack = True

if args.cutlayer > 1:
    sfl.cuda()
else:
    sfl.cpu()
sfl.s_instance.cuda()

'''ResSFL training''' 
if args.enable_ressfl:
    sfl.log(f"Enable ResSFL fine-tuning: arch-{args.MIA_arch}-alpha-{args.ressfl_alpha}-ssim-{args.ressfl_target_ssim}")
    ressfl = MIA_simulator(sfl.model, args, args.MIA_arch)
    ressfl.cuda()
    args.attack = True

'''Training'''
if not args.resume:
    sfl.log(f"SFL-Moco-microbatch (Moco-{args.moco_version}, Hetero: {args.hetero}, Sample_Ratio: {args.client_sample_ratio}) Train on {args.dataset} with cutlayer {args.cutlayer} and {args.num_client} clients with {args.noniid_ratio}-data-distribution: total epochs: {args.num_epoch}, total number of batches for each client is {num_batch}")
    if args.hetero:
        sfl.log(f"Hetero setting: {args.hetero_string}")
    
    sfl.train()
    #Training scripts (SFL-V1 style)
    knn_accu_max = 0.0

    #heterogeneous resources setting
    if args.hetero:
        rich_clients = int(float(args.hetero_string.split("|")[0].split("_")[0]) * args.num_client)
        rich_clients_batch_size = int(float(args.hetero_string.split("|")[1]) * args.batch_size)
    
    loss_status = loss_based_status(loss_threshold = args.loss_threshold)
    
    for epoch in range(1, args.num_epoch + 1):
        
        if args.loss_threshold > 0.0:
            print(f"loss_status: {loss_status.status}")

        if loss_status.status == "C":
            shuffle_map = np.random.permutation(range(num_batch)) # shuffle map for communicate

        if args.client_sample_ratio == 1.0:
            pool = range(args.num_client)
        else:
            pool = np.random.choice(range(args.num_client), int(args.client_sample_ratio * args.num_client), replace=False) # 10 out of 1000
        
        avg_loss = 0.0
        avg_accu = 0.0
        avg_gan_train_loss = 0.0
        avg_gan_eval_loss = 0.0
        for batch in range(num_batch):
            sfl.optimizer_zero_grads()

            if loss_status.status == "A" or loss_status.status == "B":
                hidden_query_list = [None for _ in range(len(pool))]
                hidden_pkey_list = [None for _ in range(len(pool))]

                #client forward
                for i, client_id in enumerate(pool): # if distributed, this can be parallelly done.
                    query, pkey = sfl.next_data_batch(client_id)

                    if args.cutlayer > 1:
                        query = query.cuda()
                        pkey = pkey.cuda()
                    hidden_query = sfl.c_instance_list[client_id](query)# pass to online 
                    hidden_query_list[i] = hidden_query
                    with torch.no_grad():
                        hidden_pkey = sfl.c_instance_list[client_id].t_model(pkey).detach() # pass to target 
                    hidden_pkey_list[i] = hidden_pkey

                stack_hidden_query = torch.cat(hidden_query_list, dim = 0)
                stack_hidden_pkey = torch.cat(hidden_pkey_list, dim = 0)

                if args.loss_threshold > 0.0:
                    torch.save(stack_hidden_query, f"replay_tensors/stack_hidden_query_{batch}.pt")
                    torch.save(stack_hidden_pkey, f"replay_tensors/stack_hidden_pkey_{batch}.pt")
            else:
                
                stack_hidden_query = torch.load(f"replay_tensors/stack_hidden_query_{shuffle_map[batch]}.pt")
                stack_hidden_pkey = torch.load(f"replay_tensors/stack_hidden_pkey_{shuffle_map[batch]}.pt")
            
            stack_hidden_query = stack_hidden_query.cuda()
            stack_hidden_pkey = stack_hidden_pkey.cuda()

            sfl.s_optimizer.zero_grad()
            #server compute
            loss, gradient, accu = sfl.s_instance.compute(stack_hidden_query, stack_hidden_pkey, pool = pool)

            sfl.s_optimizer.step() # with reduced step, to simulate a large batch size.

            if VERBOSE and (batch% 50 == 0 or batch == num_batch - 1):
                sfl.log(f"epoch {epoch} batch {batch}, loss {loss}")
            avg_loss += loss
            avg_accu += accu

            # distribute gradients to clients
            if args.cutlayer <= 1:
                gradient = gradient.cpu()

            if loss_status.status == "A":
                # Initialize clients' queue, to store partial gradients
                gradient_dict = {key: [] for key in range(len(pool))}

                if not args.hetero:
                    for j in range(len(pool)):
                        gradient_dict[j] = gradient[j*args.batch_size:(j+1)*args.batch_size, :]
                else:
                    start_grad_idx = 0
                    for j in range(len(pool)):
                        if (pool[j]) < rich_clients: # if client is rich. Implement hetero backward.
                            gradient_dict[j] = gradient[start_grad_idx: start_grad_idx + rich_clients_batch_size]
                            start_grad_idx += rich_clients_batch_size
                        else:
                            gradient_dict[j] = gradient[start_grad_idx: start_grad_idx + args.batch_size]
                            start_grad_idx += args.batch_size
                
                if args.enable_ressfl:
                    for i, client_id in enumerate(pool): # if distributed, this can be parallelly done.
                        # let's use the query to train the AE
                        gan_train_loss = ressfl.train(client_id, hidden_query, query)
                        #client attacker-aware training loss
                        gan_eval_loss, gan_grad = ressfl.regularize_grad(client_id, hidden_query, query)

                        if gan_grad is not None:
                            gradient_dict[j] += gan_grad

                        avg_gan_train_loss += gan_train_loss
                        avg_gan_eval_loss += gan_eval_loss

                #client backward
                client_backward(sfl, pool, gradient_dict)
            else:
                # (optional) step client scheduler (lower its LR)
                pass

            gc.collect()

            if batch == num_batch - 1 or (batch % (num_batch//args.avg_freq) == (num_batch//args.avg_freq) - 1):
                # sync client-side models
                divergence_list = sfl.fedavg(pool, divergence_aware = args.divergence_aware, divergence_measure = args.divergence_measure)
                if divergence_list is not None:
                    sfl.log(f"divergence mean: {np.mean(divergence_list)}, std: {np.std(divergence_list)} and detailed_list: {divergence_list}")
                    
        sfl.s_scheduler.step()

        avg_accu = avg_accu / num_batch
        avg_loss = avg_loss / num_batch
        if args.enable_ressfl:
            avg_gan_train_loss = avg_gan_train_loss / num_batch / len(pool)
            avg_gan_eval_loss = avg_gan_eval_loss / num_batch / len(pool)
        
        loss_status.record_loss(epoch, avg_loss)
        
        knn_val_acc = sfl.knn_eval(memloader=mem_loader)
        if args.cutlayer <= 1:
            sfl.c_instance_list[0].cpu()
        if knn_val_acc > knn_accu_max:
            knn_accu_max = knn_val_acc
            sfl.save_model(epoch, is_best = True)
        epoch_logging_msg = f"epoch:{epoch}, knn_val_accu: {knn_val_acc:.2f}, contrast_loss: {avg_loss:.2f}, contrast_acc: {avg_accu:.2f}"
        
        if args.enable_ressfl:
            epoch_logging_msg += f", gan_train_loss: {avg_gan_train_loss:.2f}, gan_eval_loss: {avg_gan_eval_loss:.2f}"
        
        sfl.log(epoch_logging_msg)
        gc.collect()
if args.loss_threshold > 0.0:
    saving = loss_status.epoch_recording["C"] + loss_status.epoch_recording["B"]/2
    sfl.log(f"Communiation saving: {saving} / {args.num_epoch}")


'''Testing'''
sfl.load_model() # load model that has the lowest contrastive loss.
# finally, do a thorough evaluation.
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
    '''Evaluate Privacy'''
    if args.resume:
        sfl.load_model() # load model that has the lowest contrastive loss.
    val_acc = sfl.knn_eval(memloader=mem_loader)
    sfl.log(f"final knn evaluation accuracy is {val_acc:.2f}")
    MIA = MIA_attacker(sfl.model, train_loader, args, "res_normN4C64")
    MIA.MIA_attack()
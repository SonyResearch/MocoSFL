import socket
import torch
import pickle

from cmath import inf
from os.path import dirname, abspath
parent_dir = dirname(dirname(abspath(__file__))) # /home/kristina/desire-directory
import sys
sys.path.append(f'{parent_dir}/')
import datasets
from configs import get_sfl_args, set_deterministic
import torch
import torch.nn as nn
import numpy as np
from models import resnet
from models import vgg
from models.resnet import init_weights
from functions.sflmoco_functions import sflmoco_simulator
import gc
import time
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
                                                        pairloader_option = args.pairloader_option, hetero = args.hetero, hetero_string = args.hetero_string, path_to_data = f"{parent_dir}/data")
del train_loader[1:]
num_batch = len(train_loader[0])

host='43.6.20.229' #set client ip and port
port = 4005

server = ('43.6.20.157', 4000) # set server ip and port

s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
s.bind((host,port))

message = "start"
s.sendto(message.encode('utf-8'), server)

def client_receive():
    packed_tensor = s.recv(16384)
    tensorA = pickle.loads(packed_tensor)
    return tensorA

def client_send(tensor, addr):
    message = pickle.dumps(tensor)
    s.sendto(message, server)

if "ResNet" in args.arch or "resnet" in args.arch:
    if "resnet" in args.arch:
        args.arch = "ResNet" + args.arch.split("resnet")[-1]
    create_arch = getattr(resnet, args.arch)
elif "vgg" in args.arch:
    create_arch =  getattr(vgg, args.arch)

args.num_client = 1
#get model - use a larger classifier, as in Zhuang et al. Divergence-aware paper
global_model = create_arch(cutting_layer=args.cutlayer, num_client = args.num_client, num_class=args.K_dim, group_norm=True, input_size= args.data_size,
                             adds_bottleneck=args.adds_bottleneck, bottleneck_option=args.bottleneck_option, c_residual = args.c_residual, WS = args.WS)

global_model.merge_classifier_cloud()
del global_model.cloud
global_model.cloud = None
#get loss function
criterion = nn.CrossEntropyLoss()

#initialize sfl
sfl = sflmoco_simulator(global_model, criterion, train_loader, test_loader, args)

'''Initialze with ResSFL resilient model ''' 
if args.initialze_path != "None":
    sfl.log("Load from resilient model, train with client LR of {}".format(args.c_lr))
    checkpoint_c = torch.load(args.initialze_path + '/checkpoint_c_best.tar', map_location=torch.device('cpu'))
    sfl.model.local_list[0].load_state_dict(checkpoint_c)
sfl.cpu()

'''Training'''
sfl.log(f"SFL-Moco-microbatch (Moco-{args.moco_version}, Hetero: {args.hetero}, Sample_Ratio: {args.client_sample_ratio}) Train on {args.dataset} with cutlayer {args.cutlayer} and {args.num_client} clients with {args.noniid_ratio}-data-distribution: total epochs: {args.num_epoch}, total number of batches for each client is {num_batch}")

sfl.train()

for epoch in range(1, args.num_epoch + 1):
    
    for batch in range(num_batch):
        sfl.optimizer_zero_grads()

        #see whether this client is selected
        select_msg = client_receive()
        if select_msg == "start":
            start_time = time.time()
            
            #client forward
            query, pkey = sfl.next_data_batch(0)

            hidden_query = sfl.c_instance_list[0](query)# pass to online 
            with torch.no_grad():
                hidden_pkey = sfl.c_instance_list[0].t_model(pkey).detach() # pass to target 

            client_send(hidden_query, server)
            client_send(hidden_pkey, server)

            if args.c_lr > 0.0:
                #client backward
                gradient = client_receive()
                sfl.c_instance_list[0].backward(gradient)
                sfl.c_optimizer_list[0].step()
                sfl.c_scheduler_list[0].step()

                gc.collect()
            if args.c_lr > 0.0:
                if batch == num_batch - 1 or (batch % (num_batch//args.avg_freq) == (num_batch//args.avg_freq) - 1):
                    # send server the updated model
                    client_send(sfl.model.local_list[0].state_dict(), server)

                    #receive the synced model from server
                    new_local_state_dict = client_receive()
                    sfl.model.local_list[0].load_state_dict(new_local_state_dict)
                
                gc.collect()
            time_per_step = time.time() - start_time

            sfl.log(f"epoch {epoch}, batch {batch}: client forward, spend {time_per_step}s")

s.close()

exit()
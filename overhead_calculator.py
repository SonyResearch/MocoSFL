# %%

from pyexpat import model
import torch
import torch.nn as nn
import numpy as np

from models import resnet
from models import vgg
import subprocess
import sys

try:
    from fvcore.nn import FlopCountAnalysis
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", 'fvcore'])
finally:
    from fvcore.nn import FlopCountAnalysis
    from fvcore.nn import flop_count_table

#archs
arch="ResNet18"

#Dataset related
input_channel_size = 3
data_size=32
num_class=10

#Scheme related # Change Here
scheme="FL" #SFL/FL
TAResSFL_enable = False
cutlayer=2
adds_bottleneck = False
bottleneck_option="C4S2"

#Moco-related inputs
if scheme=="SFL":
    moco_version="V2"
    K_dim=1024
    sync_frequency = 1
    num_epoch_per_client=200
    num_data_per_client=50
    batch_size=1
else:
    moco_version="largeV2"
    K_dim=2048
    sync_frequency = 5 # how many client epoch per sync
    num_epoch_per_client=500
    num_data_per_client=10000
    batch_size=128

# server-side computations
num_clients = 1000
client_sampling_ratio = 0.1

if scheme=="FL":
    adds_bottleneck = False
if "ResNet" in arch or "resnet" in arch:
    if "resnet" in arch:
        arch = "ResNet" + arch.split("resnet")[-1]
    create_arch = getattr(resnet, arch)
elif "vgg" in arch:
    create_arch =  getattr(vgg, arch)

#get model - use a larger classifier, as in Zhuang et al. Divergence-aware paper
global_model = create_arch(cutting_layer=cutlayer, num_client = 1, num_class=K_dim, group_norm=True, input_size= data_size,
                             adds_bottleneck=adds_bottleneck, bottleneck_option=bottleneck_option)

if moco_version == "largeV2": # This one uses a larger classifier, same as in Zhuang et al. Divergence-aware paper
    classifier_list = [nn.Linear(512 * global_model.expansion, 4096),
                    nn.BatchNorm1d(4096),
                    nn.ReLU(True),
                    nn.Linear(4096, K_dim)]
    global_model.classifier = nn.Sequential(*classifier_list)
elif "V2" in moco_version:
    classifier_list = [nn.Linear(512 * global_model.expansion, K_dim * global_model.expansion),
                    nn.ReLU(True),
                    nn.Linear(K_dim * global_model.expansion, K_dim)]
    global_model.classifier = nn.Sequential(*classifier_list)

global_model.merge_classifier_cloud()

if scheme=="FL":
    if global_model.get_num_of_cloud_layer() != 0:
        global_model.resplit(0)
if scheme == "SFL":
    latent_vector_total_size=np.prod(global_model.get_smashed_data_size(1, data_size))

weight_param_size = 0

for key in global_model.local_list[0].state_dict().keys():
    weight_param_size += np.prod(global_model.local_list[0].state_dict()[key].size())

communication_overhead_weight = num_epoch_per_client//sync_frequency * weight_param_size
if scheme == "FL":
    communication_overhead_weight_latent_vector = 0
elif scheme == "SFL":
    communication_overhead_weight_latent_vector = 2 * num_epoch_per_client * num_data_per_client * latent_vector_total_size

if scheme == "SFL" and TAResSFL_enable:
    communication_overhead_weight = 0.0
    communication_overhead_weight_latent_vector = communication_overhead_weight_latent_vector/2
    
communication_overhead = communication_overhead_weight +  communication_overhead_weight_latent_vector



print("===============================")
print(f"Model weight communication overhead: {communication_overhead_weight*4/1024/1024:.2f} MB")
print(f"Latent vector communication overhead: {communication_overhead_weight_latent_vector*4/1024/1024:.2f} MB")
print(f"Total communication overhead: {communication_overhead*4/1024/1024:.2f} MB")
print("===============================")

#get_memory_usage
global_model.local_list[0].cuda()
noise_input = torch.ones([batch_size, input_channel_size, data_size, data_size])
noise_label = torch.ones(global_model.get_smashed_data_size(batch_size, data_size))
criterion = nn.MSELoss()
noise_input = noise_input.cuda()
noise_label = noise_label.cuda()
params = list(global_model.local_list[0].parameters())
optimizer = torch.optim.SGD(params, lr=0.02, momentum=0.9, weight_decay=5e-4)

if scheme == "SFL" and TAResSFL_enable:
    with torch.no_grad():
        output = global_model.local_list[0](noise_input)
        print("Total CUDA Memory Allocated for inference: %.2f MB"%(torch.cuda.memory_allocated(0)/1024/1024))

#GPU warmup
for i in range(5):
    optimizer.zero_grad()
    output = global_model.local_list[0](noise_input)
    f_loss = criterion(output, noise_label)
    if i == 4:
        print("Total CUDA Memory Allocated for training: %.2f MB"%(torch.cuda.memory_allocated(0)/1024/1024))
    f_loss.backward()
    
    optimizer.step()



noise_input = torch.ones([1, input_channel_size, data_size, data_size])
noise_input = noise_input.cuda()
print("===============================")
flops = FlopCountAnalysis(global_model.local_list[0], noise_input)

if scheme == "SFL":
    noise_input = torch.ones(global_model.get_smashed_data_size(1, data_size))
    noise_input = noise_input.cuda()
    print("===============================")
    server_flops = 2 * FlopCountAnalysis(global_model.cloud, noise_input).total() + weight_param_size
else:
    server_flops = weight_param_size

if scheme == "SFL" and TAResSFL_enable: # if TAResSFL_enable, then no training, no momentum, if not, then + backward + momemtum forward.
    print(f"FLOPs/image: {flops.total()/1024/1024:.2f} M")
    print(f"Total FLOPs: {num_epoch_per_client * num_data_per_client * flops.total()/1024/1024/1024:.2f} G")
else:
    print(f"FLOPs/image: {3* flops.total()/1024/1024:.2f} M")
    print(f"Total FLOPs: {3*num_epoch_per_client * num_data_per_client * flops.total()/1024/1024/1024:.2f} G")
print(f"Total server-side FLOPs: {num_clients * num_epoch_per_client // sync_frequency * server_flops/1024/1024/1024:.2f} G")
# print(f"Total FLOPs (by operator): {flops.by_operator()} M")
# print(f"Total FLOPs (by module): {flops.by_module()} M")
# print(flop_count_table(flops))
print("===============================")
#!/bin/bash
cd "$(dirname "$0")"

#fixed arguments
# num_epoch=200
num_epoch=200
lr=0.003
c_lr=0.00
moco_version=V2
arch=ResNet18
seed=1234
aux_data=cifar100
non_iid_list="0.2"
cutlayer_list="2"
num_client=1000
dataset=cifar10
loss_threshold=0.0
ressfl_alpha=0.0
bottleneck_option_list="C4S2"
data_proportion_list="0.01"
ressfl_target_ssim_list="0.6"
for cutlayer in $cutlayer_list; do
        for ressfl_target_ssim in $ressfl_target_ssim_list; do
                for bottleneck_option in $bottleneck_option_list; do
                        for data_proportion in $data_proportion_list; do
                                for noniid_ratio in $non_iid_list; do
                                        initialze_path="../expert_target_aware/mocosfl${moco_version}_${arch}_${dataset}_cut${cutlayer}_bnl${bottleneck_option}_expert_batchsize128_ressfl_2.0_SSIM${ressfl_target_ssim}_data_proportion_${data_proportion}_aux_${aux_data}"
                                        output_dir="./outputs/mocosfl${moco_version}_${arch}_${dataset}_cut${cutlayer}_bnl${bottleneck_option}_client${num_client}_nonIID${noniid_ratio}_initialize_CLR${c_lr}_SSIM${ressfl_target_ssim}_data_proportion_${data_proportion}"
                                        CUDA_VISIBLE_DEVICES=0 python demo_server_web.py --num_client ${num_client} --lr ${lr} --c_lr ${c_lr} --cutlayer ${cutlayer} --num_epoch ${num_epoch}\
                                                --noniid_ratio ${noniid_ratio} --output_dir ${output_dir}\
                                                --moco_version ${moco_version} --arch ${arch} --dataset ${dataset} --loss_threshold ${loss_threshold}\
                                                --ressfl_alpha ${ressfl_alpha} --bottleneck_option ${bottleneck_option} --auto_adjust --initialze_path ${initialze_path} --seed ${seed}
                                done
                        done
                done
        done
done
## for test, add --resume --attack
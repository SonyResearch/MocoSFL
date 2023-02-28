#!/bin/bash
cd "$(dirname "$0")"
cd ../../
arch=ResNet18
dataset=cifar10
aux_data=cifar100
num_client=1
num_epoch=200
batch_size=128
lr=0.06
num_workers=4
moco_version=V2
cutlayer_list="1 2"
bottleneck_option_list="C4S2"
ressfl_alpha_list="2.0"
ressfl_target_ssim_list="0.6"
data_proportion_list="0.00 0.005 0.01"
for data_proportion in $data_proportion_list; do
        for ressfl_target_ssim in $ressfl_target_ssim_list; do
                for ressfl_alpha in $ressfl_alpha_list; do
                        for cutlayer in $cutlayer_list; do
                                for bottleneck_option in $bottleneck_option_list; do
                                        output_dir="./expert_target_aware/mocosfl${moco_version}_${arch}_${dataset}_cut${cutlayer}_bnl${bottleneck_option}_expert_batchsize${batch_size}_ressfl_${ressfl_alpha}_SSIM${ressfl_target_ssim}_data_proportion_${data_proportion}_aux_${aux_data}"
                                        CUDA_VISIBLE_DEVICES=0 python run_sflmoco_taressfl.py --num_client ${num_client} --lr ${lr} --cutlayer ${cutlayer} --num_epoch ${num_epoch}\
                                                --output_dir ${output_dir} --batch_size ${batch_size} --num_workers ${num_workers}\
                                                --moco_version ${moco_version} --bottleneck_option ${bottleneck_option} --ressfl_alpha ${ressfl_alpha}\
                                                --ressfl_target_ssim ${ressfl_target_ssim} --arch ${arch} --dataset ${dataset} --data_proportion ${data_proportion}\
                                                --aux_data ${aux_data}
                                done
                        done
                done
        done
done

## for test, add --resume
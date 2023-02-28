#!/bin/bash
cd "$(dirname "$0")"
cd ../../

#fixed arguments
num_epoch=200
lr=0.06
moco_version=V2
arch=ResNet18

non_iid_list="0.2 1.0"
cutlayer_list="1 2"
num_client_list="5"
dataset=cifar10
loss_threshold=0.0
ressfl_alpha=0.0
bottleneck_option=None
for num_client in $num_client_list; do
        for noniid_ratio in $non_iid_list; do
                for cutlayer in $cutlayer_list; do
                        output_dir="./outputs/mocosfl${moco_version}_${arch}_${dataset}_cut${cutlayer}_bnl${bottleneck_option}_client${num_client}_nonIID${noniid_ratio}"
                        python run_sflmoco.py --num_client ${num_client} --lr ${lr} --cutlayer ${cutlayer} --num_epoch ${num_epoch}\
                                --noniid_ratio ${noniid_ratio} --output_dir ${output_dir}\
                                --moco_version ${moco_version} --arch ${arch} --dataset ${dataset} --loss_threshold ${loss_threshold}\
                                --ressfl_alpha ${ressfl_alpha} --bottleneck_option ${bottleneck_option} --auto_adjust
                done
        done
done

non_iid_list="0.2 1.0"
cutlayer_list="1 2"
num_client_list="20"
dataset=cifar10
loss_threshold=0.0
ressfl_alpha=0.0
bottleneck_option=None
batch_size=20
client_sample_ratio=0.25
avg_freq=10
for num_client in $num_client_list; do
        for noniid_ratio in $non_iid_list; do
                for cutlayer in $cutlayer_list; do
                        output_dir="./outputs/mocosfl${moco_version}_${arch}_${dataset}_cut${cutlayer}_bnl${bottleneck_option}_client${num_client}_nonIID${noniid_ratio}"
                        python run_sflmoco.py --num_client ${num_client} --lr ${lr} --cutlayer ${cutlayer} --num_epoch ${num_epoch}\
                                --noniid_ratio ${noniid_ratio} --output_dir ${output_dir}\
                                --moco_version ${moco_version} --arch ${arch} --dataset ${dataset} --loss_threshold ${loss_threshold}\
                                --ressfl_alpha ${ressfl_alpha} --bottleneck_option ${bottleneck_option} --batch_size ${batch_size}\
                                --client_sample_ratio ${client_sample_ratio} --avg_freq ${avg_freq}
                done
        done
done
## for test, add --resume --attack
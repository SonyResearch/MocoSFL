#!/bin/bash
cd "$(dirname "$0")"
cd ../../

#fixed arguments
num_epoch=200
moco_version=V2
arch=MobileNetV2
seed_list="1234"

#### RERUN cut-1 with 0.03 LR

non_iid_list="1.0 0.2"
cutlayer_list="2"
lr=0.03
num_client_list="100"
dataset=cifar10
loss_threshold=0.0
ressfl_alpha=0.0
bottleneck_option=None
for seed in $seed_list; do
        for num_client in $num_client_list; do
                for noniid_ratio in $non_iid_list; do
                        for cutlayer in $cutlayer_list; do
                                output_dir="./outputs/other_archs/mocosfl${moco_version}_${arch}_${dataset}_cut${cutlayer}_bnl${bottleneck_option}_client${num_client}_nonIID${noniid_ratio}_seed_${seed}_lr_${lr}"
                                python run_sflmoco.py --num_client ${num_client} --lr ${lr} --cutlayer ${cutlayer} --num_epoch ${num_epoch}\
                                        --noniid_ratio ${noniid_ratio} --output_dir ${output_dir}\
                                        --moco_version ${moco_version} --arch ${arch} --dataset ${dataset} --loss_threshold ${loss_threshold}\
                                        --ressfl_alpha ${ressfl_alpha} --bottleneck_option ${bottleneck_option} --auto_adjust --seed ${seed}
                        done
                done
        done
done

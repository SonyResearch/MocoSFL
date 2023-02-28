#!/bin/bash
cd "$(dirname "$0")"
cd ../../

#fixed arguments
num_epoch=200
lr=0.06
moco_version=V2
arch=ResNet18

seed_list="1234 1235 1236"
non_iid_list="0.2 1.0"
cutlayer_list="1 2"
num_client_list="100 200 1000"
dataset_list="cifar10 cifar100 imagenet12"
loss_threshold=0.0
ressfl_alpha=0.0
bottleneck_option=None

python prepare_imagenet12.py

for dataset in $dataset_list; do
        for seed in $seed_list; do
                for num_client in $num_client_list; do
                        for noniid_ratio in $non_iid_list; do
                                for cutlayer in $cutlayer_list; do
                                        output_dir="./outputs/imagenet12_cut2/mocosfl${moco_version}_${arch}_${dataset}_cut${cutlayer}_bnl${bottleneck_option}_client${num_client}_nonIID${noniid_ratio}_seed_${seed}"
                                        python run_sflmoco.py --num_client ${num_client} --lr ${lr} --cutlayer ${cutlayer} --num_epoch ${num_epoch}\
                                                --noniid_ratio ${noniid_ratio} --output_dir ${output_dir}\
                                                --moco_version ${moco_version} --arch ${arch} --dataset ${dataset} --loss_threshold ${loss_threshold}\
                                                --ressfl_alpha ${ressfl_alpha} --bottleneck_option ${bottleneck_option} --auto_adjust --seed ${seed}
                                done
                        done
                done
        done
done
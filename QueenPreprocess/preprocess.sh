#!/usr/bin/bash
export PYTHONPATH=.

# Parameters
dir_exp=exp/20231011
pro_n_ep=100
n_classes=100
pro_lr=0.1
gamma=0.2
pro_arch=vgg16_cifar100
trainset=cifar100
testset=cifar100
pro_bs=100
seed=27
pro_stepsize=20
map_n_ep=100
size=32
in_dim=512
out_dim=2
n_layers=8
step_down=2
beta1=0.9
beta2=0.99
map_bs=5000
map_lr=0.01
map_stepsize=10
map_gamma=0.5
shadow_n_ep=1
shadow_arch=vgg11
shadow_lr=0.1
n_shadows=10
n_samples=1000
shadow_bs=100

echo "Copy for reproduction"
mkdir -p $dir_exp/repr_scripts
cp QueenPreprocess/preprocess.sh $dir_exp/repr_scripts

# echo "Train the protectee network"
# python QueenPreprocess/train_protectee_net.py \
#     --n_ep $pro_n_ep \
#     --n_classes $n_classes \
#     --lr $pro_lr \
#     --model_arch $pro_arch \
#     --optimizer sgd \
#     --scheduler steplr \
#     --criteria crossentropy \
#     --dir_train ../../data \
#     --dir_test ../../data \
#     --dir_exp $dir_exp \
#     --trainset $trainset \
#     --testset $testset \
#     --bs $pro_bs \
#     --seed $seed \
#     --step_size $pro_stepsize \
#     --gamma $gamma \
#     --momentum 0.9 \
#     --weight_decay 0.01 \
#     --drop_last \
#     --img_size $size \

# echo "Extract the training features"
# python QueenPreprocess/extract_training_features.py \
#     --model_arch $pro_arch \
#     --n_classes $n_classes \
#     --pth_ckpt $dir_exp/cls_ckpt/${pro_arch}_${trainset}.pt \
#     --dir_train ../../data \
#     --dir_exp $dir_exp \
#     --trainset $trainset \
#     --bs 512 \
#     --seed 27 \
#     --img_size $size \

# echo "Train the mapping network"
# python QueenPreprocess/train_mapping_network.py \
#     --model_arch $pro_arch \
#     --pth_feats $dir_exp/training_features/feats_${pro_arch}_${trainset}.pt \
#     --dir_exp $dir_exp \
#     --dataset_train $trainset \
#     --n_ep $map_n_ep \
#     --bs $map_bs \
#     --seed $seed \
#     --img_size $size \
#     --in_dim $in_dim \
#     --out_dim $out_dim \
#     --num_layers $n_layers \
#     --step_down $step_down \
#     --lr $map_lr \
#     --beta1 $beta1 \
#     --beta2 $beta2 \
#     --shuffle \
#     --step_size $map_stepsize \
#     --gamma $map_gamma \

echo Sensitivity Analysis
python QueenPreprocess/sensitivity_analysis.py \
    --model_arch $pro_arch \
    --n_classes $n_classes \
    --pth_ckpt $dir_exp/map_net_ckpt/map_net_${pro_arch}_${trainset}.pt \
    --pth_feats $dir_exp/training_features/feats_${pro_arch}_${trainset}.pt \
    --dir_exp $dir_exp \
    --dataset_train $trainset \
    --bs $map_bs \
    --seed $seed \
    --img_size $size \
    --in_dim $in_dim \
    --out_dim $out_dim \
    --num_layers $n_layers \
    --step_down $step_down \

# echo Train Shadow Models
# python QueenPreprocess/train_shadow_models.py \
#     --n_ep $shadow_n_ep \
#     --n_classes $n_classes \
#     --lr $shadow_lr \
#     --model_arch $shadow_arch \
#     --optimizer sgd \
#     --scheduler steplr \
#     --criteria crossentropy \
#     --dir_train ../../data \
#     --dir_test ../../data \
#     --dir_exp $dir_exp \
#     --dataset_train $trainset \
#     --dataset_test $testset \
#     --bs $shadow_bs \
#     --seed $seed \
#     --step_size $step_size \
#     --gamma 0.02 \
#     --momentum 0.9 \
#     --weight_decay 0.01 \
#     --img_size $size \
#     --n_shadow $n_shadows \
#     --n_samples $n_samples \
#     --drop_last \
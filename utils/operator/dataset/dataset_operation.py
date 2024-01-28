'''
Desc: Dataset Operation for Neural Network
Last upate: 20/12/2023
'''
import sys
sys.path.append('.')
import torch
from os import path as P
from utils.operator.os import get_subdirs
from typing import *
import random
from torch.utils.data import Dataset, Subset



def split_distilled_dataset_batch(pth_distilled_dir:str, ratio:float) -> list:
    '''
    @Desc:
        Split the distilled dataset into training and test datasets in batch.
    @Args:
        pth_distilled_dir: The path to the directory containing all the subdirectories of the distilled datasets;
        ratio: The ratio of the training samples to the entire distilled set;
    @Out:
        [split_set_1, ..., split_set_n]
    '''
    lst_subdirs = get_subdirs(pth_distilled_dir)
    print(lst_subdirs)
    lst_split_distilled_datasets = [split_distilled_dataset(dir, ratio) for dir in lst_subdirs]
    return lst_split_distilled_datasets


def split_distilled_dataset(pth_distilled_dir:str, ratio:float) -> dict:
    '''
    @Desc:
        Split the distilled dataset into training and test datasets.
        The balance of classes is not considered, since it is for the training of the network arch. classifier.
        
    @Args:
        pth_distilled_dir: The path to the directory containing the distilled samples, labels and learning rate;
        ratio: The ratio of the training samples to the entire distilled set;
        
    @Out:
        dict{
            'train_samples': Tensor,
            'train_labels': Tensor,
            'test_samples': Tensor,
            'test_labels': Tensor
            'lr': float
        }
    '''
    pth_samples = P.join(pth_distilled_dir, 'images_best.pt')
    pth_labels = P.join(pth_distilled_dir, 'labels_best.pt')
    pth_lr = P.join(pth_distilled_dir, 'lr_best.pt')
    
    samples = torch.load(pth_samples)  # Tensor: n c w h
    labels = torch.load(pth_labels)  # Tensor: n d
    lr = torch.load(pth_lr)  # Tensor float
    num_samples = samples.shape[0]
    
    '''Create indices for training and test subset'''
    idx = torch.randperm(num_samples)
    num_training_samples = int(num_samples * ratio)
    idx_train = idx[:num_training_samples]
    idx_test = idx[num_training_samples:]
    
    '''Split the dataset'''
    train_samples = samples[idx_train]
    train_labels = labels[idx_train]
    test_samples = samples[idx_test]
    test_labels = labels[idx_test]
    
    split_distilled_dataset = {
        'train_samples': train_samples,
        'train_labels': train_labels,
        'test_samples': test_samples,
        'test_labels': test_labels,
        'lr': lr
    }
    
    return split_distilled_dataset


def create_train_test_indices(num_samples:int, ratio_train:float) -> Tuple[List[int]]:
    '''
    @Desc: To create non-overlapping indices to split a dataset into a training and a test set.
    '''
    train_idx, test_idx = None, None
    total_idx = list(range(num_samples))
    num_train_samples = int(num_samples * ratio_train)
    train_idx = random.sample(total_idx, num_train_samples)
    test_idx = list(set(total_idx) - set(train_idx))
    return train_idx, test_idx


def get_subsets(D:Dataset, n_subsets:int, n_samples_per_subset:int):
    '''To create subsets from the dataset regardless of classes.'''
    n_samples_total = n_subsets * n_samples_per_subset
    n_samples = len(D)
    assert n_samples_total < n_samples, 'Subset size must be smaller than the dataset size'
    idx = random.sample(list(range(n_samples)), n_samples_total)
    subsets = ()
    for i in range(n_subsets):
        sp = i*n_samples_per_subset
        ep = (i+1)*n_samples_per_subset
        cur_idx = idx[sp:ep]
        subsets = subsets + (Subset(D, cur_idx), )
    return subsets


'''Debug'''
if __name__ == '__main__':
    # split_distilled_dataset('DATM/distill/logged_files/CIFAR10/1000/ConvNet/dandy-water-36/Normal', 0.8)
    # print(len(split_distilled_dataset_batch('DATM/distill/logged_files/CIFAR10/1000/ConvNet/dandy-water-36/', 0.8)))
    a, b = create_train_test_indices(10000, 0.75)
    print(len(a), len(b))
    for v in a:
        if v in b: print('Overlapped!')
    
            
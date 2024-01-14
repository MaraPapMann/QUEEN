'''
@Desc: Use the trained classifier network to extract the training features.
'''
import torch
import argparse
import random
import MHC.GeneralOperation.pylib as py
from torch import Tensor
import MHC.utils as ut
import DataFetcher as Fetcher
import NetworkTrainer as Trainer
from torch.utils.data import DataLoader
import time


def get_feat_centers(feats:Tensor, labels:Tensor, n_classes:int):
    centers = {}
    for y in range(n_classes):
        cur_feats = feats[labels == y]
        cur_center = torch.mean(cur_feats, 0).unsqueeze(0)
        centers.update({y:cur_center})
    return centers


def main():
    '''
    Parse Arguments
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_arch', nargs='?', type=str, default='res34')
    parser.add_argument('--n_classes', nargs='?', type=int, default=10)
    parser.add_argument('--pth_ckpt', nargs='?', type=str, default='exp/20230904/cls_ckpt/res34_mnist.pt')
    parser.add_argument('--dir_train', nargs='?', type=str, default='../../data')
    parser.add_argument('--dir_exp', nargs='?', type=str, default='exp/20230904')
    parser.add_argument('--trainset', nargs='?', type=str, default='mnist')
    parser.add_argument('--bs', nargs='?', type=int, default=512)
    parser.add_argument('--seed', nargs='?', type=int, default=27)
    parser.add_argument('--img_size', nargs='?', type=int, default=32)
    parser.add_argument('--drop_last', action='store_true')
    args = parser.parse_args()
    

    '''
    Initialize
    '''
    dir_save = py.join(args.dir_exp, 'training_features')
    pth_feats = py.join(dir_save, 'feats_%s_%s.pt'%(args.model_arch, args.trainset))
    pth_centers = py.join(dir_save, 'centers_%s_%s.pt'%(args.model_arch, args.trainset))
    py.mkdir(args.dir_exp)
    py.mkdir(dir_save)
    
    
    '''
    Set random seed
    '''
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    
    
    '''
    Set classifier network
    '''
    classifier = Trainer.get_classifier(args.model_arch, args.n_classes)
    Trainer.load_ckpt(classifier, args.pth_ckpt)
    
    
    '''
    Set dataset
    '''
    D_train = Fetcher.load_dataset(args.trainset, args.dir_train, True, True, args.img_size)
    trainloader = DataLoader(D_train, args.bs, True)
    
    
    '''
    Extract features
    '''
    feats, labels = Trainer.get_feats_labels(classifier, trainloader)
    feat_centers = get_feat_centers(feats, labels, args.n_classes)
    feats = feats.detach().cpu()
    labels = labels.detach().cpu()
    torch.save((feats, labels), pth_feats)
    torch.save(feat_centers, pth_centers)
    
    print(feats, labels)

if __name__ == '__main__':
    start = time.time()
    main()
    end = time.time()
    print('Time  consumption of extracting the training features: %.4f seconds'%(end - start))
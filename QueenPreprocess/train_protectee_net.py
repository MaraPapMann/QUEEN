import torch
import argparse
import random
import MHC.GeneralOperation.pylib as py
from torchvision.utils import save_image
import torchvision 
import NetworkTrainer as Trainer
import DataFetcher as Fetcher
from torch.utils.data import DataLoader


def main():
    '''
    Parse Arguments
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_ep', nargs='?', type=int, default=2)
    parser.add_argument('--n_classes', nargs='?', type=int, default=10)
    parser.add_argument('--lr', nargs='?', type=float, default=0.01)
    parser.add_argument('--model_arch', nargs='?', type=str, default='res34')
    parser.add_argument('--optimizer', nargs='?', type=str, default='sgd')
    parser.add_argument('--scheduler', nargs='?', type=str, default='steplr')
    parser.add_argument('--criteria', nargs='?', type=str, default='crossentropy')
    parser.add_argument('--dir_train', nargs='?', type=str, default='../../data')
    parser.add_argument('--dir_test', nargs='?', type=str, default='../../data')
    parser.add_argument('--dir_exp', nargs='?', type=str, default='exp/20230914')
    parser.add_argument('--trainset', nargs='?', type=str, default='mnist')
    parser.add_argument('--testset', nargs='?', type=str, default='mnist')
    parser.add_argument('--bs', nargs='?', type=int, default=512)
    parser.add_argument('--seed', nargs='?', type=int, default=27)
    parser.add_argument('--step_size', nargs='?', type=int, default=10)
    parser.add_argument('--gamma', nargs='?', type=float, default=0.02)
    parser.add_argument('--momentum', nargs='?', type=float, default=0.9)
    parser.add_argument('--weight_decay', nargs='?', type=float, default=0.01)
    parser.add_argument('--drop_last', action='store_true')
    parser.add_argument('--img_size', nargs='?', type=int, default=32)
    args = parser.parse_args()
    print(args)
    

    '''
    Initialize
    '''
    dir_ckpt = py.join(args.dir_exp, 'cls_ckpt')
    pth_save = py.join(dir_ckpt, '%s_%s.pt'%(args.model_arch, args.trainset))
    py.mkdir(args.dir_exp)
    py.mkdir(dir_ckpt)
    
    
    '''
    Set random seed
    '''
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    
    
    '''
    Set classifier network
    '''
    classifier = Trainer.get_classifier(args.model_arch, args.n_classes)
    optimizer = Trainer.get_optimizer(classifier, args.optimizer, args.lr, momentum=args.momentum, 
                                      weight_decay=args.weight_decay, nesterov=True)
    scheduler = Trainer.get_scheduler(optimizer, args.scheduler, step_size=args.step_size, gamma=args.gamma)
    criteria = Trainer.get_criteria(args.criteria)
    
    
    '''
    Set dataset
    '''
    if args.custom_dataset:
        D_train = Fetcher.load_tensor_dataset(args.pth_tensor_dataset)
    else:
        D_train = Fetcher.load_dataset(args.trainset, args.dir_train, True, True, args.img_size)
    trainloader = DataLoader(D_train, args.bs, True, drop_last=args.drop_last)

    D_test = Fetcher.load_dataset(args.testset, args.dir_test, False, True, args.img_size)
    testloader = DataLoader(D_test, args.bs, False)
    
    
    '''
    Train classifier network
    '''
    Trainer.train_classifier(classifier, optimizer, scheduler, criteria, args.n_ep, trainloader, testloader, pth_save)


if __name__ == '__main__':
    main()
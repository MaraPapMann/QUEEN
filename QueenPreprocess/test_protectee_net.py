import torch
import argparse
import random
import MHC.GeneralOperation.pylib as py
from torchvision.utils import save_image
import torchvision 
import NetworkTrainer as Trainer
import DataFetcher as Fetcher
from torch.utils.data import DataLoader
from torchvision.utils import save_image


def main():
    '''
    Parse Arguments
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_classes', nargs='?', type=int, default=10)
    parser.add_argument('--model_arch', nargs='?', type=str, default='res34')
    parser.add_argument('--dir_test', nargs='?', type=str, default='../../data')
    parser.add_argument('--pth_ckpt', nargs='?', type=str, default='exp/20231127_fashionmnist/cls_ckpt/res34_fashionmnist.pt')
    parser.add_argument('--testset', nargs='?', type=str, default='emnist')
    parser.add_argument('--bs', nargs='?', type=int, default=100)
    parser.add_argument('--seed', nargs='?', type=int, default=27)
    parser.add_argument('--img_size', nargs='?', type=int, default=32)
    args = parser.parse_args()
    print(args)
    
    
    '''
    Set classifier network
    '''
    classifier = Trainer.get_classifier(args.model_arch, args.n_classes)
    classifier.load_state_dict(torch.load(args.pth_ckpt))
    criteria = torch.nn.CrossEntropyLoss()
    
    
    '''
    Set dataset
    '''
    D_test = Fetcher.load_dataset(args.testset, args.dir_test, False, True, args.img_size)
    testloader = DataLoader(D_test, args.bs, False)
    
    # for x, y in testloader:
    #     save_image(x, 'checkthisimage.png')
    #     print(y)
    #     break

    
    
    '''
    Train classifier network
    '''
    test_acc, test_loss = Trainer.test_classifier(classifier, testloader, criteria)
    print('Test acc: %.4f, Test loss: %.4f'%(test_acc, test_loss))


if __name__ == '__main__':
    main()
import sys
sys.path.append('../..')
import torch
from torchvision import transforms, datasets
from torch.utils.data import Subset, Dataset, DataLoader, TensorDataset
import random
from defenses import datasets as DS


class DataFetcher():
    def __init__(self) -> None:
        '''
        @Desc: A fetcher that is in charge of loading and processing the mainstream datasets.
        '''
        return
    
    @staticmethod
    def load_dataset(dataset:str, root:str, train:bool, download:bool, resize:int, return_loader:bool=False, **kwargs):
        # Set dataset
        print(dataset)
        assert dataset in ['mnist', 'emnistdigit', 'emnistletter', 
                        'cifar10', 'CIFAR-10', 'CIFAR10',
                        'cifar100', 'CIFAR-100', 'CIFAR100',
                        'cifar10ext', 'cifar100ext', 'fashionmnist', 'gtsrb', 'imagenette', 'Caltech256'], 'Enter a valid dataset.'
        D = None

        # MNIST
        if dataset == 'mnist':
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,)),
                transforms.Lambda(lambda x: x.repeat(3, 1, 1)), # Grayscale to RGB
                transforms.Resize((resize, resize)),
            ])
            D = datasets.MNIST(root, train, transform, download=download)
        
        # EMNIST-Digits
        if dataset == 'emnistdigit':
            transform = transforms.Compose([
                lambda img: transforms.functional.rotate(img, -90),
                lambda img: transforms.functional.hflip(img),
                transforms.ToTensor(),
                transforms.Normalize((0.1733,), (0.3317,)),
                transforms.Lambda(lambda x: x.repeat(3, 1, 1)), # Grayscale to RGB
                transforms.Resize((resize, resize)),
            ])
            D = datasets.EMNIST(root, split='digits', train=train, transform=transform, download=download)
        
        # EMNIST-Letters
        if dataset == 'emnistletter':
            transform = transforms.Compose([
                lambda img: transforms.functional.rotate(img, -90),
                lambda img: transforms.functional.hflip(img),
                transforms.ToTensor(),
                transforms.Normalize((0.1733,), (0.3317,)),
                transforms.Lambda(lambda x: x.repeat(3, 1, 1)), # Grayscale to RGB
                transforms.Resize((resize, resize)),
            ])
            D = datasets.EMNIST(root, split='letters', train=train, transform=transform, download=download)
        
        # CIFAR-10
        if dataset in ['cifar10', 'CIFAR-10', 'CIFAR10']:
            if train:
                transform = transforms.Compose([
                    transforms.Resize((resize, resize)),
                    transforms.ToTensor(),
                    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                ])
                D = datasets.CIFAR10(root, train, transform, download=download)
            else:
                transform = transforms.Compose([
                    transforms.Resize((resize, resize)),
                    transforms.ToTensor(),
                    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
                ])
                D = datasets.CIFAR10(root, train, transform, download=download)
        
        # CIFAR-10E
        if dataset == 'cifar10ext':
            transform = transforms.Compose([
                transforms.Resize((resize, resize)),
                transforms.ToTensor(),
                transforms.Normalize((0.4715, 0.4701, 0.4249), (0.2409, 0.2352, 0.2593)),
            ])
            D = datasets.ImageFolder(root+'/CIFAR10_EXT', transform)
        
        # CIFAR-100
        if dataset in ['cifar100', 'CIFAR-100', 'CIFAR100']:
            if train:
                transform = transforms.Compose([
                    transforms.Resize((resize, resize)),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5071, 0.4866, 0.4409), (0.2673, 0.2564, 0.2762)),
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                ])
                D = datasets.CIFAR100(root, train, transform, download=download)
            else:
                transform = transforms.Compose([
                    transforms.Resize((resize, resize)),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5071, 0.4866, 0.4409), (0.2673, 0.2564, 0.2762)),
                ])
                D = datasets.CIFAR100(root, train, transform, download=download)
        
        # CIFAR-100E
        if dataset == 'cifar100ext':
            transform = transforms.Compose([
                transforms.Resize((resize, resize)),
                transforms.ToTensor(),
                transforms.Normalize((0.4672, 0.4581, 0.4093), (0.2580, 0.2470, 0.2671)),
            ])
            D = datasets.ImageFolder(root+'/CIFAR100_EXT', transform)
        
        # FashionMNIST
        if dataset == 'fashionmnist':
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.2860,), (0.3530,)),
                transforms.Lambda(lambda x: x.repeat(3, 1, 1)), # Grayscale to RGB
                transforms.Resize((resize, resize)),
            ])
            D = datasets.FashionMNIST(root, train, transform, download=download)
        
        # GTSRB
        if dataset == 'gtsrb':
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.3805, 0.3484, 0.3574), (0.3031, 0.2950, 0.3007)),
                transforms.Resize((resize, resize)),
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
            ])
            if train:
                D = datasets.GTSRB(root, 'train', transform, download=download)
            else:
                D = datasets.GTSRB(root, 'test', transform, download=download)
        
        # ImageNette
        if dataset == 'imagenette':
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4672, 0.4581, 0.4093), (0.2580, 0.2470, 0.2671)),
                transforms.Resize((resize, resize)),
                transforms.RandomCrop(128, padding=16),
                transforms.RandomHorizontalFlip(),
            ])
            if train:
                D = datasets.ImageFolder(root+'/imagenette'+'/train', transform)
            else:
                D = datasets.ImageFolder(root+'/imagenette'+'/val', transform)
        
        # Caltech256
        if dataset == 'Caltech256':
            if train:
                transform = transforms.Compose([
                    transforms.RandomResizedCrop(224),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225]),
                    ]),
                D = DS.Caltech256(True, transform, target_transform=None, download=True)
            else:
                transform = transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225]),
                ])
                D = DS.Caltech256(False, transform, target_transform=None, download=True)
        print(D[0])
        if return_loader:
            loader = DataLoader(D, kwargs['bs'], kwargs['shuffle'], pin_memory=True, drop_last=kwargs['drop_last'])
            return loader
        else:
            return D

    @staticmethod
    def get_subsets(D:Dataset, n_subsets:int, n_samples_per_subset:int):
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

    @staticmethod
    def load_tensor_dataset(pth_tensor_ds:str) -> Dataset:
        D = torch.load(pth_tensor_ds)
        D = TensorDataset(D)
        return D
    
    @staticmethod
    def tensors_to_dataset(*tensors:torch.Tensor) -> Dataset:
        return TensorDataset(*tensors)
    
    @staticmethod
    def data_loader(dataset:Dataset, bs:int, shuffle:bool, num_worker:int=2, pin_memory:bool=True) -> DataLoader:
        return DataLoader(dataset, bs, shuffle, num_workers=num_worker, pin_memory=pin_memory)


# Debug
if __name__ == '__main__':
    print()
    # D = load_dataset('emnistletter', '../../data', True, True, 32)
    # print('Sample size: ', (D[0][0].shape))
    # # subsets = get_subsets(D, 5, 5)
    # # print(len(subsets[0]))
    # print('Length of the dataset: ', len(D))
    # sumel = 0.0
    # countel = 0
    # for img, _ in D:
    #     sumel += img.sum([1, 2])
    #     countel += torch.numel(img[0])
    # mean = sumel/countel
    # print('Dataset Mean: ', mean)
    
    # sumel = 0.0
    # countel = 0
    # for img, _ in D:
    #     img = (img - mean.unsqueeze(1).unsqueeze(1))**2
    #     sumel += img.sum([1, 2])
    #     countel += torch.numel(img[0])
    # std = torch.sqrt(sumel/countel)
    # print('Dataset Std: ', std)
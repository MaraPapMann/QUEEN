"""
@Author: M.H.C
@Desc: A custom dataset loader for Pytorch
"""
import torch
import numpy as np
import torchvision as tv
import torchvision.transforms as T
from torchvision.utils import save_image
import MHC.GeneralOperation.pylib as py
import tqdm
from typing import List, Any, Dict
from torch.utils.data import DataLoader
from pathlib import Path
import random
import shutil
__all__ = ['DatasetLoader']


class DatasetLoader():
    def __init__(self, 
                 dir_data:str,
                 is_custom:bool,
                 dataset:str,
                 is_train:bool,
                 is_download:bool,
                 classes:List[str],
                 transform:List,
                 bs:int,
                 shuffle:bool,
                 num_workers:int,
                 pin_memory:bool,
                 drop_last:bool) -> None:
        # Doc params
        super(DatasetLoader, self).__init__()
        self.dir_data = dir_data
        self.is_custom = is_custom
        self.dataset = dataset
        self.is_train = is_train
        self.transtorm = transform
        self.bs = bs
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.drop_last = drop_last
        
        # Init dataset and loader
        if is_custom:  # Custom dataset
            self.classes = classes
            self.dataset = tv.datasets.ImageFolder(dir_data, transform)
            self.dataloader = DataLoader(self.dataset, bs, shuffle, num_workers=num_workers, pin_memory=pin_memory, drop_last=drop_last)
        else:  # Official released dataset
            assert dataset in ['cifar10', 'mnist', 'imagenet'], 'The dataset must be one of the official released datasets!'
            if dataset == 'cifar10':
                self.classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
                self.dataset = tv.datasets.CIFAR10(dir_data, is_train, transform, download=is_download)
                self.dataloader = DataLoader(self.dataset, bs, shuffle, num_workers=num_workers, pin_memory=pin_memory, drop_last=drop_last)
    
    
    def output_dataset_img(self, dir_out:str) -> None:
        py.mkdir(dir_out)
        for class_ in self.classes:
            py.mkdir(py.join(dir_out, class_))
            
        counter = 1
        for x, y in tqdm.tqdm(self.dataset, desc='Saving Images'):
            class_ = self.classes[y]
            dir_save = py.join(dir_out, class_)
            x.save(py.join(dir_save, '%d.png'%(counter)))
            counter += 1
        
        print('All images in this dataset have been saved in PNG format.')
        return

    
    def split_dataset(self, n_split:int, dir_out:str, ext:str) -> None:
        for i in range(n_split):
            self.mkdir_classes(dir_out, i+1)
        dict_data_pths = self.get_data_pths(ext)
        
        self.split_classes(dict_data_pths, self.shuffle, n_split, dir_out)
        return
    
    
    def get_data_pths(self, ext:str) -> Dict:
        dict_data_pths = {}
        for class_ in self.classes:
            cur_dir = py.join(self.dir_data, class_)
            f_lst = list(Path(cur_dir).glob('*.%s'%ext))
            for i in range(len(f_lst)):
                f_lst[i] = str(f_lst[i])
            dict_data_pths.update({class_:f_lst})
        return dict_data_pths
    
    
    def split_classes(self, data_pths:dict, shuffle:bool, n_split:int, dir_out:str) -> None:
        portion_per_class = len(self.dataset) // len(self.classes) // n_split
        for class_ in tqdm.tqdm(data_pths.keys(), desc='splitting classes'):
            cur_lst_pth = data_pths[class_]
            if shuffle:
                random.shuffle(cur_lst_pth)
            
            counter_class = 0
            for i in tqdm.trange(n_split, desc='reallocating class "%s"'%class_):
                # dir_out/i/key = dir_out/split_index/class_
                dest = py.join(dir_out, py.join(str(i+1), class_))
                for j in range(counter_class, counter_class+portion_per_class):
                    shutil.copy(cur_lst_pth[j], py.join(dest, '%s.png'%(j+1)))
                counter_class += portion_per_class
        return
    
    
    def mkdir_classes(self, dir_out:str, index:int) -> None:
        dest = py.join(dir_out, str(index))
        py.mkdir(dest)
        for class_ in self.classes:
            py.mkdir(py.join(dest, class_))
        return
    
    
    def output_imgs_in_1_dir(self, dir_out:str) -> None:
        py.mkdir(dir_out)
        counter = 1
        for x, y in tqdm.tqdm(self.dataloader, desc='Saving images'):
            for i in range(x.shape[0]):
                save_image(x[i], py.join(dir_out, '%d.png'%counter))
                counter += 1
        print('Output complete!')
        return


    def create_subset(self, dir_out:str, n_data_p_class:int, **kwargs) -> None:
        data_pths = self.get_data_pths(**kwargs)

        for class_ in self.classes:
            py.mkdir(py.join(dir_out, class_))

        for class_ in tqdm.tqdm(data_pths.keys(), desc='splitting classes'):
            cur_lst_pth = data_pths[class_]
            if shuffle:
                random.shuffle(cur_lst_pth)
            
            for i in tqdm.trange(n_data_p_class, desc='Creating subset from class "%s"'%class_):
                # dir_out/i/key = dir_out/split_index/class_
                dest = py.join(dir_out, class_)
                shutil.copy(cur_lst_pth[i], py.join(dest, '%s.png'%(i+1)))
        
        return
        
                
                
if __name__ == '__main__':
    """
    ==============
    === Params ===
    ==============
    """
    # dir_data = 'data/cifar10'
    dir_data = 'data/synth_cifar10/cls_guide/wide_resnet/train'
    # is_custom = False
    is_custom = True
    dataset = 'custom'
    # dataset = 'cifar10'
    is_train = True
    is_download = True
    classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    transform = T.Compose([T.ToTensor()])
    bs = 100
    shuffle = True
    num_workers = 2
    pin_memory = True
    drop_last = False
    
    dir_out = 'data/synth_cifar10/subset/cls_guide/10000'
    n_split = 5
    ext = 'png'
    num_data_per_class = 10000
    
    """
    =================
    === Execution ===
    =================
    """
    data = DatasetLoader(dir_data, is_custom, dataset, is_train, is_download, classes, transform, bs, shuffle, num_workers, pin_memory, drop_last)
    # data_loader.output_dataset_img(dir_out)
    # data.split_dataset(n_split, dir_out, ext)
    # data.output_imgs_in_1_dir('data/cifar10_syn_gw0.0_1dir')
    data.create_subset(dir_out, num_data_per_class, ext=ext)
    
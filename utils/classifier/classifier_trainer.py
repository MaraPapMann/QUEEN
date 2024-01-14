import sys
sys.path.append('../..')
import torch
import torch.nn as nn
from torch import Tensor
from torch.optim import Optimizer
from torch.optim.lr_scheduler import StepLR
import tqdm
from utils.stats.average_meter import AverageMeter
from torch.utils.data import DataLoader
import utils.GeneralOperation.pylib as py
from utils.classifier import zoo
from utils.stats.timer import Timer
from utils.operator.os.os_operation import *
from utils.lossfunc import SoftCrossEntropyLoss


class ClassifierTrainer():
    def __init__(self, model_arch:str, num_classes:int, optimizer:str, scheduler:str, lr:float, criteria:str, dir_out:str, device:str='cuda:0', **kwargs) -> None:
        self.dir_out = dir_out
        self.device = device
        self.classifier = self.get_classifier(model_arch, num_classes, device, **kwargs)
        self.optimizer = self.get_optimizer(self.classifier, optimizer, lr, **kwargs)
        self.scheduler = self.get_scheduler(self.optimizer, scheduler, **kwargs)
        self.criteria = self.get_criteria(criteria, **kwargs)
        return
    
    def reset(self, model_arch:str, num_classes:int, optimizer:str, scheduler:str, lr:float, criteria:str, dir_out:str, device:str='cuda:0', **kwargs) -> None:
        self.classifier = self.get_classifier(model_arch, num_classes, device, **kwargs)
        self.optimizer = self.get_optimizer(self.classifier, optimizer, lr, **kwargs)
        self.scheduler = self.get_scheduler(self.optimizer, scheduler, **kwargs)
        self.criteria = self.get_criteria(criteria, **kwargs)
        return
    
    @staticmethod
    def get_classifier(model_arch:str, num_classes:int, device:str='cuda:0', **kwargs) -> nn.Module:
        '''
        @Desc: To get an untrained classifier network.
        '''
        cls = zoo.get_cls(model_arch, num_classes, device, **kwargs)
        return cls 
    
    @staticmethod
    def get_optimizer(net:nn.Module, optimizer:str, lr:float=0.01, weight_decay:float=1e-4, momentum:float=0.9, **kwargs):
        assert optimizer in ['sgd', 'adam'], 'Enter a valid optimizer.'
        if optimizer == 'sgd':
            optim = torch.optim.SGD(net.parameters(), lr, weight_decay=weight_decay, momentum=momentum)
        elif optimizer == 'adam':
            optim = torch.optim.Adam(net.parameters(), lr, **kwargs)
        return optim

    @staticmethod
    def get_scheduler(optimizer:Optimizer, scheduler:str, step_size:int=20, gamma:float=0.2, **kwargs):
        assert scheduler in ['steplr'], 'Enter a valid scheduler.'
        if scheduler == 'steplr':
            sdl = StepLR(optimizer, step_size=step_size, gamma=gamma)
        return sdl

    @staticmethod
    def get_criteria(criteria:str, **kwargs):
        assert criteria in ['crossentropy', 'softcrossentropy'], 'Enter a valid criteria.'
        if criteria == 'crossentropy':
            criteria = nn.CrossEntropyLoss()
        elif criteria == 'softcrossentropy':
            criteria = SoftCrossEntropyLoss()
        return criteria

    def load_ckpt(self, pth_ckpt:str):
        self.classifier.load_state_dict(torch.load(pth_ckpt))
        return

    def train_1_ep(self, train_loader:DataLoader, cur_ep:int) -> tuple:
        '''
        @Desc: To train the classifer network for one epoch.
        @Args:
            train_loader: The dataloader for the training dataset;
            cur_ep: The current epoch number;
        '''
        '''Init'''
        self.classifier.train()
        pbar = tqdm.tqdm(train_loader)
        pbar.set_description('Step Loop, Epoch:%d, Loss: Inf'%(cur_ep))
        loss_meter = AverageMeter()
        num_correct = 0
        total = 0
        
        '''Main loop'''
        for X, Y in pbar:
            X, Y = X.to(self.device), Y.to(self.device)
            
            self.optimizer.zero_grad()
            Y_ = self.classifier(X)
            loss = self.criteria(Y_, Y)  # CE by default
            loss.backward()
            self.optimizer.step()
            if len(Y.shape) > 1:
                if Y.shape[1] == 1:
                    num_correct += torch.sum(torch.argmax(Y_, 1) == Y)
                else:
                    num_correct += torch.sum(torch.argmax(Y_, 1) == torch.argmax(Y, 1))
            total += Y_.shape[0]

            pbar.set_description('Step Loop, Epoch:%d, Loss: %.4f'%(cur_ep, loss.item()))
            loss_meter.update(loss.item())
        
        '''Scheduler steps'''
        if self.scheduler is not None:
            self.scheduler.step()
        
        acc = num_correct / total
        loss = loss_meter.get_avg()
            
        return acc, loss

    @torch.no_grad()
    def test_classifier(self, test_loader:DataLoader) -> tuple:
        '''
        @Desc: To test the classifier network with the test dataset.
        @Args:
            test_loader: The dataloader for the test dataset;
        '''
        if test_loader is None:
            return 0., 0.
        else:
            self.classifier.eval()
            num_correct = 0
            total = 0
            
            loss_meter = AverageMeter()
            
            '''Main loop'''
            step_counter = 0
            for X, Y in tqdm.tqdm(test_loader, desc='Test Loop'):
                X, Y = X.cuda(), Y.cuda()
                Y_ = self.classifier(X)
                if isinstance(self.criteria, SoftCrossEntropyLoss):
                    Y = torch.nn.functional.one_hot(Y)
                    Y = Y.float()
                loss = self.criteria(Y_, Y).item()

                num_correct += torch.sum(torch.argmax(Y_, 1) == Y)
                total += Y_.shape[0]
                step_counter += 1
                loss_meter.update(loss)
            # Eval ends

            acc = num_correct / total
            loss = loss_meter.get_avg()
            
            return acc, loss

    def save_best_classifier(self, acc:float, acc_best:float):
        pth_save = py.join(self.dir_out, 'best.pth')
        if acc > acc_best:
            acc_best = acc
            torch.save(self.classifier.state_dict(), pth_save)
        return acc_best

    def train_classifier(self, n_ep:int, train_loader:DataLoader, test_loader:DataLoader, ow:bool=True, **kwargs):
        '''Init'''
        if ow and pth_exist(self.dir_out):
            print('Purging the existing directory %s'%(self.dir_out))
            os.system('rm -rf %s'%(self.dir_out))
        if pth_exist(self.dir_out):
            raise Exception('The output dir exists. Consider to delete it or overwrite it.')
        timer = Timer()
        acc_best = 0.
        pth_log = py.join(self.dir_out, 'training_log.txt')
        py.mkdir(self.dir_out)
        
        for ep in range(n_ep):
            """Train loop"""
            acc_train, loss_train = self.train_1_ep(train_loader, cur_ep=ep)

            """Test loop"""
            acc_test, loss_test = self.test_classifier(test_loader)

            # Save the best model
            acc_best = self.save_best_classifier(acc_test, acc_best)
            
            """Write Log"""
            pth_log = py.join(self.dir_out, 'training_log.txt')
            with open(pth_log, 'a') as f:
                if ep == 0:
                    f.write('EP\tTrainLoss\tTrainAcc\tTestLoss\tTestAcc\n')
                f.write('%d\t%.4f\t%.4f\t%.4f\t%.4f\n'%(ep, loss_train, acc_train, loss_test, acc_test))
        
        '''Log time'''
        T = timer.end()
        with open(pth_log, 'a') as f:
            f.write('Time Consumption: %.5f seconds\n'%T)
                    
        return

    # @torch.no_grad()
    # def get_feats_labels(net:nn.Module, loader:DataLoader):
    #     feats = None
    #     labels = None
    #     for x, y_true in tqdm.tqdm(loader, desc='Extracting features.'):
    #         x = x.cuda()
    #         y_true = y_true.cuda()
    #         x = net.get_feats(x)
    #         # Stack it up
    #         if feats == None:
    #             feats = x
    #         else:
    #             feats = torch.cat((feats, x), 0)
            
    #         if labels == None:
    #             labels = y_true
    #         else:
    #             labels = torch.cat((labels, y_true), 0)
            
    #     return feats, labels
        
    @torch.no_grad()
    def get_output(self, X:Tensor) -> Tensor:
        self.classifier.eval()
        return self.classifier(X)
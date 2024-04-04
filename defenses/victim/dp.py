'''
@Desc: Query Unlearning.
'''
import sys
# sys.path.append('../../..')
from typing import Any
from torch.nn import Module
from torch.optim import Optimizer
import torch
from torch import Tensor
import tqdm
import random
import numpy as np
from scipy.optimize import minimize
from .blackbox import Blackbox
import utils.operator as opt
from defenses import datasets
from torch.utils.data import DataLoader, TensorDataset
from defenses.models.mapping_net import feature_mapping_net
from utils.lossfunc import SupervisedContrastiveLoss
from matplotlib import pyplot as plt
import torch.nn.functional as F
from utils.classifier import ClassifierTrainer
from utils.data import DataFetcher
from defenses import config as CFG
from torch.autograd import detect_anomaly
import math
from torch import nn


class DP(Blackbox):
    def __init__(self, epsilon:float,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        '''Hyperparameters'''
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.epsilon = epsilon
    
    def truncation(self, outputs, num):
        for i in range(0, num):
            outputs[outputs.index(min(outputs))] = 100
        for i in range(0, len(outputs)):
            if (outputs[i] == 100):
                outputs[i] = 0
        return outputs

    def defense(self, outputs, eps):
        sumExp = 0.0
        for i in range(0, len(outputs)):
            sumExp = sumExp + math.exp(eps / 2 * outputs[i])
        for i in range(0, len(outputs)):
            outputs[i] = math.exp(eps / 2 * outputs[i]) / sumExp   
        return outputs
        
    
    @torch.no_grad()
    def __call__(self, query_input:Tensor, stat:bool=True, return_origin:bool=False):
        self.model.eval()
        batch_size = query_input.shape[0]
        
        query_input = query_input.to(self.device)

        '''Perturbation'''
        sm_org = self.model(query_input) 
        sm_org = torch.nn.functional.softmax(sm_org, dim=1)
        # print(sm_org.shape)
        sm_ptb = sm_org.tolist() 
        # print(len(sm_ptb), len(sm_ptb[0]))
        for i in range(len(sm_ptb)):
            sm_ptb[i] = self.defense(sm_ptb[i], self.epsilon)
            # sm_ptb[i] = self.truncation(sm_ptb[i], 5)
        sm_ptb = torch.Tensor(sm_ptb).cpu().detach()
        sm_org = sm_org.cpu().detach()
        # print(torch.sum(sm_ptb, dim=1))
        # print(sm_org - sm_ptb)
        # print(torch.max(sm_org, dim=1)[1])
        # print(torch.max(sm_ptb, dim=1)[1])
        # print(torch.sum(torch.max(sm_org, dim=1)[1] != torch.max(sm_ptb, dim=1)[1]))
        # print(len(sm_ptb))
        
        # print(noise)
        
        # std_out = '====== Cumulative Report: Query: %d, Within: %d, Recorded: %d, Reversed: %d ======\n'%(self.counter_query, self.counter_within, self.counter_record, self.counter_reverse)
        # with open(opt.os.join(self.out_dir, 'queen_log.txt'), 'a') as f:
        #     f.write(std_out)
        #     f.close()
        
        self.call_count += query_input.shape[0]
        
        if return_origin:
            return sm_ptb, sm_org
        else:
            return sm_ptb
    

def debug():
    defender_dp = DP
    defender_dp = defender_dp.from_modeldir(
                                    model_dir='experiment/victim/CIFAR10-vgg16_bn-train-nodefense',
                                    n_classes=10,
                                    epsilon=0.1,
                                )
    
    X = torch.rand((100, 3, 32, 32)).cuda()
    a = defender_dp(X)
    return

if __name__ == '__main__':
    debug()
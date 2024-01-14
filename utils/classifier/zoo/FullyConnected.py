import torch
import torch.nn as nn

class FullyConnected(nn.Module):
    def __init__(self, num_classes:int, in_dim:int=100, depth:int=3, n_feat:int=256, base:int=2, **kwargs) -> None:
        super(FullyConnected, self).__init__()
        print('check')
        assert depth > 1, 'The depth of the network must be greater than 1!'
        feature = ()
        
        '''Create feature block'''
        for i in range(0, depth):
            if i == 0:
                feature += (nn.Linear(in_dim, base**i*n_feat),)
            else:
                feature += (nn.Linear(base**(i-1)*n_feat, base**i*n_feat),)
            feature += (nn.BatchNorm1d(base**i*n_feat),)
            feature += (nn.ReLU(),)
        self.feature = nn.Sequential(*feature)
        
        '''Create classifier block'''
        self.classifier = nn.Linear(base**(depth-1)*n_feat, num_classes)
    
    def forward(self, x:torch.Tensor) -> torch.Tensor:
        x = self.feature(x)
        x = self.classifier(x)
        return x


'''Debug'''
if __name__ == '__main__':
    fc = FullyConnected(2, 100, 3).cuda()
    x = torch.rand((3, 100)).cuda()
    print(fc(x))
        
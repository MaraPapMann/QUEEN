'''
@Desc: Compute the cluster centers and mean feature-center distances for the training dataset in all classes using the mapping network
'''
import torch
import argparse
import random
import MHC.GeneralOperation.pylib as py
from torch import Tensor
import tqdm
import SensitivityAnalysis as SA
import NetworkTrainer as Trainer
import DataFetcher as Fetcher
from torch.utils.data import TensorDataset, DataLoader
import time


def main():
    '''
    Parse Arguments
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_arch', nargs='?', type=str, default='res34')
    parser.add_argument('--n_classes', nargs='?', type=int, default=10)
    parser.add_argument('--pth_ckpt', nargs='?', type=str, default='exp/20230904/map_net_ckpt/map_net_res34_mnist.pt')
    parser.add_argument('--pth_feats', nargs='?', type=str, default='exp/20230904/training_feats/feats_res34_mnist.pt')
    parser.add_argument('--dir_exp', nargs='?', type=str, default='exp/20230904')
    parser.add_argument('--dataset_train', nargs='?', type=str, default='mnist')
    parser.add_argument('--bs', nargs='?', type=int, default=1024)
    parser.add_argument('--seed', nargs='?', type=int, default=27)
    parser.add_argument('--img_size', nargs='?', type=int, default=32)
    parser.add_argument('--in_dim', nargs='?', type=int, default=512)
    parser.add_argument('--out_dim', nargs='?', type=int, default=2)
    parser.add_argument('--num_layers', nargs='?', type=int, default=4)
    parser.add_argument('--step_down', nargs='?', type=int, default=4)
    args = parser.parse_args()
    

    '''
    Initialize
    '''
    model_arch = args.model_arch
    n_classes = args.n_classes
    pth_ckpt = args.pth_ckpt
    seed = args.seed
    dir_exp = args.dir_exp
    pth_feats = args.pth_feats
    dataset_train = args.dataset_train
    drop_last = False
    shuffle = False
    bs = args.bs
    in_dim = args.in_dim
    out_dim = args.out_dim
    num_layers = args.num_layers
    step_down = args.step_down
    
    dir_save = py.join(dir_exp, 'sensitivity_analysis')
    pth_save_centers = py.join(dir_save, 'centers_%s_%s.pt'%(model_arch, dataset_train))
    pth_save_avgdist = py.join(dir_save, 'avgdist_%s_%s.pt'%(model_arch, dataset_train))
    pth_save_feats2d = py.join(dir_save, 'feats2d_%s_%s.pt'%(model_arch, dataset_train))
    py.mkdir(dir_exp)
    py.mkdir(dir_save)
    
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
    
    
    '''
    Set random seed
    '''
    torch.manual_seed(seed)
    random.seed(seed)
    
    
    '''
    Set mapping net
    '''
    map_net = Trainer.get_mapping_net(in_dim, out_dim, num_layers, step_down)
    map_net.load_state_dict(torch.load(pth_ckpt))
    map_net = map_net.to(device)
    
    
    '''
    Set dataset
    '''
    D_feats = torch.load(pth_feats)
    D_feats = TensorDataset(D_feats[0], D_feats[1])
    feat_loader = DataLoader(D_feats, bs, True)
    
    
    '''
    Get 2D features
    '''
    map_net.eval()
    feats_2d, labels = None, None
    with torch.no_grad():
        for xs, ys in tqdm.tqdm(feat_loader):
            xs = xs.to(device)
            ys = ys.to(device)
            
            xs = map_net(xs)
            
            if feats_2d == None:
                feats_2d = xs
            else:
                feats_2d = torch.cat((feats_2d, xs))
            
            if labels == None:
                labels = ys
            else:
                labels = torch.cat((labels, ys))
    torch.save((feats_2d.detach().cpu(), labels.detach().cpu()), pth_save_feats2d)
    
    
    '''
    Compute the cluster centers and MFC distances
    '''
    centers, avgdist = SA.sensitivity_analysis(feats_2d, labels, n_classes)
    torch.save(centers, pth_save_centers)
    torch.save(avgdist, pth_save_avgdist)
    

if __name__ == '__main__':
    start = time.time()
    main()
    end = time.time()
    print('Time  consumption of sensitivity analysis: %.4f seconds'%(end - start))
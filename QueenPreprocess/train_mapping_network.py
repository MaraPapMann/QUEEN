'''
@Desc: Train the mapping network
'''
import os
cwd = os.getcwd()
import sys
sys.path.insert(0, cwd)
import torch
import argparse
import random
import MHC.GeneralOperation.pylib as py
from MHC.DeepLearning.Torch.LossFunc.supervised_contrastive_loss import SupConLoss
import tqdm
from MHC.utils.AverageMeter import AverageMeter
from matplotlib import pyplot as plt
import NetworkTrainer as Trainer
import DataFetcher as Fetcher
from torch.utils.data import DataLoader, TensorDataset
import time
# from MHC.DeepLearning.Torch.LossFunc.hierarchical_contrastive_loss import HierConLoss


def main():
    '''
    Parse Arguments
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_arch', nargs='?', type=str, default='res34')
    parser.add_argument('--pth_feats', nargs='?', type=str, default='exp/20230904/training_feats/feats_res34_mnist.pt')
    parser.add_argument('--dir_exp', nargs='?', type=str, default='exp/20230904')
    parser.add_argument('--dataset_train', nargs='?', type=str, default='mnist')
    parser.add_argument('--n_ep', nargs='?', type=int, default=100)
    parser.add_argument('--bs', nargs='?', type=int, default=1024)
    parser.add_argument('--seed', nargs='?', type=int, default=27)
    parser.add_argument('--img_size', nargs='?', type=int, default=32)
    parser.add_argument('--in_dim', nargs='?', type=int, default=512)
    parser.add_argument('--out_dim', nargs='?', type=int, default=2)
    parser.add_argument('--num_layers', nargs='?', type=int, default=4)
    parser.add_argument('--step_down', nargs='?', type=int, default=4)
    parser.add_argument('--lr', nargs='?', type=float, default=0.001)
    parser.add_argument('--beta1', nargs='?', type=float, default=0.9)
    parser.add_argument('--beta2', nargs='?', type=float, default=0.99)
    parser.add_argument('--shuffle', action='store_true')
    parser.add_argument('--drop_last', action='store_true')
    parser.add_argument('--step_size', nargs='?', type=int, default=10)
    parser.add_argument('--gamma', nargs='?', type=float, default=0.5)
    args = parser.parse_args()
    

    '''
    Initialize
    '''
    model_arch = args.model_arch
    seed = args.seed
    dir_exp = args.dir_exp
    pth_feats = args.pth_feats
    dataset_train = args.dataset_train
    drop_last = args.drop_last
    shuffle = args.shuffle
    bs = args.bs
    in_dim = args.in_dim
    out_dim = args.out_dim
    num_layers = args.num_layers
    step_down = args.step_down
    beta1 = args.beta1
    beta2 = args.beta2
    step_size = args.step_size
    gamma = args.gamma
    lr = args.lr
    n_ep = args.n_ep
    
    dir_save = py.join(dir_exp, 'map_net_ckpt')
    pth_save = py.join(dir_save, 'map_net_%s_%s.pt'%(model_arch, dataset_train))
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
    
    optimizer = torch.optim.Adam(map_net.parameters(), lr, betas=(beta1, beta2))
    
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size, gamma, verbose=True)
    
    criteria = SupConLoss()
    # criteria = HierConLoss()
    
    
    '''
    Set dataset
    '''
    D_feats = torch.load(pth_feats)
    D_feats = TensorDataset(D_feats[0], D_feats[1])
    feat_loader = DataLoader(D_feats, bs, shuffle)
    
    
    '''
    Train the mapping net
    '''
    loss_avgmeter = AverageMeter()
    map_net.train()
    for ep in range(n_ep):
        pbar = tqdm.tqdm(feat_loader)
        pbar.set_description('Ep: %d, Loss: nan'%ep)
        for xs, ys in pbar:
            xs = xs.to(device)
            ys = ys.to(device)

            optimizer.zero_grad()
            xs = map_net(xs)
            xs = xs.unsqueeze(1)
            torch.nn.functional.one_hot()
            print(ys.shape)
            loss = criteria(xs, ys)
            
            loss.backward()
            optimizer.step()
            
            pbar.set_description('Ep: %d, Loss: %.4f'%(ep, loss.item()))
        
        '''Get average loss'''
        with torch.no_grad():
            for xs, ys in pbar:
                xs = xs.to(device)
                ys = ys.to(device)
                
                # print(xs.shape)
                # quit()

                xs = map_net(xs)
                xs = xs.unsqueeze(1)
                loss = criteria(xs, ys)
                loss_avgmeter.update(loss.item())
                
        cur_loss_avg = loss_avgmeter.get_avg()
        print(cur_loss_avg)
        # print('Average loss: %.4f at Epoch %d'%(cur_loss_avg, ep))
        if cur_loss_avg == loss_avgmeter.get_best():
            torch.save(map_net.state_dict(), pth_save)
        loss_avgmeter.reset()
        
        scheduler.step()
    
    
    '''
    Plot the 2D features
    '''
    with torch.no_grad():
        map_net.eval()
        feats_2d, labels = None, None
        D_feats_sub = Fetcher.get_subsets(D_feats, 2, 1000)[0]
        feat_loader = DataLoader(D_feats_sub, 100)
        for xs, ys in feat_loader:
            xs = xs.to(device)
            xs = map_net(xs)
            if feats_2d == None:
                feats_2d = xs
            else:
                feats_2d = torch.cat((feats_2d, xs), 0)
            
            if labels == None:
                labels = ys
            else:
                labels = torch.cat((labels, ys), 0)
    feats_2d = feats_2d.cpu()
    labels = labels.cpu()
    plt.scatter(feats_2d[:, 0], feats_2d[:, 1], 1, labels)
    plt.savefig('%s.png'%pth_save)
    

if __name__ == '__main__':
    start = time.time()
    main()
    end = time.time()
    print('Time  consumption of training the mapping network: %.4f seconds'%(end - start))
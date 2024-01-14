'''
@Desc: Train a set of shadow models as an ensemble to represent the piracy model.
'''
import torch
import argparse
import random
import MHC.GeneralOperation.pylib as py
import NetworkTrainer as Trainer
import DataFetcher as Fetcher
from torch.utils.data import DataLoader


def main():
    '''
    Parse Arguments
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_ep', nargs='?', type=int, default=1)
    parser.add_argument('--n_classes', nargs='?', type=int, default=10)
    parser.add_argument('--lr', nargs='?', type=float, default=0.1)
    parser.add_argument('--model_arch', nargs='?', type=str, default='res18')
    parser.add_argument('--optimizer', nargs='?', type=str, default='sgd')
    parser.add_argument('--scheduler', nargs='?', type=str, default='steplr')
    parser.add_argument('--criteria', nargs='?', type=str, default='crossentropy')
    parser.add_argument('--dir_train', nargs='?', type=str, default='../../data')
    parser.add_argument('--dir_test', nargs='?', type=str, default='../../data')
    parser.add_argument('--dir_exp', nargs='?', type=str, default='exp/20230904')
    parser.add_argument('--dataset_train', nargs='?', type=str, default='mnist')
    parser.add_argument('--dataset_test', nargs='?', type=str, default='mnist')
    parser.add_argument('--bs', nargs='?', type=int, default=500)
    parser.add_argument('--seed', nargs='?', type=int, default=27)
    parser.add_argument('--step_size', nargs='?', type=int, default=5)
    parser.add_argument('--gamma', nargs='?', type=float, default=0.02)
    parser.add_argument('--momentum', nargs='?', type=float, default=0.)
    parser.add_argument('--weight_decay', nargs='?', type=float, default=0.)
    parser.add_argument('--drop_last', action='store_true')
    parser.add_argument('--img_size', nargs='?', type=int, default=32)
    parser.add_argument('--n_shadow', nargs='?', type=int, default=10)
    parser.add_argument('--n_samples', nargs='?', type=int, default=500)
    parser.add_argument('--n_per_class', nargs='?', type=int, default=500)
    args = parser.parse_args()
    

    '''
    Initialize
    '''
    n_ep = args.n_ep
    model_arch = args.model_arch
    n_classes = args.n_classes
    seed = args.seed
    optimizer = args.optimizer
    scheduler = args.scheduler
    dir_exp = args.dir_exp
    root_train = args.dir_train
    root_test = args.dir_test
    lr = args.lr
    step_size = args.step_size
    gamma = args.gamma
    criteria = args.criteria
    dataset_train = args.dataset_train
    dataset_test = args.dataset_test
    bs = args.bs
    drop_last = args.drop_last
    momentum = args.momentum
    weight_decay = args.weight_decay
    img_size = args.img_size
    n_shadow = args.n_shadow
    n_samples = args.n_samples
    
    # Make directories
    dir_ckpt = py.join(dir_exp, 'shadow_ckpt')
    pth_save = py.join(dir_ckpt, '%s_%s'%(model_arch, dataset_train))
    py.mkdir(dir_exp)
    py.mkdir(dir_ckpt)
    
    # Set random seed
    torch.manual_seed(seed)
    random.seed(seed)
    
    
    '''
    Set shadow models
    '''
    lst_shadow_models = []
    for i in range(n_shadow):
        lst_shadow_models.append(Trainer.get_classifier(model_arch, n_classes))
    
    
    '''
    Set dataset
    '''
    D_train = Fetcher.load_dataset(dataset_train, root_train, True, True, img_size)
    D_train_subs = Fetcher.get_subsets(D_train, n_shadow, n_samples)
    
    D_test = Fetcher.load_dataset(dataset_test, root_test, False, True, img_size)
    test_loader = DataLoader(D_test, bs, False)
    
    
    '''
    Train shadow models
    '''
    for i, shadow_model in enumerate(lst_shadow_models):
        optim = Trainer.get_optimizer(shadow_model, optimizer, lr, momentum=momentum, weight_decay=weight_decay, nesterov=True)
        sdl = Trainer.get_scheduler(optim, scheduler, step_size=step_size, gamma=gamma)
        crt = Trainer.get_criteria(criteria)
        train_loader = DataLoader(D_train_subs[i], bs, True)
        Trainer.train_classifier(shadow_model, optim, sdl, crt, n_ep, train_loader, test_loader, pth_save+'.%d.pt'%i, use_scheduler=False)


if __name__ == '__main__':
    main()
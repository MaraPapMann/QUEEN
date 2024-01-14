'''
@Desc: Query Unlearning.
'''
import sys
sys.path.append('../..')
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
from defenses.models.mapping_net import FeatMapNet
from utils.lossfunc import SupervisedContrastiveLoss
from matplotlib import pyplot as plt
import torch.nn.functional as F
from utils.classifier import ClassifierTrainer
from utils.data import DataFetcher
from defenses import config as CFG


class Queen(Blackbox):
    def __init__(self, r:float, threshold:float, k:int,
                        in_dim:int, out_dim:int, num_layers:int, step_down:int, 
                        shadow_arch:str, *args, **kwargs):
        super().__init__(*args, **kwargs)
        print(kwargs)


        '''Set device'''
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'


        '''Hyperparameters'''
        self.alpha = 1.
        self.beta = 1.
        self.r = r
        self.threshold = threshold
        self.k = k  # Number of shadow models used in each falsified softmax generation
        self.num_shadows = 10
        assert self.k <= self.num_shadows, 'k must be less or equal to the number of shadow models!'
        self.out_dir=kwargs['out_path']
        self.counter_within = 0
        self.counter_reverse = 0
        self.counter_record = 0
        self.counter_query = 0
        self.num_classes = kwargs['num_classes']
        self.cqs = torch.zeros(self.num_classes).to(self.device)
        self.record = {}
        for i in range(self.num_classes):
            self.record.update({i:None})


        '''Check training features'''
        self.pth_feature_dataset = opt.os.join(self.out_dir, 'training_feats/training_feats.pt')
        if opt.os.pth_exist(self.pth_feature_dataset):
            print('=> Loading the existing training features...')
            to_load = torch.load(self.pth_feature_dataset)
            features, labels = to_load['features'], to_load['labels']
            del to_load
        else:
            print('=> No training features. Extracting training features...')
            opt.os.mkdir(opt.os.get_dir(self.pth_feature_dataset))
            modelfamily = datasets.dataset_to_modelfamily[self.dataset_name]
            transform = datasets.modelfamily_to_transforms[modelfamily]['train']
            train_set = datasets.__dict__[self.dataset_name](train=True, transform=transform)
            bs = 256
            train_loader = DataLoader(train_set, bs, True)
            features, labels = None, None
            for x, y in tqdm.tqdm(train_loader, desc='Extracting training features...'):
                x = x.to(self.device)
                x = self.model.get_feats(x)
                x = x.detach().cpu()
                y = y.detach().cpu()
                features = opt.tensor.cat_tensors(features, x)
                labels = opt.tensor.cat_tensors(labels, y)
            self.feature_dataset = TensorDataset(features, labels)
            to_save = {'features':features, 'labels':labels}
            torch.save(to_save, self.pth_feature_dataset)
            print('=> Training features extracted and saved! Shape of the training features:', features.shape)
            del bs, modelfamily, transform, train_set, train_loader, to_save

        
        '''Get feature centers'''
        self.pth_feature_centers = opt.os.join(self.out_dir, 'feature_centers/feature_centers.pt')
        if opt.os.pth_exist(self.pth_feature_centers):
            print('=> Loading feature centers...')
            self.centers = torch.load(self.pth_feature_centers)
        else:
            self.centers = self.get_feat_centers(features, labels, self.num_classes)
            opt.os.mkdir(opt.os.get_dir(self.pth_feature_centers))
            torch.save(self.centers.detach().cpu(), self.pth_feature_centers)
        

        '''Check mapping network'''
        self.mapping_net = FeatMapNet(in_dim=in_dim, out_dim=out_dim, num_layers=num_layers, step_down=step_down).to(self.device)
        self.pth_mapping_net = opt.os.join(self.out_dir, 'mapping_net/mapping_net.pt')
        if opt.os.pth_exist(self.pth_mapping_net):
            print('=> Loading the mapping net...')
            self.mapping_net.load_state_dict(torch.load(self.pth_mapping_net))
        else:
            print('=> No mapping net. Training a mapping net from scratches...')
            opt.os.mkdir(opt.os.get_dir(self.pth_mapping_net))
            self.mapping_net.train()
            num_ep = 100
            bs = 10000
            lr = 0.01
            beta1, beta2 = 0.9, 0.99
            step_size, gamma = 20, 0.5
            optimizer = torch.optim.SGD(self.mapping_net.parameters(), lr, momentum=0.9, weight_decay=1e-4)
            # optimizer = torch.optim.Adam(self.mapping_net.parameters(), lr, betas=(beta1,beta2))
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size, gamma)
            criteria = SupervisedContrastiveLoss()
            feat_loader = DataLoader(self.feature_dataset, bs, True)
            for ep in tqdm.trange(num_ep, desc='Training the mapping network...'):
                pbar = tqdm.tqdm(feat_loader)
                pbar.set_description('Ep: %d, Loss: nan'%ep)
                for xs, ys in pbar:
                    xs = xs.to(self.device)
                    ys = ys.to(self.device)

                    self.mapping_net.zero_grad()
                    optimizer.zero_grad()
                    xs = self.mapping_net(xs)
                    xs = xs.unsqueeze(1)
                    loss = criteria(xs, ys)
                    
                    loss.backward()
                    optimizer.step()
                    pbar.set_description('Ep: %d, Loss: %.4f'%(ep, loss.item()))
                    
                scheduler.step()
            
            '''Plot the 2D features'''
            pth_fig = opt.os.join(opt.os.get_dir(self.pth_mapping_net), 'feature_2d_map.png')
            with torch.no_grad():
                self.mapping_net.eval()
                feats_2d, labels = None, None
                feat_loader = DataLoader(self.feature_dataset, 1000)
                for xs, ys in feat_loader:
                    xs = xs.to(self.device)
                    xs = self.mapping_net(xs)
                    feats_2d = opt.tensor.cat_tensors(feats_2d, xs)
                    labels = opt.tensor.cat_tensors(labels, ys)
            feats_2d = feats_2d.cpu()
            labels = labels.cpu()
            plt.scatter(feats_2d[:, 0], feats_2d[:, 1], 1, labels)
            plt.savefig('%s'%pth_fig)
            torch.save(self.mapping_net.state_dict(), self.pth_mapping_net)
            print('=> Mapping network trained and saved!')
            del num_ep, bs, lr, beta1, beta2, step_size, gamma, optimizer, scheduler, criteria, feat_loader
            
            print('=> Saving the 2D features...')
            to_save = {'features':feats_2d, 'labels':labels}
            self.pth_features_2d = opt.os.join(self.out_dir, 'features_2d/features_2d.pt')
            opt.os.mkdir(opt.os.get_dir(self.pth_features_2d))
            torch.save(to_save, self.pth_features_2d)
            del to_save, feats_2d, labels
        
        
        '''Sensitivity Analysis'''
        self.pth_sa_res = opt.os.join(self.out_dir, 'sensitivity_analysis/sensitivity_analysis.pt')
        if opt.os.pth_exist(self.pth_sa_res):
            print('=> Loading the sensitivity analysis results...')
            sa_res = torch.load(self.pth_sa_res)
            self.centers_2d = sa_res['centers']
            self.avgdist = sa_res['avgdist']
            del sa_res
        else:
            print('=> No sensitivity analysis results. Performing sensitivity analysis...')
            assert opt.os.pth_exist(opt.os.join(self.out_dir, 'features_2d/features_2d.pt')), '2D features not exist!'
            this_dict = torch.load(opt.os.join(self.out_dir, 'features_2d/features_2d.pt'))
            feats_2d, labels = this_dict['features'], this_dict['labels']
            self.centers_2d, self.avgdist = self.sensitivity_analysis(feats_2d, labels, kwargs['num_classes'])
            sa_res_to_save = {'centers':self.centers, 'avgdist':self.avgdist}
            opt.os.mkdir(opt.os.get_dir(self.pth_sa_res))
            torch.save(sa_res_to_save, self.pth_sa_res)
            del this_dict, feats_2d, labels, sa_res_to_save
        
        
        '''Shadow models'''
        self.dir_shadow = opt.os.join(self.out_dir, 'shadow_models')
        self.lst_pth_shadows = opt.os.get_files(self.dir_shadow, '.pt')
        self.shadow_models = []
        if opt.os.pth_exist(self.dir_shadow) and len(self.lst_pth_shadows) != 0:
            print('=> Loading shadow models...')
            for pth in self.lst_pth_shadows:
                trainer = ClassifierTrainer(shadow_arch, self.num_classes, 'sgd', 'steplr', 0.01, 'crossentropy', '')
                trainer.classifier.load_state_dict(torch.load(pth))
                self.shadow_models.append(trainer.classifier)
            del trainer
        else:
            print('No shadow models. Training shadow models...')
            dir_temp = opt.os.join(self.dir_shadow, 'temp')
            for i in range(self.num_shadows):
                trainer = ClassifierTrainer(shadow_arch, self.num_classes, 'sgd', 'steplr', 0.01, 'crossentropy', dir_temp)
                fetcher = DataFetcher()
                train_set = fetcher.load_dataset(self.dataset_name, CFG.DATASET_ROOT, True, True, 32)
                train_loader = DataLoader(train_set, 256, True)
                trainer.train_classifier(1, train_loader, None)
                torch.save(trainer.classifier.state_dict(), opt.os.join(self.dir_shadow, f'{shadow_arch}_{i}_.pt'))
                self.shadow_models.append(trainer.classifier)
            opt.os.rm_rf(dir_temp)
            del dir_temp, trainer, fetcher, train_set, train_loader
        
        print('=> Queen initialization complete!')
    
    @staticmethod
    def get_eu_dist(x:Tensor, y:Tensor) -> float:
        '''Get the Euclidean distance between two vectors.'''
        assert x.shape == y.shape
        if len(x.shape) == 2:
            eu_dist = torch.sqrt(torch.sum(torch.pow(x - y, 2), dim=1))
        else:
            eu_dist = torch.sqrt(torch.sum(torch.pow(x - y, 2), dim=0))
        return eu_dist

    def get_amct_avgdist(self, feats_2D:Tensor, labels:Tensor, label:int):
        '''Get the arithmetic center of the cluster with the label.'''
        feats_2D = feats_2D[labels == label]
        amct = feats_2D.mean(dim=0).unsqueeze(0)
        amct_ = amct.repeat(feats_2D.shape[0], 1)
        dist = torch.sort(self.get_eu_dist(feats_2D, amct_))[0]
        avg_dist = torch.mean(dist)
        return amct, avg_dist

    def sensitivity_analysis(self, feats_2D:Tensor, labels:Tensor, num_classes:int):
        '''Sensitivity analysis of the training feature space.'''
        centers, avgdist = {}, {}
        for label in range(num_classes):
            amct, avg_dist = self.get_amct_avgdist(feats_2D, labels, label)
            centers.update({label:amct})
            avgdist.update({label:avg_dist})
        return centers, avgdist
        

    @classmethod
    def get_fc_dist(self, feat2d:Tensor, center:Tensor) -> float:
        '''
        @Desc: Get the feature-center distance
        '''
        fc_dist = self.get_eu_dist(feat2d, center)
        return fc_dist
    
    
    @classmethod
    def in_sensitive_region(self, fc_dist:float, rs:float) -> bool:
        if fc_dist < rs:
            return True
        else:
            return False


    @classmethod
    def no_previous_record(self, feat2d:Tensor, record:Tensor, r:float) -> bool:
        not_recorded = True
        if record == None:
            return not_recorded
        else:
            # print(record.shape, feat2d.shape)
            feat2d = feat2d.repeat(record.shape[0], 1)
            dist = self.get_eu_dist(feat2d, record)
            # print(dist)
            # print(torch.all(dist < r))
            if not torch.all(dist > r):
                not_recorded = False
            return not_recorded
        
    
    @classmethod
    def get_sqs(self, dist:float, avg_dist:float, alpha:float) -> float:
        sqs = 0.5 * torch.erfc((alpha * (dist - avg_dist)) / avg_dist)
        return sqs


    @classmethod
    def update_cqs(self, sqs:float, cqs:Tensor, label:int, r:float, rs:float) -> Tensor:
        cqs[label] += sqs.item()**2 * (r/rs)**2
        return cqs


    @classmethod
    def cqs_over_threshold(self, cqs, label, threshold) -> bool:
        ot = False
        if cqs[label] > threshold:
            ot = True
        return ot


    @classmethod
    def get_pir_softmax(self, lst_shadows:list, q:Tensor, k:int) -> Tensor:
        lst_idx = list(range(len(lst_shadows)))
        lst_idx = random.sample(lst_idx, k)
        
        with torch.no_grad():
            y_soft_pir = None
            for idx in lst_idx:
                lst_shadows[idx].eval()
                y_soft = lst_shadows[idx](q)
                if y_soft_pir == None:
                    y_soft_pir = y_soft
                else:
                    y_soft_pir += y_soft
            y_soft_pir /= k
            
        return y_soft_pir

    @staticmethod
    def cosine_similarity_objective(x, y) -> float:
        cosine_similarity = np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))
        return -cosine_similarity
    
    @staticmethod
    def constraint_function(x) -> float:
        return np.sum(x) - 1
    
    @staticmethod
    def get_feat_centers(feats:Tensor, labels:Tensor, n_classes:int) -> Tensor:
        centers = []
        for y in range(n_classes):
            cur_feats = feats[labels == y]
            cur_center = torch.mean(cur_feats, 0)
            centers.append(cur_center.detach().cpu().numpy())
        centers = Tensor(centers)
        return centers

    def gen_falsified_softmax(self, y_target:Tensor) -> Tensor:
        '''Init'''
        y_target = y_target.detach().cpu().numpy()
        y_fal = np.random.rand(len(y_target))
        bounds = [(0, 1) for _ in y_fal]
        constraint = {'type': 'eq', 'fun': self.constraint_function}
        result = minimize(self.cosine_similarity_objective, y_fal, args=(y_target,), method='SLSQP', constraints=constraint, bounds=bounds)
        y_fal = Tensor(result.x)
        return y_fal

    def falsify_gradient(self, y_soft_pro:Tensor, y_soft_pir:Tensor) -> Tensor:
        y_soft_fal = y_soft_pir * 2 - y_soft_pro
        '''Cosine similarity maximization'''
        for i in range(y_soft_fal.shape[0]):
            y_soft_fal[i] = self.gen_falsified_softmax(y_soft_fal[i])
        return y_soft_fal


    def find_farthest_center(self, feat:Tensor, feat_centers:Tensor) -> Tensor:
        feat, feat_centers = feat.to(self.device), feat_centers.to(self.device)
        feat = feat.unsqueeze(0).repeat(feat_centers.shape[0], 1)
        # print(feat.shape, feat_centers.shape)
        dist = self.get_eu_dist(feat, feat_centers)
        idx = torch.argmax(dist)
        return feat_centers[idx]

    @torch.no_grad()
    def perturb_sm(self, feat:Tensor, center:Tensor, net:Module, step_size:float=0.05) -> Tensor:
        v = center - feat
        v = v / v.norm()
        label = torch.argmax(net.feat_to_sm(feat))
        ptb_label = label
        feat_ptb = feat
        attempts = 0
        while True:
            feat_ptb_updated = feat_ptb + v * step_size
            ptb_label = torch.argmax(net.feat_to_sm(feat_ptb_updated))
            attempts += 1
            if ptb_label != label:
                break
            elif attempts > 20:
                break
            else:
                feat_ptb = feat_ptb_updated
        return net.feat_to_sm(feat_ptb)
    
    
    @torch.no_grad()
    def __call__(self, query_input:Tensor, stat:bool=True, return_origin:bool=False):
        '''
        @Desc:
            Update the CQS.
            Generate falsified softmax output.
        '''
        self.model.eval()
        self.mapping_net.eval()
        self.counter_query += query_input.shape[0]
        
        query_input = query_input.to(self.device)
        feats = self.model.get_feats(query_input)
        feats2d = self.mapping_net(feats)
        softmax_protectee = F.softmax(self.model(query_input), 1)
        labels_protectee = torch.argmax(softmax_protectee, 1)
    
        '''Compute the CQS, and perform gradient reverse based on the CQS'''
        # Sample-wise operation
        
        softmax_falsified = torch.zeros_like(softmax_protectee)
        
        for i in tqdm.trange(feats2d.shape[0], desc='Queen Processing...'):
            # if self.counter_query < 1024:
            #     self.counter_query += 1
            #     continue
            feat2d = feats2d[i]
            feat2d = feat2d.unsqueeze(0)
            label = labels_protectee[i].item()
            # print(label)
            # print(self.centers)
            center, avgdist = self.centers_2d[label].to(self.device), self.avgdist[label].to(self.device)
            cur_record = self.record[label]
            rs = avgdist * self.beta
            
            '''Is the query sensitive?'''
            sensitive = False
            fc_dist = self.get_fc_dist(feat2d, center)
            if self.in_sensitive_region(fc_dist, rs):
                self.counter_within += 1
                if self.no_previous_record(feat2d, cur_record, self.r):
                    sensitive = True
            
            if sensitive:
                cur_softmax_protectee = softmax_protectee[i]
                if self.cqs_over_threshold(self.cqs, label, self.threshold):
                    '''Reverse the gradient, if sensitive and the CQS over the threshold'''
                    # print('Checked 1.')
                    query = query_input[i].unsqueeze(0)
                    cur_softmax_pirate = self.get_pir_softmax(self.shadow_models, query, self.k)
                    cur_softmax_falsified = self.falsify_gradient(cur_softmax_protectee, cur_softmax_pirate)
                    softmax_falsified[i] = cur_softmax_falsified
                    self.counter_reverse += 1  # Count
                else:
                    '''Record the 2D feature and perturb the output, if sensitive but the CQS beneath the threshold'''
                    # print('Checked 2.')
                    # record this feature
                    self.record[label] = opt.tensor.cat_tensors(self.record[label], feat2d)
                    self.counter_record += 1
                    # Update the CQS
                    sqs = self.get_sqs(fc_dist, rs, self.alpha)
                    self.cqs = self.update_cqs(sqs, self.cqs, label, self.r, rs)
                    # Do not change the label, but perturb the output
                    feat = feats[i]
                    farthest_center = self.find_farthest_center(feat, self.centers)
                    cur_softmax_falsified = self.perturb_sm(feat.unsqueeze(0), farthest_center, self.model)
                    softmax_falsified[i] = cur_softmax_falsified
                    # softmax_falsified[i] = cur_softmax_protectee
            else:
                ''''''
                # Do not change the label, but perturb the output
                feat = feats[i]
                farthest_center = self.find_farthest_center(feat, self.centers)
                cur_softmax_falsified = self.perturb_sm(feat.unsqueeze(0), farthest_center, self.model)
                softmax_falsified[i] = cur_softmax_falsified
        
        print('====== Cumulative Report: Query: %d, Within: %d, Recorded: %d, Reversed: %d ======'%(self.counter_query, self.counter_within, self.counter_record, self.counter_reverse))
        
        if return_origin:
            return softmax_falsified, softmax_protectee
        else:
            return softmax_falsified
    

def debug():
    defender_queen = Queen(
                                        protectee_arch='res34',
                                        n_classes=10,
                                        alpha=0.5,
                                        beta=1.,
                                        r=0.02,
                                        threshold=0.5,
                                        k=5,
                                        pth_protectee_net_ckpt='exp/20230904/cls_ckpt/res34_mnist.pt',
                                        in_dim=512,
                                        out_dim=2,
                                        num_layers=4,
                                        step_down=4,
                                        pth_mapping_net_ckpt='exp/20230904/map_net_ckpt/map_net_res34_mnist.pt',
                                        dir_shadow='exp/20230904/shadow_ckpt',
                                        shadow_arch='res18',
                                        pth_sensitive_analysis='exp/20230904/sensitivity_analysis/sa_res_res34_mnist.pt',
                                        pth_feature_centers='exp/20230904/training_feats/feats_res34_mnist.pt.feat_centers.pt',
                                        )
    
    X = torch.rand((100, 3, 32, 32)).cuda()
    return

if __name__ == '__main__':
    debug()
import os
import datetime
import logging
import random
import numpy as np
from sklearn import metrics
from typing import Union
from collections import defaultdict
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn import DataParallel
from torch.utils.tensorboard import SummaryWriter
from metrics.base_metrics_class import calculate_metrics_for_train
from .base_detector import AbstractDetector
from detectors import DETECTOR
from networks import BACKBONE
from loss import LOSSFUNC
from networks.EffConv import *
from networks.CSC import *
from networks.FDD import *
import matplotlib.pyplot as plt
from PIL import Image


logger = logging.getLogger(__name__)


def vis_norm(var):
    var = (var - np.min(var)) / (np.max(var) - np.min(var))
    var = var * 255
    return var

def denormalize(tensor, mean, std):
        mean = torch.tensor(mean).view(-1, 1, 1).to(tensor.device)
        std = torch.tensor(std).view(-1, 1, 1).to(tensor.device)
        return tensor * std + mean
    
@DETECTOR.register_module(module_name='ucf')
class UCFDetector(AbstractDetector):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.num_classes = config['backbone_config']['num_classes']
        self.encoder_feat_dim = config['encoder_feat_dim']
        self.half_fingerprint_dim = self.encoder_feat_dim//2

        self.encoder_f = self.build_backbone(config)
        self.encoder_c = self.build_backbone(config)

        self.loss_func = self.build_loss(config)
        self.prob, self.label = [], []
        self.correct, self.total = 0, 0
        
        self.lr = nn.LeakyReLU(inplace=True)
        self.do = nn.Dropout(0.2)
        self.pool = nn.AdaptiveAvgPool2d(1)

        self.con_gan = Conditional_UNet()

        specific_task_number = len(config['train_dataset']) + 1  
        self.head_spe = Head(
            in_f=self.half_fingerprint_dim, 
            hidden_dim=self.encoder_feat_dim,
            out_f=specific_task_number
        )
        self.head_sha = Head(
            in_f=self.half_fingerprint_dim,
            hidden_dim=self.encoder_feat_dim, 
            out_f=self.num_classes
        )
        self.block_spe = Conv2d1x1(
            in_f=self.encoder_feat_dim,
            hidden_dim=self.half_fingerprint_dim, 
            out_f=self.half_fingerprint_dim
        )
        self.block_sha = Conv2d1x1(
            in_f=self.encoder_feat_dim, 
            hidden_dim=self.half_fingerprint_dim, 
            out_f=self.half_fingerprint_dim
        )

        self.EffConv = EffConv(dim=256, n_div=4)
        self.csc = CSC(dim=512)
        self.fdd = FDD(256,256)
        self.i = 0
        
    def build_backbone(self, config):
        backbone_class = BACKBONE[config['backbone_name']]
        model_config = config['backbone_config']
        backbone = backbone_class(model_config)
        state_dict = torch.load(config['pretrained'])
        for name, weights in state_dict.items():
            if 'pointwise' in name:
                state_dict[name] = weights.unsqueeze(-1).unsqueeze(-1)
        state_dict = {k:v for k, v in state_dict.items() if 'fc' not in k}
        backbone.load_state_dict(state_dict, False)
        logger.info('Load pretrained model successfully!')
        return backbone
    
    def build_loss(self, config):
        cls_loss_class = LOSSFUNC[config['loss_func']['cls_loss']]
        spe_loss_class = LOSSFUNC[config['loss_func']['spe_loss']]
        con_loss_class = LOSSFUNC[config['loss_func']['con_loss']]
        rec_loss_class = LOSSFUNC[config['loss_func']['rec_loss']]
        cls_loss_func = cls_loss_class()
        spe_loss_func = spe_loss_class()
        con_loss_func = con_loss_class(margin=3.0)
        rec_loss_func = rec_loss_class()
        loss_func = {
            'cls': cls_loss_func, 
            'spe': spe_loss_func,
            'con': con_loss_func,
            'rec': rec_loss_func,
        }
        return loss_func
    
    def features(self, data_dict: dict) -> torch.tensor:
        cat_data = data_dict['image']
        f_all = self.encoder_f.features(cat_data)
        c_all = self.encoder_c.features(cat_data)
        f_all = self.csc(f_all)
        c_all = self.csc(c_all)
        feat_dict = {'forgery': f_all, 'content': c_all}
        return feat_dict

    def classifier(self, features: torch.tensor) -> torch.tensor:
        f_spe = self.block_spe(features)
        f_share = self.block_sha(features)
        f_spel,f_speh = self.fdd(f_spe)
        f_spel = self.EffConv(f_spel)
        f_speh = self.EffConv(f_speh)
        f_spe = f_spel + f_speh

        f_sharel,f_shareh = self.fdd(f_share)
        f_sharel = self.EffConv(f_sharel)
        f_shareh = self.EffConv(f_shareh)
        f_share = f_sharel +f_shareh
        return f_spe, f_share
    
    def get_losses(self, data_dict: dict, pred_dict: dict) -> dict:
        if 'label_spe' in data_dict and 'recontruction_imgs' in pred_dict:
            return self.get_train_losses(data_dict, pred_dict)
        else: 
            return self.get_test_losses(data_dict, pred_dict)

    def get_train_losses(self, data_dict: dict, pred_dict: dict) -> dict:
        cat_data = data_dict['image']
        real_img, fake_img = cat_data.chunk(2, dim=0)
        reconstruction_image_1, \
        reconstruction_image_2, \
        self_reconstruction_image_1, \
        self_reconstruction_image_2 \
            = pred_dict['recontruction_imgs']
        label = data_dict['label']
        label_spe = data_dict['label_spe']
        pred = pred_dict['cls']
        pred_spe = pred_dict['cls_spe']

        # 1. classification loss for common features
        loss_sha = self.loss_func['cls'](pred, label)
        
        # 2. classification loss for specific features
        loss_spe = self.loss_func['spe'](pred_spe, label_spe)

        # 3. reconstruction loss
        self_loss_reconstruction_1 = self.loss_func['rec'](fake_img, self_reconstruction_image_1)
        self_loss_reconstruction_2 = self.loss_func['rec'](real_img, self_reconstruction_image_2)
        cross_loss_reconstruction_1 = self.loss_func['rec'](fake_img, reconstruction_image_2)
        cross_loss_reconstruction_2 = self.loss_func['rec'](real_img, reconstruction_image_1)
        loss_reconstruction = \
            self_loss_reconstruction_1 + self_loss_reconstruction_2 + \
            cross_loss_reconstruction_1 + cross_loss_reconstruction_2

        # 4. constrative loss
        common_features = pred_dict['feat']
        specific_features = pred_dict['feat_spe']
        loss_con = self.loss_func['con'](common_features, specific_features, label_spe)

        # 5. total loss
        loss = loss_sha + 0.1*loss_spe + 0.3*loss_reconstruction + 0.05*loss_con
        loss_dict = {
            'overall': loss,
            'common': loss_sha,
            'specific': loss_spe,
            'reconstruction': loss_reconstruction,
            'contrastive': loss_con,
        }
        return loss_dict

    def get_test_losses(self, data_dict: dict, pred_dict: dict) -> dict:
        label = data_dict['label']
        pred = pred_dict['cls']
        loss = self.loss_func['cls'](pred, label)
        loss_dict = {'common': loss}
        return loss_dict

    def get_train_metrics(self, data_dict: dict, pred_dict: dict) -> dict:
        def get_accracy(label, output):
            _, prediction = torch.max(output, 1)   
            correct = (prediction == label).sum().item()
            accuracy = correct / prediction.size(0)
            return accuracy

        label = data_dict['label']
        pred = pred_dict['cls']
        label_spe = data_dict['label_spe']
        pred_spe = pred_dict['cls_spe']

        auc, eer, acc, ap = calculate_metrics_for_train(label.detach(), pred.detach())
        acc_spe = get_accracy(label_spe.detach(), pred_spe.detach())
        metric_batch_dict = {'acc': acc, 'acc_spe': acc_spe, 'auc': auc, 'eer': eer, 'ap': ap}
        return metric_batch_dict
    

    def forward(self, data_dict: dict, inference=False) -> dict:
        original_images = denormalize(data_dict['image'], mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        original_images = torch.clamp(original_images, 0, 1)

        features = self.features(data_dict)
        forgery_features, content_features = features['forgery'], features['content']
        f_spe, f_share = self.classifier(forgery_features)

        bs = f_share.size(0) 
        aug_idx = random.random()

        if aug_idx < 0.7:
            idx_list = list(range(0, bs // 2))
            random.shuffle(idx_list)
            f_share[0: bs // 2] = f_share[idx_list]

            idx_list_fake = list(range(bs // 2, bs))
            random.shuffle(idx_list_fake)
            f_share[bs // 2: bs] = f_share[idx_list_fake]

            original_indices = idx_list + idx_list_fake  
        else:
            original_indices = list(range(bs))  

        features= self.features(data_dict)
        forgery_features, content_features = features['forgery'], features['content']
        f_spe, f_share = self.classifier(forgery_features)

        if inference:
            out_sha, sha_feat = self.head_sha(f_share)
            out_spe, spe_feat = self.head_spe(f_spe)
            prob_sha = torch.softmax(out_sha, dim=1)[:, 1]
            self.prob.append(
                prob_sha
                .detach()
                .squeeze()
                .cpu()
                .numpy()
            )
            self.label.append(
                data_dict['label']
                .detach()
                .squeeze()
                .cpu()
                .numpy()
            )
            _, prediction_class = torch.max(out_sha, 1)
            common_label = (data_dict['label'] >= 1)
            correct = (prediction_class == common_label).sum().item()
            self.correct += correct
            self.total += data_dict['label'].size(0)

            pred_dict = {'cls': out_sha, 'prob': prob_sha, 'feat': sha_feat}
            return  pred_dict

        bs = f_share.size(0)
        aug_idx = random.random()
        if aug_idx < 0.7:
            idx_list = list(range(0, bs//2))
            random.shuffle(idx_list)
            f_share[0: bs//2] = f_share[idx_list]
            idx_list = list(range(bs//2, bs))
            random.shuffle(idx_list)
            f_share[bs//2: bs] = f_share[idx_list]
        
        f_all = torch.cat((f_spe, f_share), dim=1)
        
        f2, f1 = f_all.chunk(2, dim=0)
        c2, c1 = content_features.chunk(2, dim=0)

        self_reconstruction_image_1 = self.con_gan(f1, c1)

        self_reconstruction_image_2 = self.con_gan(f2, c2)

        reconstruction_image_1 = self.con_gan(f1, c2)
        reconstruction_image_2 = self.con_gan(f2, c1)

        out_spe, spe_feat = self.head_spe(f_spe)
        out_sha, sha_feat = self.head_sha(f_share)

        prob_sha = torch.softmax(out_sha, dim=1)[:, 1]
        prob_spe = torch.softmax(out_spe, dim=1)[:, 1]

        pred_dict = {
            'cls': out_sha, 
            'prob': prob_sha, 
            'feat': sha_feat,
            'cls_spe': out_spe,
            'prob_spe': prob_spe,
            'feat_spe': spe_feat,
            'feat_content': content_features,
            'recontruction_imgs': (
                reconstruction_image_1,
                reconstruction_image_2,
                self_reconstruction_image_1, 
                self_reconstruction_image_2
            )
        }
        return pred_dict

def sn_double_conv(in_channels, out_channels):
    return nn.Sequential(
        nn.utils.spectral_norm(
            nn.Conv2d(in_channels, in_channels, 3, padding=1)),
        nn.utils.spectral_norm(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, stride=2)),
        nn.LeakyReLU(0.2, inplace=True)
    )

def r_double_conv(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, 3, padding=1),
        nn.ReLU(inplace=True)
    )

class AdaIN(nn.Module):
    def __init__(self, eps=1e-5):
        super().__init__()
        self.eps = eps

    def c_norm(self, x, bs, ch, eps=1e-7):
        x_var = x.var(dim=-1) + eps
        x_std = x_var.sqrt().view(bs, ch, 1, 1)
        x_mean = x.mean(dim=-1).view(bs, ch, 1, 1)
        return x_std, x_mean

    def forward(self, x, y1):
        assert x.size(0)==y1.size(0)
        size = x.size()
        bs, ch = size[:2]
        x_ = x.view(bs, ch, -1)
        y_ = y1.reshape(bs, ch, -1)
        x_std, x_mean = self.c_norm(x_, bs, ch, eps=self.eps)
        y_std, y_mean = self.c_norm(y_, bs, ch, eps=self.eps)
        out =   ((x - x_mean.expand(size)) / x_std.expand(size)) \
                * y_std.expand(size) + y_mean.expand(size)
        return out

class Conditional_UNet(nn.Module):

    def init_weight(self, std=0.2):
        for m in self.modules():
            cn = m.__class__.__name__
            if cn.find('Conv') != -1:
                m.weight.data.normal_(0., std)
            elif cn.find('Linear') != -1:
                m.weight.data.normal_(1., std)
                m.bias.data.fill_(0)

    def __init__(self):
        super(Conditional_UNet, self).__init__()

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.maxpool = nn.MaxPool2d(2)
        self.dropout = nn.Dropout(p=0.3)
        
        self.adain3 = AdaIN()
        self.adain2 = AdaIN()
        self.adain1 = AdaIN()

        self.dconv_up3 = r_double_conv(512, 256)
        self.dconv_up2 = r_double_conv(256, 128)
        self.dconv_up1 = r_double_conv(128, 64)
        
        self.conv_last = nn.Conv2d(64, 3, 1)
        self.up_last = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.activation = nn.Tanh()
        
    def forward(self, c, x):
        x = self.adain3(x, c)
        x = self.upsample(x)
        x = self.dropout(x)
        x = self.dconv_up3(x)
        c = self.upsample(c)
        c = self.dropout(c)
        c = self.dconv_up3(c)

        x = self.adain2(x, c)
        x = self.upsample(x)   
        x = self.dropout(x)     
        x = self.dconv_up2(x)
        c = self.upsample(c)   
        c = self.dropout(c)     
        c = self.dconv_up2(c)

        x = self.adain1(x, c)
        x = self.upsample(x)   
        x = self.dropout(x)
        x = self.dconv_up1(x)
        
        x = self.conv_last(x)
        out = self.up_last(x)
        
        return self.activation(out)

class MLP(nn.Module):
    def __init__(self, in_f, hidden_dim, out_f):
        super(MLP, self).__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.mlp = nn.Sequential(nn.Linear(in_f, hidden_dim),
                                nn.LeakyReLU(inplace=True),
                                nn.Linear(hidden_dim, hidden_dim),
                                nn.LeakyReLU(inplace=True),
                                nn.Linear(hidden_dim, out_f),)

    def forward(self, x):
        x = self.pool(x)
        x = self.mlp(x)
        return x

class Conv2d1x1(nn.Module):
    def __init__(self, in_f, hidden_dim, out_f):
        super(Conv2d1x1, self).__init__()
        self.conv2d = nn.Sequential(nn.Conv2d(in_f, hidden_dim, 1, 1),
                                nn.LeakyReLU(inplace=True),
                                nn.Conv2d(hidden_dim, hidden_dim, 1, 1),
                                nn.LeakyReLU(inplace=True),
                                nn.Conv2d(hidden_dim, out_f, 1, 1),)

    def forward(self, x):
        x = self.conv2d(x)
        return x

class Head(nn.Module):
    def __init__(self, in_f, hidden_dim, out_f):
        super(Head, self).__init__()
        self.do = nn.Dropout(0.2)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.mlp = nn.Sequential(nn.Linear(in_f, hidden_dim),
                                nn.LeakyReLU(inplace=True),
                                nn.Linear(hidden_dim, out_f),)

    def forward(self, x):
        bs = x.size()[0]
        x_feat = self.pool(x).view(bs, -1)
        x = self.mlp(x_feat)
        x = self.do(x)
        return x, x_feat

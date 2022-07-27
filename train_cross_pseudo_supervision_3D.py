import argparse
import logging
import os
import random
import shutil
import sys
import time
import wandb

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.nn import BCEWithLogitsLoss
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import make_grid
from tqdm import tqdm
import yaml

from dataset import utils
from dataset.BCVData import BCVDataset
from dataset.brats2019 import (BraTS2019, CenterCrop, RandomCrop,
                                   RandomRotFlip, ToTensor,
                                   TwoStreamBatchSampler)
from networks.net_factory_3d import net_factory_3d
from utils import losses, metrics, ramps
from utils.util import save_config
from val_3D import test_all_case,test_all_case_BCV

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str,
                    default='../data/BraTS2019', help='Name of Experiment')
parser.add_argument('--dataset', type=str,
                    default='BCV', help='use which dataset')
parser.add_argument('--exp', type=str,
                    default='MMWHS_CPS_patch160', help='experiment_name')
parser.add_argument('--model', type=str,
                    default='unet_3D', help='model_name')
parser.add_argument('--max_iterations', type=int,
                    default=30000, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=2,
                    help='batch_size per gpu')
parser.add_argument('--deterministic', type=int, default=1,
                    help='whether use deterministic training')
parser.add_argument('--base_lr', type=float, default=0.01,
                    help='segmentation network learning rate')
parser.add_argument('--patch_size', type=list, default=[96, 160, 160],
                    help='patch size of network input')
parser.add_argument('--seed', type=int, default=1337, help='random seed')
parser.add_argument('--consistency_rampup', type=float,
                    default=400.0, help='consistency_rampup')
# label and unlabel
parser.add_argument('--labeled_bs', type=int, default=1,
                    help='labeled_batch_size per gpu')
parser.add_argument('--labeled_num', type=int, default=10,
                    help='labeled data')

# costs
parser.add_argument('--ema_decay', type=float, default=0.99, help='ema_decay')
parser.add_argument('--consistency_type', type=str,
                    default="mse", help='consistency_type')
parser.add_argument('--consistency', type=float,
                    default=0.1, help='consistency')
parser.add_argument('--began_semi_iter', type=int, default=8000, help='iteration to began semi loss')
parser.add_argument('--began_eval_iter', type=int, default=2000, help='iteration to began evaluation')
parser.add_argument('--gpu', default='3', type=str, help='id(s) for CUDA_VISIBLE_DEVICES')
parser.add_argument('--config', type=str,
                    default='train_config.yaml', help='model_name')
args = parser.parse_args()


def get_current_consistency_weight(epoch):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return args.consistency * ramps.sigmoid_rampup(epoch, args.consistency_rampup)


def update_ema_variables(model, ema_model, alpha, global_step):
    # Use the true average until the exponential average is more correct
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(1 - alpha, param.data)


def kaiming_normal_init_weight(model):
    for m in model.modules():
        if isinstance(m, nn.Conv3d):
            torch.nn.init.kaiming_normal_(m.weight)
        elif isinstance(m, nn.BatchNorm3d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
    return model


def xavier_normal_init_weight(model):
    for m in model.modules():
        if isinstance(m, nn.Conv3d):
            torch.nn.init.xavier_normal_(m.weight)
        elif isinstance(m, nn.BatchNorm3d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
    return model


def train_cross_pseudo_3d(args, snapshot_path, db_train=None):
    exp = config['exp']
    base_lr = config['base_lr']
    train_data_path = config['root_path']
    base_lr = config['base_lr']
    backbone = config['backbone']
    train_data_path = config['root_path']
    max_iterations = config['max_iterations']
    began_semi_iter = config['began_semi_iter']
    ema_decay = config['ema_decay']
    began_condition_iter = config['began_condition_iter']
    began_eval_iter = config['began_eval_iter']
    
    # config for dataset
    dataset = config['DATASET']
    patch_size = dataset['patch_size']
    labeled_num = dataset['labeled_num']
    
    batch_size = dataset['batch_size']
    labeled_bs = dataset['labeled_bs']
    dataset_name = dataset['name']
    dataset_config = dataset[dataset_name]
    num_classes = dataset_config['num_classes']
    training_data_num = dataset_config['training_data_num']
    train_list = dataset_config['train_list']
    test_list = dataset_config['test_list']

    # config for method
    method_name = config['METHOD']['name']
    method_config = config['METHOD'][method_name]


    net1 = net_factory_3d(net_type=backbone, in_chns=1, class_num=num_classes).cuda()
    net2 = net_factory_3d(net_type=backbone, in_chns=1, class_num=num_classes).cuda()
    model1 = kaiming_normal_init_weight(net1)
    model2 = xavier_normal_init_weight(net2)
    model1.train()
    model2.train()
    if dataset_name in ['BCV','MMWHS']:
        db_train = BCVDataset(img_list_file=train_list,
                                patch_size=patch_size,
                                labeled_num=labeled_num,
                                transforms=transforms.Compose([
                                RandomRotFlip(),
                                RandomCrop(patch_size),
                                ToTensor(),
                            ]))
    else:
        db_train = BraTS2019(base_dir=train_data_path,
                            split='train',
                            num=None,
                            transform=transforms.Compose([
                                RandomRotFlip(),
                                RandomCrop(patch_size),
                                ToTensor(),
                            ]))

    def worker_init_fn(worker_id):
        random.seed(config['seed'] + worker_id)

    labeled_idxs = list(range(0, labeled_num))
    unlabeled_idxs = list(range(labeled_num, training_data_num))
    batch_sampler = TwoStreamBatchSampler(
        labeled_idxs, unlabeled_idxs, batch_size, batch_size - labeled_bs)

    trainloader = DataLoader(db_train, batch_sampler=batch_sampler,
                             num_workers=4, pin_memory=True, 
                             worker_init_fn=worker_init_fn)

    optimizer1 = optim.SGD(model1.parameters(), lr=base_lr,
                           momentum=0.9, weight_decay=0.0001)
    optimizer2 = optim.SGD(model2.parameters(), lr=base_lr,
                           momentum=0.9, weight_decay=0.0001)
    best_performance1 = 0.0
    best_performance2 = 0.0
    iter_num = 0
    ce_loss = CrossEntropyLoss()
    dice_loss = losses.DiceLoss(num_classes)
    
    wandb.init(
        project="semi-supervised-segmentation", 
        name=f"{dataset_name}_{method_name}_{backbone}_labeled{labeled_num}_{exp}",
        config=config
    )
    wandb.tensorboard.patch(root_logdir=snapshot_path + '/log')
    writer = SummaryWriter(snapshot_path + '/log')
    logging.info("{} iterations per epoch".format(len(trainloader)))

    max_epoch = max_iterations // len(trainloader) + 1
    iterator = tqdm(range(max_epoch), ncols=70)
    for epoch_num in iterator:
        for i_batch, sampled_batch in enumerate(trainloader):

            volume_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            volume_batch, label_batch = volume_batch.cuda(), label_batch.cuda()

            outputs1 = model1(volume_batch)
            outputs_soft1 = torch.softmax(outputs1, dim=1)

            outputs2 = model2(volume_batch)
            outputs_soft2 = torch.softmax(outputs2, dim=1)
            consistency_weight = get_current_consistency_weight(iter_num // 4)

            loss1 = 0.5 * (ce_loss(outputs1[:labeled_bs],
                                   label_batch[:][:labeled_bs].long()) + 
                                   dice_loss(outputs_soft1[:labeled_bs], 
                                             label_batch[:labeled_bs].unsqueeze(1)))
            loss2 = 0.5 * (ce_loss(outputs2[:labeled_bs],
                                   label_batch[:][:labeled_bs].long()) + 
                                   dice_loss(outputs_soft2[:labeled_bs], 
                                             label_batch[:labeled_bs].unsqueeze(1)))

            pseudo_outputs1 = torch.argmax(outputs_soft1[labeled_bs:].detach(), 
                                           dim=1, keepdim=False)
            pseudo_outputs2 = torch.argmax(outputs_soft2[labeled_bs:].detach(), 
                                           dim=1, keepdim=False)
            if iter_num < began_semi_iter:
                pseudo_supervision1 = torch.FloatTensor([0]).cuda()
                pseudo_supervision2 = torch.FloatTensor([0]).cuda()
            else:
                pseudo_supervision1 = ce_loss(outputs1[labeled_bs:], 
                                              pseudo_outputs2)
                pseudo_supervision2 = ce_loss(outputs2[labeled_bs:], 
                                              pseudo_outputs1)

            model1_loss = loss1 + consistency_weight * pseudo_supervision1
            model2_loss = loss2 + consistency_weight * pseudo_supervision2

            loss = model1_loss + model2_loss

            optimizer1.zero_grad()
            optimizer2.zero_grad()

            loss.backward()

            optimizer1.step()
            optimizer2.step()

            iter_num = iter_num + 1

            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group1 in optimizer1.param_groups:
                param_group1['lr'] = lr_
            for param_group2 in optimizer2.param_groups:
                param_group2['lr'] = lr_

            writer.add_scalar('lr', lr_, iter_num)
            writer.add_scalar('consistency_weight/consistency_weight', 
                              consistency_weight, iter_num)
            writer.add_scalar('loss/model1_loss', model1_loss, iter_num)
            writer.add_scalar('loss/model2_loss', model2_loss, iter_num)
            logging.info(
                'iteration %d : model1 loss : %f model2 loss : %f' % (iter_num, 
                 model1_loss.item(), model2_loss.item()))
            if iter_num % 50 == 0:
                image = volume_batch[0, 0:1, :, :, 20:61:10].permute(
                    3, 0, 1, 2).repeat(1, 3, 1, 1)
                grid_image = make_grid(image, 5, normalize=True)
                writer.add_image('train/Image', grid_image, iter_num)

                image = outputs_soft1[0, 0:1, :, :, 20:61:10].permute(
                    3, 0, 1, 2).repeat(1, 3, 1, 1)
                grid_image = make_grid(image, 5, normalize=False)
                writer.add_image('train/Model1_Predicted_label',
                                 grid_image, iter_num)

                image = outputs_soft2[0, 0:1, :, :, 20:61:10].permute(
                    3, 0, 1, 2).repeat(1, 3, 1, 1)
                grid_image = make_grid(image, 5, normalize=False)
                writer.add_image('train/Model2_Predicted_label',
                                 grid_image, iter_num)

                image = label_batch[0, :, :, 20:61:10].unsqueeze(
                    0).permute(3, 0, 1, 2).repeat(1, 3, 1, 1)
                grid_image = make_grid(image, 5, normalize=False)
                writer.add_image('train/Groundtruth_label',
                                 grid_image, iter_num)

            if iter_num > began_eval_iter and iter_num % 200 == 0:
                model1.eval()
                if  dataset_name in ['BCV','MMWHS']:
                    avg_metric1 = test_all_case_BCV(model1, test_list=test_list,
                                                    num_classes=num_classes, 
                                                    patch_size=patch_size,
                                                    stride_xy=64, stride_z=64)
                else:
                    avg_metric1 = test_all_case(
                        model1, train_data_path, 
                        test_list="val.txt", 
                        num_classes=2, 
                        patch_size=patch_size,
                        stride_xy=64, stride_z=64
                        )
                if avg_metric1[:, 0].mean() > best_performance1:
                    best_performance1 = avg_metric1[:, 0].mean()
                    save_mode_path = os.path.join(
                        snapshot_path,
                        'model1_iter_{}_dice_{}.pth'.format(
                            iter_num, round(best_performance1, 4)
                            )
                    )
                    save_best = os.path.join(
                        snapshot_path,
                        '{}_best_model1.pth'.format(backbone)
                        )
                    torch.save(model1.state_dict(), save_mode_path)
                    torch.save(model1.state_dict(), save_best)

                writer.add_scalar(
                    'info/model1_val_dice_score',
                    avg_metric1[0, 0], iter_num
                    )
                writer.add_scalar(
                    'info/model1_val_hd95',
                    avg_metric1[0, 1], iter_num
                    )
                logging.info(
                    'iteration %d : model1_dice_score : %f model1_hd95 : %f' % (
                        iter_num, avg_metric1[0, 0].mean(), 
                        avg_metric1[0, 1].mean()
                        )
                    )
                model1.train()

                model2.eval()
                if  dataset_name in ['BCV','MMWHS']:
                    avg_metric2 = test_all_case_BCV(
                        model2, test_list=test_list,
                        num_classes=num_classes, patch_size=patch_size,
                        stride_xy=64, stride_z=64
                        )
                else:
                    avg_metric2 = test_all_case(
                        model2, train_data_path, test_list="val.txt", 
                        num_classes=num_classes, patch_size=patch_size,
                        stride_xy=64, stride_z=64
                        )
                if avg_metric2[:, 0].mean() > best_performance2:
                    best_performance2 = avg_metric2[:, 0].mean()
                    save_mode_path = os.path.join(
                        snapshot_path, 'model2_iter_{}_dice_{}.pth'.format(
                            iter_num, round(best_performance2, 4)
                            )
                        )
                    save_best = os.path.join(
                        snapshot_path,
                        '{}_best_model2.pth'.format(backbone)
                        )
                    torch.save(model2.state_dict(), save_mode_path)
                    torch.save(model2.state_dict(), save_best)

                writer.add_scalar('info/model2_val_dice_score',
                                  avg_metric2[0, 0], iter_num)
                writer.add_scalar('info/model2_val_hd95',
                                  avg_metric2[0, 1], iter_num)
                logging.info(
                    'iteration %d : model2_dice_score : %f model2_hd95 : %f' % (
                        iter_num, avg_metric2[0, 0].mean(), 
                        avg_metric2[0, 1].mean()
                        )
                    )
                model2.train()

            if iter_num % 3000 == 0:
                save_mode_path = os.path.join(
                    snapshot_path, 'model1_iter_' + str(iter_num) + '.pth')
                torch.save(model1.state_dict(), save_mode_path)
                logging.info("save model1 to {}".format(save_mode_path))

                save_mode_path = os.path.join(
                    snapshot_path, 'model2_iter_' + str(iter_num) + '.pth')
                torch.save(model2.state_dict(), save_mode_path)
                logging.info("save model2 to {}".format(save_mode_path))

            if iter_num >= max_iterations:
                break
            time1 = time.time()
        if iter_num >= max_iterations:
            iterator.close()
            break
    writer.close()


if __name__ == "__main__":
    args = parser.parse_args()
    config = yaml.safe_load(open(args.config, 'r'))
    save_config(config) # save config to yaml file with timestamp
    os.environ["CUDA_VISIBLE_DEVICES"] = config['gpu']
    if not config['deterministic']:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True

    random.seed(config['seed'])
    np.random.seed(config['seed'])
    torch.manual_seed(config['seed'])
    torch.cuda.manual_seed(config['seed'])

    snapshot_path = "../model/{}_{}_{}_{}/{}".format(
        config['DATASET']['name'], 
        config['DATASET']['labeled_num'], 
        config['method'], 
        config['exp'],
        config['backbone']   
    )
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)
    # move config to snapshot_path
    shutil.copyfile(args.config, snapshot_path+"/"+
                                 time.strftime("%Y-%m-%d=%H:%M:%S", 
                                               time.localtime())+
                                               "_train_config.yaml")
    logging.basicConfig(
        filename=snapshot_path+"/log.txt", 
        level=logging.INFO,
        format='[%(asctime)s.%(msecs)03d] %(message)s', 
        datefmt='%H:%M:%S'
    )
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(config))
    train_cross_pseudo_3d(config, snapshot_path)

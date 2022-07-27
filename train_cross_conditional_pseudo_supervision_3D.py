import argparse
from itertools import count
import logging
import os
import random
import shutil
import sys
import time
import wandb
import yaml

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

from dataset import utils
from dataset.BCVData import BCVDataset,BCVDatasetCAC
from dataset.brats2019 import (BraTS2019, CenterCrop, RandomCrop,
                                   RandomRotFlip, ToTensor,
                                   TwoStreamBatchSampler)
from networks.net_factory_3d import net_factory_3d
from utils import losses, metrics, ramps
from utils.util import save_config
from val_3D import test_all_case,test_all_case_BCV
import torch.cuda.amp as amp

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str,
                    default='train_config.yaml', help='model_name')


def get_current_consistency_weight(epoch):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return config['consistency'] * ramps.sigmoid_rampup(epoch, config['consistency_rampup'])


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

def cross_entropy_loss_con(output, target, condition):
    """
    cross entropy loss for conditional network
    """
    softmax = torch.softmax(output,dim=1)
    B,C,D,H,W = softmax.shape
    softmax_con = torch.zeros(B,2,D,H,W).cuda()
    softmax_con[:,1,...] = softmax[np.arange(B),condition.squeeze(),...] 
    softmax_con[:,0,...] = 1.0 - softmax_con[:,1,...]
    log = -torch.log(softmax_con.gather(1, target.unsqueeze(1)) + 1e-7)
    loss = log.mean()
    return loss

def train_C3PS(config, snapshot_path):
    exp = config['exp']
    backbone = config['backbone']
    method_name = config['method']
    base_lr = config['initial_lr']
    train_data_path = config['root_path']
    max_iterations = config['max_iterations']
    use_CAC = config['use_CAC']
    began_semi_iter = config['began_semi_iter']
    began_condition_iter = config['began_condition_iter']
    began_eval_iter = config['began_eval_iter']
    # config for dataset
    dataset = config['DATASET']
    patch_size = dataset['patch_size']
    labeled_num = dataset['labeled_num']
    batch_size = dataset['batch_size']
    labeled_bs = dataset['labeled_bs']
    dataset_name = config['dataset_name']
    dataset_config = dataset[dataset_name]
    num_classes = dataset_config['num_classes']
    train_list = dataset_config['train_list']
    test_list = dataset_config['test_list']
    training_data_num = dataset_config['training_data_num']
    testing_data_num = dataset_config['testing_data_num']
    cut_upper = dataset_config['cut_upper']
    cut_lower = dataset_config['cut_lower']

    # config for Context-Aware-Consistency
    CAC_config = config['METHOD']['C3PS']
    stride = CAC_config['stride']
    iou_bound_low = CAC_config['iou_bound_low']
    iou_bound_high = CAC_config['iou_bound_high']


    model1 = net_factory_3d(net_type=backbone, in_chns=1, 
                            class_num=num_classes,model_config=config['model'])
    model2 = net_factory_3d("unet_3D_condition", in_chns=1, class_num=2)
    #model1 = kaiming_normal_init_weight(net1)
    if config['continue_training']:
        model1_state_dict = torch.load(config['model1_checkpoint'])
        model2_state_dict = torch.load(config['model2_checkpoint'])
        model1.load_state_dict(model1_state_dict)
        model2.load_state_dict(model2_state_dict)
        print(f"==>load model1 weights from {config['model1_checkpoint']}")
        print(f"==>load model2 weights from {config['model2_checkpoint']}")
        current_iter_num = config['current_iter_num']
    else:
        model1 = xavier_normal_init_weight(model1)
        model2 = xavier_normal_init_weight(model2)
        print("==>use xavier normal init weights")
        current_iter_num = 0
    if not use_CAC:
        model1=nn.DataParallel(model1,device_ids=[0,1])
        model2=nn.DataParallel(model2,device_ids=[0,1])
    model1.train()
    model2.train()

    if dataset_name in ['BCV','MMWHS','FLARE']:
        if use_CAC:
            print("======> use CAC ")
            db_train = BCVDatasetCAC(
                img_list_file=train_list, 
                patch_size=patch_size, 
                num_class=num_classes, 
                stride=stride, 
                iou_bound=[iou_bound_low,iou_bound_high],
                labeled_num=labeled_num,
                cutout=True,
                affine_trans=False,
                upper=cut_upper,
                lower=cut_lower
            )
        else:
            print("======> not use CAC ")
            db_train = BCVDataset(
                img_list_file=train_list,
                patch_size=patch_size,
                transforms=transforms.Compose([
                    RandomRotFlip(),
                    RandomCrop(patch_size),
                    ToTensor(),
                ])
            )
    else:
        db_train = BraTS2019(
            base_dir=train_data_path,
            split='train',
            num=None,
            transform=transforms.Compose([
                RandomRotFlip(),
                RandomCrop(patch_size),
                ToTensor(),
            ])
        )

    def worker_init_fn(worker_id):
        random.seed(config['seed'] + worker_id)

    labeled_idxs = list(range(0, labeled_num))
    unlabeled_idxs = list(range(labeled_num, training_data_num))
    batch_sampler = TwoStreamBatchSampler(
        labeled_idxs, 
        unlabeled_idxs, 
        batch_size, 
        batch_size - labeled_bs
    )

    trainloader = DataLoader(
        db_train, 
        batch_sampler=batch_sampler,
        num_workers=4, 
        pin_memory=True, 
        worker_init_fn=worker_init_fn
    )

    optimizer1 = optim.SGD(model1.parameters(), lr=base_lr,
                            momentum=0.9, weight_decay=0.00001)
    optimizer2 = optim.SGD(model2.parameters(), lr=base_lr,
                            momentum=0.9, weight_decay=0.00001)
    # optimizer1 = torch.optim.Adam(model1.parameters(), lr=1e-3, 
    #                               weight_decay=1e-5)
    # optimizer2 = torch.optim.Adam(model2.parameters(), lr=1e-3, 
    #                               weight_decay=1e-5)
    scaler1 = amp.GradScaler()
    scaler2 = amp.GradScaler()
    
    best_performance1 = 0.0
    best_performance2 = 0.0
    iter_num = current_iter_num
    ce_loss = CrossEntropyLoss()
    dice_loss = losses.DiceLoss(num_classes)
    dice_loss_con = losses.DiceLoss(2) # dice loss for condition network

    experiment_name = f"{dataset_name}_{method_name}_"\
                               f"{backbone}_labeled{labeled_num}_"\
                               f"{exp}"
    wandb.init(name=experiment_name, project="semi-supervised-segmentation",
               config = config)
    wandb.tensorboard.patch(root_logdir=snapshot_path + '/log')
    writer = SummaryWriter(snapshot_path + '/log')
    logging.info(f"{len(trainloader)} iterations per epoch")

    max_epoch = max_iterations // len(trainloader) + 1
    iter_each_epoch = len(trainloader)
    iterator = tqdm(range(max_epoch), ncols=70)
    condition_cnt = [0] * num_classes
    for epoch_num in iterator:
        for i_batch, sampled_batch in enumerate(trainloader):

            volume_batch, label_batch = (
                sampled_batch['image'], 
                sampled_batch['label']
            )
            volume_batch, label_batch = volume_batch.cuda(), label_batch.cuda()
            label_batch[labeled_bs:] = -1 # for test CAC effectiveness
            ul1,br1,ul2,br2 = [],[],[],[]
            labeled_idxs_batch = torch.arange(0,labeled_bs)
            unlabeled_idxs_batch = torch.arange(labeled_bs,batch_size)
            if use_CAC:
                ul1,br1 = sampled_batch['ul1'],sampled_batch['br1']
                ul2,br2 = sampled_batch['ul2'],sampled_batch['br2']
                volume_batch = torch.cat(
                    [volume_batch[:,0,...],volume_batch[:,1,...]],
                    dim=0
                )
                label_batch = torch.cat(
                    [label_batch[:,0,...],label_batch[:,1,...]],
                    dim=0
                )
                labeled_idxs2_batch = torch.arange(
                    batch_size,
                    batch_size+labeled_bs
                )
                labeled_idxs1_batch = torch.arange(0,labeled_bs)
                labeled_idxs_batch = torch.cat(
                    [labeled_idxs1_batch,labeled_idxs2_batch]
                )
                unlabeled_idxs1_batch = torch.arange(labeled_bs,batch_size)
                unlabeled_idxs2_batch = torch.arange(
                    batch_size+labeled_bs, 
                    2 * batch_size
                )
                unlabeled_idxs_batch = torch.cat(
                    [unlabeled_idxs1_batch,unlabeled_idxs2_batch]
                )
            with amp.autocast():
                outputs1 = model1(volume_batch)
                outputs_soft1 = torch.softmax(outputs1, dim=1)

                #prepare input for condition net
                condition_batch = sampled_batch['condition'].cuda()
                if config['use_CAC']:
                    condition_batch = torch.cat(
                        [condition_batch,condition_batch],
                        dim=0
                    )
                # for con in condition_batch:
                #     condition_cnt[con]+=1
                # print("condition count:",condition_cnt)

                outputs2 = model2(volume_batch, condition_batch)
                outputs_soft2 = torch.softmax(outputs2, dim=1)
                label_batch_con = (
                    label_batch==condition_batch.unsqueeze(-1).unsqueeze(-1)
                ).long()

                consistency_weight = get_current_consistency_weight(
                    (iter_num- began_semi_iter) // iter_each_epoch
                )

                loss1 = 0.5 * (
                    ce_loss(
                        outputs1[labeled_idxs_batch],
                        label_batch[labeled_idxs_batch].long()
                    ) + 
                    dice_loss(
                        outputs_soft1[labeled_idxs_batch], 
                        label_batch[labeled_idxs_batch].unsqueeze(1)
                    )
                )
                loss2 = 0.5 * (
                    ce_loss(
                        outputs2[labeled_idxs_batch],
                        label_batch_con[labeled_idxs_batch].long()
                    ) + 
                    dice_loss_con(
                        outputs_soft2[labeled_idxs_batch], 
                        label_batch_con[labeled_idxs_batch].unsqueeze(1)
                    )
                )


                if use_CAC:
                    overlap_soft1_list = []
                    overlap_soft2_list = []
                    overlap_outputs1_list = []
                    overlap_outputs2_list = []
                    for unlabeled_idx1,unlabeled_idx2 in zip(
                        unlabeled_idxs1_batch, 
                        unlabeled_idxs2_batch
                    ):
                        # overlap region pred soft by model1
                        overlap1_soft1 = outputs_soft1[
                            unlabeled_idx1,
                            :,
                            ul1[0][1]:br1[0][1],
                            ul1[1][1]:br1[1][1],
                            ul1[2][1]:br1[2][1]
                        ]
                        overlap2_soft1 = outputs_soft1[
                            unlabeled_idx2,
                            :,
                            ul2[0][1]:br2[0][1],
                            ul2[1][1]:br2[1][1],
                            ul2[2][1]:br2[2][1]
                        ]
                        assert overlap1_soft1.shape == overlap2_soft1.shape, (
                            "overlap  region size must equal"
                        )
                        
                        # overlap region pred by model1
                        overlap1_outputs1 = outputs1[
                            unlabeled_idx1,
                            :,
                            ul1[0][1]:br1[0][1],
                            ul1[1][1]:br1[1][1],
                            ul1[2][1]:br1[2][1]
                        ]
                        overlap2_outputs1 = outputs1[
                            unlabeled_idx2,
                            :,
                            ul2[0][1]:br2[0][1],
                            ul2[1][1]:br2[1][1],
                            ul2[2][1]:br2[2][1]
                        ]
                        assert overlap1_outputs1.shape == overlap2_outputs1.shape, (
                            "overlap  region size must equal"
                        )
                        overlap_outputs1_list.append(overlap1_outputs1.unsqueeze(0))
                        overlap_outputs1_list.append(overlap2_outputs1.unsqueeze(0))

                        # overlap region pred by model2
                        overlap1_soft2 = outputs_soft2[
                            unlabeled_idx1,
                            :,
                            ul1[0][1]:br1[0][1],
                            ul1[1][1]:br1[1][1],
                            ul1[2][1]:br1[2][1]
                        ]
                        overlap2_soft2 = outputs_soft2[
                            unlabeled_idx2,
                            :,
                            ul2[0][1]:br2[0][1],
                            ul2[1][1]:br2[1][1],
                            ul2[2][1]:br2[2][1]
                        ]
                        assert overlap1_soft2.shape == overlap2_soft2.shape, (
                            "overlap  region size must equal"
                        )
                        
                        # overlap region pred by model2
                        overlap1_outputs2 = outputs2[
                            unlabeled_idx1,
                            :,
                            ul1[0][1]:br1[0][1],
                            ul1[1][1]:br1[1][1],
                            ul1[2][1]:br1[2][1]
                        ]
                        overlap2_outputs2 = outputs2[
                            unlabeled_idx2,
                            :,
                            ul2[0][1]:br2[0][1],
                            ul2[1][1]:br2[1][1],
                            ul2[2][1]:br2[2][1]
                        ]
                        assert overlap1_outputs2.shape == overlap2_outputs2.shape, (
                            "overlap  region size must equal"
                        )
                        overlap_outputs2_list.append(overlap1_outputs2.unsqueeze(0))
                        overlap_outputs2_list.append(overlap2_outputs2.unsqueeze(0))

                        #merge overlap region pred
                        overlap_soft1_tmp = (overlap1_soft1 + overlap2_soft1) / 2.
                        overlap_soft2_tmp = (overlap1_soft2 + overlap2_soft2) / 2.
                        overlap_soft1_list.append(overlap_soft1_tmp.unsqueeze(0))
                        overlap_soft2_list.append(overlap_soft2_tmp.unsqueeze(0))
                    overlap_soft1 = torch.cat(overlap_soft1_list, 0)
                    overlap_soft2 = torch.cat(overlap_soft2_list, 0)
                    overlap_outputs1 = torch.cat(overlap_outputs1_list, 0)
                    overlap_outputs2 = torch.cat(overlap_outputs2_list, 0)
                
                if iter_num < began_condition_iter:
                    pseudo_supervision1 = torch.FloatTensor([0]).cuda()
                else:
                    if use_CAC:
                        overlap_pseudo_outputs2 = torch.argmax(
                            overlap_soft2.detach(), 
                            dim=1, 
                            keepdim=False
                        )
                        overlap_pseudo_outputs2 = torch.cat(
                            [overlap_pseudo_outputs2, overlap_pseudo_outputs2]
                        )
                        pseudo_supervision1 = cross_entropy_loss_con(
                            overlap_outputs1, 
                            overlap_pseudo_outputs2, 
                            condition_batch[unlabeled_idxs_batch]
                        )
                    else:
                        pseudo_outputs2 = torch.argmax(
                            outputs_soft2[labeled_bs:].detach(), 
                            dim=1, 
                            keepdim=False
                        )
                        pseudo_supervision1 = cross_entropy_loss_con(
                            outputs1[labeled_bs:], 
                            pseudo_outputs2, 
                            condition_batch[labeled_bs:]
                        )
                
                if iter_num < began_semi_iter:
                    pseudo_supervision2 = torch.FloatTensor([0]).cuda()
                else:
                    if use_CAC:
                        overlap_pseudo_outputs1 = torch.argmax(
                            overlap_soft1.detach(), 
                            dim=1, 
                            keepdim=False
                        )
                        overlap_pseudo_outputs1 = torch.cat(
                            [overlap_pseudo_outputs1, overlap_pseudo_outputs1]
                        )
                        pseudo_supervision2 = ce_loss(
                            overlap_outputs2, 
                            (
                                overlap_pseudo_outputs1==condition_batch[unlabeled_idxs_batch].unsqueeze(-1).unsqueeze(-1)
                            ).long()
                        ) 
                    else:
                        pseudo_outputs1 = torch.argmax(
                            outputs_soft1[labeled_bs:].detach(), 
                            dim=1, 
                            keepdim=False
                        )
                        pseudo_supervision2 = ce_loss(
                            outputs2[labeled_bs:], 
                            (pseudo_outputs1==condition_batch[labeled_bs:].unsqueeze(-1).unsqueeze(-1)).long()
                        )


                model1_loss = loss1 + consistency_weight * pseudo_supervision1
                model2_loss = loss2 + consistency_weight * pseudo_supervision2

                loss = model1_loss + model2_loss

            optimizer1.zero_grad()
            optimizer2.zero_grad()
            scaler1.scale(model1_loss).backward()
            scaler2.scale(model2_loss).backward()
            scaler1.step(optimizer1)
            scaler2.step(optimizer2)
            scaler1.update()
            scaler2.update()

            #loss.backward()

            # optimizer1.step()
            # optimizer2.step()

            iter_num = iter_num + 1

            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group1 in optimizer1.param_groups:
                param_group1['lr'] = lr_
            for param_group2 in optimizer2.param_groups:
                param_group2['lr'] = lr_

            writer.add_scalar('lr', lr_, iter_num)
            writer.add_scalar(
                'consistency_weight/consistency_weight', 
                consistency_weight, 
                iter_num
            )
            writer.add_scalar('loss/model1_loss', model1_loss, iter_num)
            writer.add_scalar('loss/model2_loss', model2_loss, iter_num)
            writer.add_scalar(
                'loss/pseudo_supervision1',
                pseudo_supervision1, 
                iter_num
            )
            writer.add_scalar(
                'loss/pseudo_supervision2',
                pseudo_supervision2, 
                iter_num
            )
            logging.info(
                'iteration %d :'
                'model1 loss : %f' 
                'model2 loss : %f' 
                'pseudo_supervision1 : %f'
                'pseudo_supervision2 : %f' % (
                    iter_num, model1_loss.item(), 
                    model2_loss.item(), 
                    pseudo_supervision1.item(), 
                    pseudo_supervision2.item()
                )
            )
            if iter_num % 100 == 0:
                image = volume_batch[0, 0:1, :, :, 20:61:10].permute(
                    3, 0, 1, 2).repeat(1, 3, 1, 1)
                grid_image = make_grid(image, 5, normalize=True)
                writer.add_image('train/Image', grid_image, iter_num)

                image = outputs_soft1[0, 0:1, :, :, 20:61:10].permute(
                    3, 0, 1, 2).repeat(1, 3, 1, 1)
                grid_image = make_grid(image, 5, normalize=False)
                writer.add_image(
                    'train/Model1_Predicted_label',
                    grid_image, 
                    iter_num
                )

                image = outputs_soft2[0, 0:1, :, :, 20:61:10].permute(
                    3, 0, 1, 2).repeat(1, 3, 1, 1)
                grid_image = make_grid(image, 5, normalize=False)
                writer.add_image(
                    'train/Model2_Predicted_label',
                    grid_image, 
                    iter_num
                )

                image = label_batch[0, :, :, 20:61:10].unsqueeze(0).permute(
                    3, 0, 1, 2).repeat(1, 3, 1, 1)
                grid_image = make_grid(image, 5, normalize=False)
                writer.add_image('train/Groundtruth_label',grid_image, iter_num)

            if iter_num > began_eval_iter and iter_num % 200 == 0:
                model1.eval()
                if dataset_name in ['BCV','MMWHS','FLARE']:
                    avg_metric1 = test_all_case_BCV(
                        model1,
                        test_list=test_list,
                        num_classes=num_classes, 
                        patch_size=patch_size,
                        stride_xy=64, 
                        stride_z=64
                    )
                else:
                    avg_metric1 = test_all_case(
                        model1, 
                        train_data_path, 
                        test_list=test_list, 
                        num_classes=num_classes, 
                        patch_size=patch_size,
                        stride_xy=64, 
                        stride_z=64
                    )
                if avg_metric1[:, 0].mean() > best_performance1:
                    best_performance1 = avg_metric1[:, 0].mean()
                    save_mode_path = os.path.join(
                        snapshot_path,
                        f'model1_iter_{iter_num}_dice_{round(best_performance1, 4)}.pth'
                    )
                    save_best = os.path.join(
                        snapshot_path,
                        '{}_best_model1.pth'.format(config['backbone'])
                    )
                    torch.save(model1.state_dict(), save_mode_path)
                    torch.save(model1.state_dict(), save_best)

                writer.add_scalar(
                    'info/model1_val_dice_score',
                    avg_metric1[:, 0].mean(), 
                    iter_num
                )
                writer.add_scalar(
                    'info/model1_val_hd95',
                    avg_metric1[:, 1].mean(), 
                    iter_num
                )
                logging.info(
                    'iteration %d : model1_dice_score : %f model1_hd95 : %f' % (
                        iter_num, 
                        avg_metric1[:, 0].mean(), 
                        avg_metric1[:, 1].mean()
                    )
                )
                model1.train()

                model2.eval()
                test_num_con = testing_data_num // 2
                if iter_num  % 1000 ==0:
                    test_num_con = testing_data_num
                if dataset_name in ['BCV','MMWHS','FLARE']:
                    avg_metric2 = test_all_case_BCV(
                        model2,
                        test_list=test_list,
                        num_classes=num_classes, 
                        patch_size=patch_size,
                        stride_xy=64, 
                        stride_z=64,
                        do_condition=True,
                        test_num=test_num_con,
                    )
                else:
                    avg_metric2 = test_all_case(
                        model2, 
                        train_data_path, 
                        test_list="val.txt", 
                        num_classes=num_classes, 
                        patch_size=patch_size,
                        stride_xy=64, 
                        stride_z=64
                    )
                if avg_metric2[:, 0].mean() > best_performance2:
                    best_performance2 = avg_metric2[:, 0].mean()
                    save_mode_path = os.path.join(
                        snapshot_path,
                        'model2_iter_{}_dice_{}.pth'.format(
                            iter_num, 
                            round(best_performance2, 4)
                        )
                    )
                    save_best = os.path.join(
                        snapshot_path,
                        '{}_best_model2.pth'.format(config['backbone'])
                    )
                    torch.save(model2.state_dict(), save_mode_path)
                    torch.save(model2.state_dict(), save_best)

                writer.add_scalar(
                    'info/model2_val_dice_score',
                    avg_metric2[:, 0].mean(), 
                    iter_num
                )
                writer.add_scalar(
                    'info/model2_val_hd95',
                    avg_metric2[:, 1].mean(), 
                    iter_num
                )
                logging.info(
                    'iteration %d : model2_dice_score : %f model2_hd95 : %f' % (
                        iter_num, 
                        avg_metric2[:, 0].mean(), 
                        avg_metric2[:, 1].mean()
                    )
                )
                model2.train()

            if iter_num % 3000 == 0:
                save_mode_path = os.path.join(
                    snapshot_path, 
                    'model1_iter_' + str(iter_num) + '.pth'
                )
                torch.save(model1.state_dict(), save_mode_path)
                logging.info("save model1 to {}".format(save_mode_path))

                save_mode_path = os.path.join(
                    snapshot_path, 
                    'model2_iter_' + str(iter_num) + '.pth'
                )
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
        config['dataset_name'], 
        config['DATASET']['labeled_num'], 
        config['method'], 
        config['exp'],
        config['backbone']   
    )
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)
    shutil.copyfile(args.config, snapshot_path+"/train_config.yaml")
    logging.basicConfig(
        filename=snapshot_path+"/log.txt", 
        level=logging.INFO,
        format='[%(asctime)s.%(msecs)03d] %(message)s', 
        datefmt='%H:%M:%S'
    )
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(config))
    train_C3PS(config, snapshot_path)

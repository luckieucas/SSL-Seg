import argparse
from itertools import count
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

from dataset import utils
from dataset.BCVData import BCVDataset,BCVDatasetCAC
from dataset.brats2019 import (BraTS2019, CenterCrop, RandomCrop,
                                   RandomRotFlip, ToTensor,
                                   TwoStreamBatchSampler)
from networks.net_factory_3d import net_factory_3d
from utils import losses, metrics, ramps
from val_3D import test_all_case,test_all_case_BCV

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str,
                    default='../data/BraTS2019', help='Name of Experiment')
parser.add_argument('--dataset', type=str,
                    default='BCV', help='use which dataset')
parser.add_argument('--exp', type=str,
                    default='CCPS_refine_warmup_SGD_batch4_CAC', help='experiment_name')
parser.add_argument('--model', type=str,
                    default='unet_3D', help='model_name')
parser.add_argument('--max_iterations', type=int,
                    default=30000, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=8,
                    help='batch_size per gpu')
parser.add_argument('--deterministic', type=int, default=1,
                    help='whether use deterministic training')
parser.add_argument('--base_lr', type=float, default=0.01,
                    help='segmentation network learning rate')
parser.add_argument('--patch_size', type=list, default=[96, 160, 160],
                    help='patch size of network input')
parser.add_argument('--seed', type=int, default=1337, help='random seed')

# label and unlabel
parser.add_argument('--labeled_bs', type=int, default=4,
                    help='labeled_batch_size per gpu')
parser.add_argument('--labeled_num', type=int, default=8,
                    help='labeled data')

# costs
parser.add_argument('--ema_decay', type=float, default=0.99, help='ema_decay')
parser.add_argument('--consistency_type', type=str,
                    default="mse", help='consistency_type')
parser.add_argument('--consistency', type=float,
                    default=0.1, help='consistency')
parser.add_argument('--consistency_rampup', type=float,
                    default=200.0, help='consistency_rampup')
parser.add_argument('--began_semi_iter', type=int, default=4000, help='iteration to began semi loss')
parser.add_argument('--began_condition_iter', type=int, default=8000, help='iteration to began condition loss')
parser.add_argument('--began_eval_iter', type=int, default=1000, help='iteration to began evaluation')
parser.add_argument('--gpu', default='0,1', type=str, help='id(s) for CUDA_VISIBLE_DEVICES')
parser.add_argument('--use_CAC', type=bool, default=False, help='whether use CAC')
parser.add_argument('--iou_bound', type=float, default=0.25, help='consistency')
parser.add_argument('--stride', type=int, default=4, help='consistency')
args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu


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
def cross_entropy_loss_con(output, target, condition):
    """
    cross entropy loss for conditional network
    """
    softmax = torch.softmax(output,dim=1)
    B,C,D,H,W = softmax.shape
    softmax_con = torch.zeros(B,2,D,H,W).cuda()
    softmax_con[:,1,...] = softmax[np.arange(B),condition.squeeze(),...] 
    softmax_con[:,0,...] = 1.0 - softmax_con[:,1,...] + 1e-7
    log = -torch.log(softmax_con.gather(1, target.unsqueeze(1)))
    loss = log.mean()
    return loss

def train(args, snapshot_path):
    base_lr = args.base_lr
    train_data_path = args.root_path
    batch_size = args.batch_size
    max_iterations = args.max_iterations
    num_classes = 6
    args.train_list = "/data/liupeng/semi-supervised_segmentation/SSL4MIS-master/data/BCV/train.txt"

    net1 = net_factory_3d(net_type=args.model, in_chns=1, class_num=num_classes)
    net2 = net_factory_3d(net_type=args.model, in_chns=1, class_num=num_classes)
    model1 = kaiming_normal_init_weight(net1)
    model2 = xavier_normal_init_weight(net2)
    if not args.use_CAC:
        model1=nn.DataParallel(model1,device_ids=[0,1])
        model2=nn.DataParallel(model2,device_ids=[0,1])
    model1.train()
    model2.train()

    if args.dataset=='BCV':
        db_train = BCVDataset(img_list_file=args.train_list,
                                patch_size=args.patch_size,
                                transforms=transforms.Compose([
                                RandomRotFlip(),
                                RandomCrop(args.patch_size),
                                ToTensor(),
                            ]))
        if args.use_CAC:
            print("======> use CAC ")
            db_train = BCVDatasetCAC(img_list_file=args.train_list, patch_size=args.patch_size, num_class=num_classes, stride=args.stride, iou_bound=[args.iou_bound, 0.95])
    else:
        db_train = BraTS2019(base_dir=train_data_path,
                            split='train',
                            num=None,
                            transform=transforms.Compose([
                                RandomRotFlip(),
                                RandomCrop(args.patch_size),
                                ToTensor(),
                            ]))

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    labeled_idxs = list(range(0, args.labeled_num))
    unlabeled_idxs = list(range(args.labeled_num, 24))
    batch_sampler = TwoStreamBatchSampler(
        labeled_idxs, unlabeled_idxs, batch_size, batch_size - args.labeled_bs)

    trainloader = DataLoader(db_train, batch_sampler=batch_sampler,
                             num_workers=4, pin_memory=True, worker_init_fn=worker_init_fn)

    optimizer1 = optim.SGD(model1.parameters(), lr=base_lr,
                           momentum=0.9, weight_decay=0.0001)
    optimizer2 = optim.SGD(model2.parameters(), lr=base_lr,
                           momentum=0.9, weight_decay=0.0001)
    # optimizer1 = optim.Adam(model1.parameters(), lr=base_lr, weight_decay=0.0001)
    # optimizer2 = optim.Adam(model2.parameters(), lr=base_lr, weight_decay=0.0001)
    
    best_performance1 = 0.0
    best_performance2 = 0.0
    iter_num = 0
    ce_loss = CrossEntropyLoss()
    dice_loss = losses.DiceLoss(num_classes)

    wandb.init(project="semi-supervised-segmentation", name="{}_{}_{}_refine".format(args.dataset,args.exp,args.model),
                config=args)
    wandb.tensorboard.patch(root_logdir=snapshot_path + '/log')
    writer = SummaryWriter(snapshot_path + '/log')
    logging.info("{} iterations per epoch".format(len(trainloader)))

    max_epoch = max_iterations // len(trainloader) + 1
    iter_each_epoch = len(trainloader)
    iterator = tqdm(range(max_epoch), ncols=70)
    for epoch_num in iterator:
        for i_batch, sampled_batch in enumerate(trainloader):
            volume_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            volume_batch, label_batch = volume_batch.cuda(), label_batch.cuda()
            ul1,br1,ul2,br2 = [],[],[],[]
            labeled_idxs_batch, unlabeled_idxs_batch = torch.arange(0,args.labeled_bs),torch.arange(args.labeled_bs,args.batch_size)
            if args.use_CAC:
                ul1,br1,ul2,br2 = sampled_batch['ul1'],sampled_batch['br1'],sampled_batch['ul2'],sampled_batch['br2']
                volume_batch = torch.cat([volume_batch[:,0,...],volume_batch[:,1,...]],dim=0)
                label_batch = torch.cat([label_batch[:,0,...],label_batch[:,1,...]],dim=0)
                labeled_idxs2_batch = torch.arange(args.batch_size,args.batch_size+args.labeled_bs)
                labeled_idxs1_batch = torch.arange(0,args.labeled_bs)
                labeled_idxs_batch = torch.cat([labeled_idxs1_batch, labeled_idxs2_batch])
                unlabeled_idxs1_batch = torch.arange(args.labeled_bs,args.batch_size)
                unlabeled_idxs2_batch = torch.arange(args.batch_size+args.labeled_bs, 2 * args.batch_size)
                unlabeled_idxs_batch = torch.cat([unlabeled_idxs1_batch, unlabeled_idxs2_batch])

            outputs1 = model1(volume_batch)
            outputs_soft1 = torch.softmax(outputs1, dim=1)

            outputs2 = model2(volume_batch)
            outputs_soft2 = torch.softmax(outputs2, dim=1)

            consistency_weight = get_current_consistency_weight((iter_num-args.began_semi_iter) // iter_each_epoch)

            loss1 = 0.5 * (ce_loss(outputs1[labeled_idxs_batch],
                                   label_batch[labeled_idxs_batch].long()) + dice_loss(
                outputs_soft1[labeled_idxs_batch], label_batch[labeled_idxs_batch].unsqueeze(1)))
            loss2 = 0.5 * (ce_loss(outputs2[labeled_idxs_batch],
                                   label_batch[labeled_idxs_batch].long()) + dice_loss(
                outputs_soft2[labeled_idxs_batch], label_batch[labeled_idxs_batch].unsqueeze(1)))


            if args.use_CAC:
                overlap_soft1_list = []
                overlap_soft2_list = []
                overlap_outputs1_list = []
                overlap_outputs2_list = []
                for unlabeled_idx1,unlabeled_idx2 in zip(unlabeled_idxs1_batch, unlabeled_idxs2_batch):
                    # overlap region pred soft by model1
                    overlap1_soft1 = outputs_soft1[unlabeled_idx1][:,ul1[0][1]:br1[0][1],ul1[1][1]:br1[1][1],ul1[2][1]:br1[2][1]]
                    overlap2_soft1 = outputs_soft1[unlabeled_idx2][:,ul2[0][1]:br2[0][1],ul2[1][1]:br2[1][1],ul2[2][1]:br2[2][1]]
                    assert overlap1_soft1.shape == overlap2_soft1.shape,"overlap  region size mush equal"
                    
                    # overlap region pred by model1
                    overlap1_outputs1 = outputs1[unlabeled_idx1][:,ul1[0][1]:br1[0][1],ul1[1][1]:br1[1][1],ul1[2][1]:br1[2][1]]
                    overlap2_outputs1 = outputs1[unlabeled_idx2][:,ul2[0][1]:br2[0][1],ul2[1][1]:br2[1][1],ul2[2][1]:br2[2][1]]
                    assert overlap1_outputs1.shape == overlap2_outputs1.shape,"overlap  region size mush equal"
                    overlap_outputs1_list.append(overlap1_outputs1.unsqueeze(0))
                    overlap_outputs1_list.append(overlap2_outputs1.unsqueeze(0))

                    # overlap region pred by model2
                    overlap1_soft2 = outputs_soft2[unlabeled_idx1][:,ul1[0][1]:br1[0][1],ul1[1][1]:br1[1][1],ul1[2][1]:br1[2][1]]
                    overlap2_soft2 = outputs_soft2[unlabeled_idx2][:,ul2[0][1]:br2[0][1],ul2[1][1]:br2[1][1],ul2[2][1]:br2[2][1]]
                    assert overlap1_soft2.shape == overlap2_soft2.shape,"overlap  region size mush equal"
                    
                    # overlap region pred by model2
                    overlap1_outputs2 = outputs2[unlabeled_idx1][:,ul1[0][1]:br1[0][1],ul1[1][1]:br1[1][1],ul1[2][1]:br1[2][1]]
                    overlap2_outputs2 = outputs2[unlabeled_idx2][:,ul2[0][1]:br2[0][1],ul2[1][1]:br2[1][1],ul2[2][1]:br2[2][1]]
                    assert overlap1_outputs2.shape == overlap2_outputs2.shape,"overlap  region size mush equal"
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
            
            if iter_num < args.began_semi_iter:
                pseudo_supervision1 = torch.FloatTensor([0]).cuda()
            else:
                if args.use_CAC:
                    overlap_pseudo_outputs2 = torch.argmax(overlap_soft2.detach(), dim=1, keepdim=False)
                    overlap_pseudo_outputs2 = torch.cat([overlap_pseudo_outputs2, overlap_pseudo_outputs2])
                    pseudo_supervision1 = ce_loss(overlap_outputs1, overlap_pseudo_outputs2)
                else:
                    pseudo_outputs2 = torch.argmax(outputs_soft2[args.labeled_bs:].detach(), dim=1, keepdim=False)
                    pseudo_supervision1 = ce_loss(outputs1[args.labeled_bs:], pseudo_outputs2)
            
            if iter_num < args.began_semi_iter:
                pseudo_supervision2 = torch.FloatTensor([0]).cuda()
            else:
                if args.use_CAC:
                    overlap_pseudo_outputs1 = torch.argmax(overlap_soft1.detach(), dim=1, keepdim=False)
                    overlap_pseudo_outputs1 = torch.cat([overlap_pseudo_outputs1, overlap_pseudo_outputs1])
                    pseudo_supervision2 = ce_loss(overlap_outputs2, overlap_pseudo_outputs1.long()) 
                else:
                    pseudo_outputs1 = torch.argmax(outputs_soft1[args.labeled_bs:].detach(), dim=1, keepdim=False)
                    pseudo_supervision2 = ce_loss(outputs2[args.labeled_bs:], pseudo_outputs1.long())


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
            writer.add_scalar(
                'consistency_weight/consistency_weight', consistency_weight, iter_num)
            writer.add_scalar('loss/model1_loss',
                              model1_loss, iter_num)
            writer.add_scalar('loss/model2_loss',
                              model2_loss, iter_num)
            writer.add_scalar('loss/pseudo_supervision1',
                              pseudo_supervision1, iter_num)
            writer.add_scalar('loss/pseudo_supervision2',
                              pseudo_supervision2, iter_num)
            logging.info(
                'iteration %d : model1 loss : %f model2 loss : %f pseudo_supervision1 : %f pseudo_supervision2 : %f' % (iter_num, model1_loss.item(), model2_loss.item(), pseudo_supervision1.item(), pseudo_supervision2.item()))
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

            if iter_num > args.began_eval_iter and iter_num % 200 == 0:
                model1.eval()
                if args.dataset == 'BCV':
                    avg_metric1 = test_all_case_BCV(model1,
                    test_list="/data/liupeng/semi-supervised_segmentation/SSL4MIS-master/data/BCV/test.txt",
                    num_classes=num_classes, patch_size=args.patch_size,stride_xy=64, stride_z=64)
                else:
                    avg_metric1 = test_all_case(
                        model1, args.root_path, test_list="val.txt", num_classes=2, patch_size=args.patch_size,
                        stride_xy=64, stride_z=64)
                if avg_metric1[:, 0].mean() > best_performance1:
                    best_performance1 = avg_metric1[:, 0].mean()
                    save_mode_path = os.path.join(snapshot_path,
                                                  'model1_iter_{}_dice_{}.pth'.format(
                                                      iter_num, round(best_performance1, 4)))
                    save_best = os.path.join(snapshot_path,
                                             '{}_best_model1.pth'.format(args.model))
                    torch.save(model1.state_dict(), save_mode_path)
                    torch.save(model1.state_dict(), save_best)

                writer.add_scalar('info/model1_val_dice_score',
                                  avg_metric1[0, 0], iter_num)
                writer.add_scalar('info/model1_val_hd95',
                                  avg_metric1[0, 1], iter_num)
                logging.info(
                    'iteration %d : model1_dice_score : %f model1_hd95 : %f' % (
                        iter_num, avg_metric1[0, 0].mean(), avg_metric1[0, 1].mean()))
                model1.train()

                model2.eval()
                if args.dataset == 'BCV':
                    avg_metric2 = test_all_case_BCV(model2,
                    test_list="/data/liupeng/semi-supervised_segmentation/SSL4MIS-master/data/BCV/test.txt",
                    num_classes=num_classes, patch_size=args.patch_size,stride_xy=64, stride_z=64)
                else:
                    avg_metric2 = test_all_case(
                        model2, args.root_path, test_list="val.txt", num_classes=2, patch_size=args.patch_size,
                        stride_xy=64, stride_z=64)
                if avg_metric2[:, 0].mean() > best_performance2:
                    best_performance2 = avg_metric2[:, 0].mean()
                    save_mode_path = os.path.join(snapshot_path,
                                                  'model2_iter_{}_dice_{}.pth'.format(
                                                      iter_num, round(best_performance2, 4)))
                    save_best = os.path.join(snapshot_path,
                                             '{}_best_model2.pth'.format(args.model))
                    torch.save(model2.state_dict(), save_mode_path)
                    torch.save(model2.state_dict(), save_best)

                writer.add_scalar('info/model2_val_dice_score',
                                  avg_metric2[0, 0], iter_num)
                writer.add_scalar('info/model2_val_hd95',
                                  avg_metric2[0, 1], iter_num)
                logging.info(
                    'iteration %d : model2_dice_score : %f model2_hd95 : %f' % (
                        iter_num, avg_metric2[0, 0].mean(), avg_metric2[0, 1].mean()))
                model2.train()

            if iter_num % 2000 == 0:
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
    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    snapshot_path = "../model/{}_{}_{}_condition/{}".format(
        args.exp, args.labeled_num, args.dataset, args.model)
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)
    if os.path.exists(snapshot_path + '/code'):
        shutil.rmtree(snapshot_path + '/code')
    shutil.copytree('.', snapshot_path + '/code',
                    shutil.ignore_patterns(['.git', '__pycache__']))

    logging.basicConfig(filename=snapshot_path+"/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    train(args, snapshot_path)

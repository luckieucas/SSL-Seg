import argparse
import logging
import os
import random
import shutil
import sys
import time

import numpy as np
import wandb
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

from dataset.BCVData import BCVDataset,BCVDatasetCAC
from dataset.brats2019 import (BraTS2019, CenterCrop, RandomCrop,
                                   RandomRotFlip, ToTensor,
                                   TwoStreamBatchSampler)
from networks.net_factory_3d import net_factory_3d
from utils import losses, metrics, ramps
from val_3D import test_all_case, test_all_case_BCV
from utils.util import save_config

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str,
                    default='../data/BraTS2019', help='Name of Experiment')
parser.add_argument('--dataset', type=str,
                    default='BCV', help='use which dataset')
parser.add_argument('--exp', type=str,
                    default='BCV_ICT', help='experiment_name')
parser.add_argument('--model', type=str,
                    default='unet_3D', help='model_name')
parser.add_argument('--max_iterations', type=int,
                    default=20000, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=4,
                    help='batch_size per gpu')
parser.add_argument('--deterministic', type=int,  default=1,
                    help='whether use deterministic training')
parser.add_argument('--base_lr', type=float,  default=0.01,
                    help='segmentation network learning rate')
parser.add_argument('--patch_size', type=list,  default=[96, 160, 160],
                    help='patch size of network input')
parser.add_argument('--seed', type=int,  default=1337, help='random seed')

# label and unlabel
parser.add_argument('--labeled_bs', type=int, default=2,
                    help='labeled_batch_size per gpu')
parser.add_argument('--labeled_num', type=int, default=10,
                    help='labeled data')
parser.add_argument('--total_labeled_num', type=int, default=14,
                    help='total labeled data')
parser.add_argument('--ict_alpha', type=int, default=0.2,
                    help='ict_alpha')
# costs
parser.add_argument('--ema_decay', type=float,  default=0.99, help='ema_decay')
parser.add_argument('--consistency_type', type=str,
                    default="mse", help='consistency_type')
parser.add_argument('--consistency', type=float,
                    default=0.1, help='consistency')
parser.add_argument('--consistency_rampup', type=float,
                    default=200.0, help='consistency_rampup')
parser.add_argument('--gpu', default='8', type=str, help='id(s) for CUDA_VISIBLE_DEVICES')
parser.add_argument('--config', type=str,
                    default='train_config.yaml', help='model_name')
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


def train_ICT(config, snapshot_path):
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

    def create_model(ema=False):
        # Network definition
        net = net_factory_3d(net_type=backbone, in_chns=1, class_num=num_classes)
        model = net.cuda()
        if ema:
            for param in model.parameters():
                param.detach_()
        return model

    model = create_model()
    ema_model = create_model(ema=True)

    if dataset_name in ['BCV','MMWHS']:
        print("==>> use BCV >>")
        db_train = BCVDataset(
            img_list_file=train_list,
            patch_size=patch_size,
            labeled_num=labeled_num,
            transforms=transforms.Compose([
                RandomRotFlip(),
                RandomCrop(patch_size),
                ToTensor(),
            ])
        )
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
        labeled_idxs, unlabeled_idxs, batch_size, batch_size-labeled_bs)

    trainloader = DataLoader(db_train, batch_sampler=batch_sampler,
                             num_workers=4, pin_memory=True, worker_init_fn=worker_init_fn)

    model.train()
    ema_model.train()

    optimizer = optim.SGD(model.parameters(), lr=base_lr,
                          momentum=0.9, weight_decay=0.0001)
    ce_loss = CrossEntropyLoss()
    dice_loss = losses.DiceLoss(num_classes)
    wandb.init(
        project="semi-supervised-segmentation", 
        name=f"{dataset_name}_{method_name}_{backbone}_labeled{labeled_num}",
        config=config
    )
    wandb.tensorboard.patch(root_logdir=snapshot_path + '/log')
    writer = SummaryWriter(snapshot_path + '/log')
    logging.info("{} iterations per epoch".format(len(trainloader)))

    iter_num = 0
    max_epoch = max_iterations // len(trainloader) + 1
    best_performance = 0.0
    iterator = tqdm(range(max_epoch), ncols=70)
    for epoch_num in iterator:
        for i_batch, sampled_batch in enumerate(trainloader):

            volume_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            volume_batch, label_batch = volume_batch.cuda(), label_batch.cuda()
            labeled_volume_batch = volume_batch[:labeled_bs]
            unlabeled_volume_batch = volume_batch[labeled_bs:]

            # ICT mix factors
            ict_mix_factors = np.random.beta(method_config['ict_alpha'], 
                                             method_config['ict_alpha'], 
                                             size=(labeled_bs//2, 1, 1, 1, 1))
            ict_mix_factors = torch.tensor(
                ict_mix_factors, dtype=torch.float).cuda()
            unlabeled_volume_batch_0 = unlabeled_volume_batch[0:1, ...]
            unlabeled_volume_batch_1 = unlabeled_volume_batch[1:2, ...]

            # Mix images
            batch_ux_mixed = unlabeled_volume_batch_0 * \
                (1.0 - ict_mix_factors) + \
                unlabeled_volume_batch_1 * ict_mix_factors
            input_volume_batch = torch.cat(
                [labeled_volume_batch, batch_ux_mixed], dim=0)
            outputs = model(input_volume_batch)
            outputs_soft = torch.softmax(outputs, dim=1)
            with torch.no_grad():
                ema_output_ux0 = torch.softmax(
                    ema_model(unlabeled_volume_batch_0), dim=1)
                ema_output_ux1 = torch.softmax(
                    ema_model(unlabeled_volume_batch_1), dim=1)
                batch_pred_mixed = ema_output_ux0 * \
                    (1.0 - ict_mix_factors) + ema_output_ux1 * ict_mix_factors

            loss_ce = ce_loss(outputs[:labeled_bs],
                              label_batch[:labeled_bs][:])
            loss_dice = dice_loss(
                outputs_soft[:labeled_bs], label_batch[:labeled_bs].unsqueeze(1))
            supervised_loss = 0.5 * (loss_dice + loss_ce)
            consistency_weight = get_current_consistency_weight(iter_num//150)
            consistency_loss = torch.mean(
                (outputs_soft[labeled_bs:] - batch_pred_mixed)**2)
            if iter_num > began_semi_iter:
                loss = supervised_loss + consistency_weight * consistency_loss
            else:
                loss = supervised_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            update_ema_variables(model, ema_model, ema_decay, iter_num)

            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_

            iter_num = iter_num + 1
            writer.add_scalar('info/lr', lr_, iter_num)
            writer.add_scalar('info/total_loss', loss, iter_num)
            writer.add_scalar('info/loss_ce', loss_ce, iter_num)
            writer.add_scalar('info/loss_dice', loss_dice, iter_num)
            writer.add_scalar('info/consistency_loss',
                              consistency_loss, iter_num)
            writer.add_scalar('info/consistency_weight',
                              consistency_weight, iter_num)

            logging.info(
                'iteration %d : loss : %f, loss_ce: %f, loss_dice: %f' %
                (iter_num, loss.item(), loss_ce.item(), loss_dice.item()))
            writer.add_scalar('loss/loss', loss, iter_num)

            if iter_num % 20 == 0:
                image = volume_batch[0, 0:1, :, :, 20:61:10].permute(
                    3, 0, 1, 2).repeat(1, 3, 1, 1)
                grid_image = make_grid(image, 5, normalize=True)
                writer.add_image('train/Image', grid_image, iter_num)

                image = outputs_soft[0, 1:2, :, :, 20:61:10].permute(
                    3, 0, 1, 2).repeat(1, 3, 1, 1)
                grid_image = make_grid(image, 5, normalize=False)
                writer.add_image('train/Predicted_label',
                                 grid_image, iter_num)

                image = label_batch[0, :, :, 20:61:10].unsqueeze(
                    0).permute(3, 0, 1, 2).repeat(1, 3, 1, 1)
                grid_image = make_grid(image, 5, normalize=False)
                writer.add_image('train/Groundtruth_label',
                                 grid_image, iter_num)

            if iter_num > 0 and iter_num % 200 == 0:
                model.eval()
                if dataset_name in ['BCV','MMWHS']:
                    avg_metric = test_all_case_BCV(model,test_list=test_list,
                                                   num_classes=num_classes, 
                                                   patch_size=patch_size,
                                                   stride_xy=64, stride_z=64)
                else:
                    avg_metric = test_all_case(model, train_data_path, 
                                               test_list="val.txt", 
                                               num_classes=num_classes, 
                                               patch_size=patch_size,
                                               stride_xy=64, stride_z=64)
                if avg_metric[:, 0].mean() > best_performance:
                    best_performance = avg_metric[:, 0].mean()
                    save_mode_path = os.path.join(snapshot_path,
                                                  'iter_{}_dice_{}.pth'.format(
                                                  iter_num, 
                                                  round(best_performance, 4)))
                    save_best = os.path.join(snapshot_path,
                                             '{}_best_model.pth'.format(
                                             backbone))
                    torch.save(model.state_dict(), save_mode_path)
                    torch.save(model.state_dict(), save_best)

                writer.add_scalar('info/val_dice_score',
                                  avg_metric[0, 0], iter_num)
                writer.add_scalar('info/val_hd95',
                                  avg_metric[0, 1], iter_num)
                logging.info(
                    'iteration %d : dice_score : %f hd95 : %f' % (iter_num, 
                    avg_metric[0, 0].mean(), avg_metric[0, 1].mean()))
                model.train()

            if iter_num % 3000 == 0:
                save_mode_path = os.path.join(
                    snapshot_path, 'iter_' + str(iter_num) + '.pth')
                torch.save(model.state_dict(), save_mode_path)
                logging.info("save model to {}".format(save_mode_path))

            if iter_num >= max_iterations:
                break
        if iter_num >= max_iterations:
            iterator.close()
            break
    writer.close()
    return "Training Finished!"


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

    snapshot_path = "../model/{}_{}_{}_ICT/{}".format(
        config['exp'], 
        config['DATASET']['labeled_num'], 
        config['DATASET']['name'], 
        config['backbone']
    )
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)
    # move config to snapshot_path
    shutil.copyfile(args.config, snapshot_path+"/"+
                    time.strftime("%Y-%m-%d=%H:%M:%S", time.localtime())+
                    "_train_config.yaml")
    # if os.path.exists(snapshot_path + '/code'):
    #     shutil.rmtree(snapshot_path + '/code')
    # shutil.copytree('.', snapshot_path + '/code',
    #                 shutil.ignore_patterns(['.git', '__pycache__']))

    logging.basicConfig(
        filename=snapshot_path+"/log.txt", 
        level=logging.INFO,
        format='[%(asctime)s.%(msecs)03d] %(message)s', 
        datefmt='%H:%M:%S'
    )
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(config))
    train_ICT(config, snapshot_path)

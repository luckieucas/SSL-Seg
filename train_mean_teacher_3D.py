import argparse
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


def train_MT(config, snapshot_path):
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
        db_train = BCVDataset(
            img_list_file=train_list,
            patch_size=patch_size,
            labeled_num= labeled_num,
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
        random.seed(config['seed']+ worker_id)

    labeled_idxs = list(range(0, labeled_num))
    unlabeled_idxs = list(range(labeled_num, training_data_num))
    batch_sampler = TwoStreamBatchSampler(
        labeled_idxs, 
        unlabeled_idxs, 
        batch_size, 
        batch_size-labeled_bs
    )

    trainloader = DataLoader(
        db_train, 
        batch_sampler=batch_sampler,
        num_workers=2, pin_memory=True, 
        worker_init_fn=worker_init_fn
    )

    model.train()
    ema_model.train()
    optimizer = optim.SGD(
        model.parameters(), 
        lr=base_lr,
        momentum=0.9, 
        weight_decay=0.0001
    )
    ce_loss = CrossEntropyLoss()
    dice_loss = losses.DiceLoss(num_classes)
    wandb.init(
        project="semi-supervised-segmentation", 
        name=f"{dataset_name}_{exp}_{backbone}_labeled{labeled_num}",
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
            volume_batch, label_batch = (
                sampled_batch['image'], 
                sampled_batch['label']
            )
            volume_batch, label_batch = volume_batch.cuda(), label_batch.cuda()
            unlabeled_volume_batch = volume_batch[labeled_bs:]

            noise = torch.clamp(
                torch.randn_like(unlabeled_volume_batch) * 0.1, 
                -0.2, 
                0.2
            )
            ema_inputs = unlabeled_volume_batch + noise

            outputs = model(volume_batch)
            outputs_soft = torch.softmax(outputs, dim=1)
            with torch.no_grad():
                ema_output = ema_model(ema_inputs)
                ema_output_soft = torch.softmax(ema_output, dim=1)

            loss_ce = ce_loss(outputs[:labeled_bs],
                              label_batch[:labeled_bs][:])
            loss_dice = dice_loss(
                outputs_soft[:labeled_bs], label_batch[:labeled_bs].unsqueeze(1))
            supervised_loss = 0.5 * (loss_dice + loss_ce)
            
            consistency_weight = get_current_consistency_weight(iter_num//4)
            if iter_num > began_semi_iter:
                consistency_loss = torch.mean(
                (outputs_soft[labeled_bs:] - ema_output_soft)**2)
            else:
                consistency_loss = torch.FloatTensor([0]).cuda()
            loss = supervised_loss + consistency_weight * consistency_loss
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

            if iter_num > began_eval_iter and iter_num % 200 == 0:
                model.eval()
                if dataset_name in ['BCV','MMWHS']:
                    avg_metric = test_all_case_BCV(
                        model,
                        test_list=test_list,
                        num_classes=num_classes, 
                        patch_size=patch_size,
                        stride_xy=64, 
                        stride_z=64
                    )
                else:
                    avg_metric = test_all_case(
                        model, 
                        train_data_path, 
                        test_list="val.txt", 
                        num_classes=2, 
                        patch_size=patch_size,
                        stride_xy=64, 
                        stride_z=64
                    )
                if avg_metric[:, 0].mean() > best_performance:
                    best_performance = avg_metric[:, 0].mean()
                    save_mode_path = os.path.join(snapshot_path,
                                                  'iter_{}_dice_{}.pth'.format(
                                                      iter_num, round(best_performance, 4)))
                    save_best = os.path.join(snapshot_path,
                                             '{}_best_model.pth'.format(backbone))
                    torch.save(model.state_dict(), save_mode_path)
                    torch.save(model.state_dict(), save_best)

                writer.add_scalar('info/val_dice_score',
                                  avg_metric[0, 0], iter_num)
                writer.add_scalar('info/val_hd95',
                                  avg_metric[0, 1], iter_num)
                logging.info(
                    'iteration %d : dice_score : %f hd95 : %f' % (iter_num, avg_metric[0, 0].mean(), avg_metric[0, 1].mean()))
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

    snapshot_path = "../model/{}_{}_{}_MT/{}".format(
        config['exp'], 
        config['DATASET']['labeled_num'], 
        config['DATASET']['name'], 
        config['backbone']
    )
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)
    # move config to snapshot_path
    shutil.copyfile(args.config, snapshot_path+"/train_config.yaml")
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
    train_MT(config, snapshot_path)

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
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import make_grid
from tqdm import tqdm

from dataset import utils
from dataset.BCVData import BCVDataset
from dataset.brats2019 import (BraTS2019, CenterCrop, RandomCrop,
                                   RandomRotFlip, ToTensor,
                                   TwoStreamBatchSampler)
from dataset.pet import PETDataset
from networks.net_factory_3d import net_factory_3d
from utils import losses, metrics, ramps
from utils.util import kaiming_normal_init_weight
from val_3D import test_all_case,test_all_case_BCV
from test import test_model_3d

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str,
                    default='../data/BraTS2019', help='Name of Experiment')
parser.add_argument('--dataset', type=str,
                    default='BCV', help='use which dataset')
parser.add_argument('--train_list', type=str,
                    default='/data/liupeng/PET/pet_additional/train124.txt', help='file with training image list')
parser.add_argument('--test_list', type=str,
                    default='/data/liupeng/PET/pet_additional/test14.txt', help='file with testing image list')
parser.add_argument('--exp', type=str,
                    default='MMWHS_Fully_Supervised_SGD', help='experiment_name')
parser.add_argument('--model', type=str,
                    default='unet_3D', help='model_name')
parser.add_argument('--max_iterations', type=int,
                    default=30000, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=2,
                    help='batch_size per gpu')
parser.add_argument('--deterministic', type=int,  default=1,
                    help='whether use deterministic training')
parser.add_argument('--base_lr', type=float,  default=0.01,
                    help='segmentation network learning rate')
parser.add_argument('--patch_size', type=list,  default=[96, 160, 160],
                    help='patch size of network input')
parser.add_argument('--seed', type=int,  default=1337, help='random seed')
parser.add_argument('--labeled_num', type=int, default=4,
                    help='labeled data')
parser.add_argument('--began_eval_iter', type=int, default=0, help='iteration to began evaluation')
parser.add_argument('--gpu', default='1', type=str, help='id(s) for CUDA_VISIBLE_DEVICES')

args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu


def train(args, snapshot_path):
    base_lr = args.base_lr
    train_data_path = args.root_path
    batch_size = args.batch_size
    max_iterations = args.max_iterations
    num_classes = 5
    args.train_list = "../data/Flare/Flare_train.txt"
    args.test_list = "../data/Flare/Flare_test.txt"
    
    model = net_factory_3d(net_type=args.model, in_chns=1, class_num=num_classes)
    model = kaiming_normal_init_weight(model)
    if args.dataset == "pet":
        db_train = PETDataset("/data/liupeng/PET/pet_additional/train124.txt",
                                transforms=transforms.Compose([
                                RandomRotFlip(),
                                ToTensor(),
                            ])
                             )
        db_test = PETDataset("/data/liupeng/PET/pet_additional/test14.txt")
    elif args.dataset == "BCV":
        db_train = BCVDataset(img_list_file=args.train_list,
                            patch_size=args.patch_size,
                            labeled_num=args.labeled_num,
                            transforms=transforms.Compose([
                            RandomRotFlip(),
                            RandomCrop(args.patch_size),
                            ToTensor(),
                        ]))
    else:
        db_train = BraTS2019(base_dir=train_data_path,
                            split='train',
                            num=args.labeled_num,
                            transform=transforms.Compose([
                                RandomRotFlip(),
                                RandomCrop(args.patch_size),
                                ToTensor(),
                            ]))

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    trainloader = DataLoader(db_train, batch_size=batch_size, shuffle=True,
                             num_workers=4, pin_memory=True, worker_init_fn=worker_init_fn)
    #testloader = DataLoader(db_test, batch_size=batch_size*2, shuffle=False,
    #                         num_workers=16, pin_memory=True, worker_init_fn=worker_init_fn)
    model.train()

    optimizer = optim.SGD(model.parameters(), lr=base_lr,
                          momentum=0.9, weight_decay=0.0001)
    ce_loss = CrossEntropyLoss()
    dice_loss = losses.DiceLoss(num_classes)

    wandb.init(
        project="semi-supervised-segmentation", 
        name="{}_{}_labeledNum{}".format(args.exp,args.model,args.labeled_num),
        config=args
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

            volume_batch, label_batch = sampled_batch['image'], sampled_batch['label'].long()
            volume_batch, label_batch = volume_batch.cuda(), label_batch.cuda()

            outputs = model(volume_batch)
            outputs_soft = torch.softmax(outputs, dim=1)

            loss_ce = ce_loss(outputs, label_batch.long())
            loss_dice = dice_loss(outputs_soft, label_batch.unsqueeze(1))
            loss = 0.5 * (loss_dice + loss_ce)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_

            iter_num = iter_num + 1
            writer.add_scalar('info/lr', lr_, iter_num)
            writer.add_scalar('info/total_loss', loss, iter_num)
            writer.add_scalar('info/loss_ce', loss_ce, iter_num)
            writer.add_scalar('info/loss_dice', loss_dice, iter_num)

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

            if iter_num > args.began_eval_iter and iter_num % 200 == 0:
                model.eval()
                if args.dataset=='pet':
                    #avg_metric = test_model_3d(testloader,model)
                    print("evaulate pet data")
                elif args.dataset=='BCV':
                    avg_metric = test_all_case_BCV(model,
                    test_list= args.test_list,
                    num_classes=num_classes, patch_size=args.patch_size,stride_xy=64, stride_z=64)
                else:
                    avg_metric = test_all_case(
                        model, args.root_path, test_list="val.txt", num_classes=2, patch_size=args.patch_size,
                        stride_xy=64, stride_z=64)
                if avg_metric[:, 0].mean() > best_performance:
                    best_performance = avg_metric[:, 0].mean()
                    save_mode_path = os.path.join(snapshot_path,
                                                  'iter_{}_dice_{}.pth'.format(
                                                      iter_num, round(best_performance, 4)))
                    save_best = os.path.join(snapshot_path,
                                             '{}_best_model.pth'.format(args.model))
                    torch.save(model.state_dict(), save_mode_path)
                    torch.save(model.state_dict(), save_best)

                writer.add_scalar('info/val_dice_score',
                                  avg_metric[:, 0].mean(), iter_num)
                writer.add_scalar('info/val_hd95',
                                  avg_metric[:, 1].mean(), iter_num)
                logging.info(
                    'iteration %d : dice_score : %f hd95 : %f' % (iter_num, avg_metric[:, 0].mean(), avg_metric[:, 1].mean()))
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

    snapshot_path = "../model/fully_supervised_{}_labeledNum{}/{}".format(args.exp, args.labeled_num, args.model)
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)

    logging.basicConfig(filename=snapshot_path+"/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    train(args, snapshot_path)

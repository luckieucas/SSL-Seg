from logging import raiseExceptions
import os
import sys
from cv2 import threshold
import yaml
import torch
import torch.nn as nn
from torch.optim import lr_scheduler
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
import torch.cuda.amp as amp
from tensorboardX import SummaryWriter
import random
import wandb
from tqdm import tqdm
import numpy as np
import argparse

from utils import losses,ramps,cac_loss
from dataset.BCVData import BCVDataset, BCVDatasetCAC,DatasetSR
from dataset.dataset import DatasetSemi
from dataset.sampler import BatchSampler, ClassRandomSampler
from networks.net_factory_3d import net_factory_3d
from dataset.dataset import TwoStreamBatchSampler
from dataset.dataset_old import (BaseDataSets, RandomGenerator,
                                 TwoStreamBatchSampler)
from semi_trainer import SemiSupervisedTrainerBase
from networks.net_factory import net_factory
from networks.vision_transformer import SwinUnet as ViT_seg
from config import get_config
from arguments import Namespace



class SemiSupervisedTrainer2D(SemiSupervisedTrainerBase):
    def __init__(self, config, output_folder, logging, root_path) -> None:
        SemiSupervisedTrainerBase.__init__(self, config, output_folder, logging)
        self.root_path = root_path
        self.dataset_val = None
        self.dataloader_val = None
        parser = argparse.ArgumentParser()
        args = parser.parse_args()
        args.cfg = "../code/configs/swin_tiny_patch4_window7_224_lite.yaml"
        for key, value in Namespace(config).__dict__.items():
            vars(args)[key] = value
        self.args = args
    
    
    def load_dataset(self):
        self.dataset = BaseDataSets(base_dir=self.root_path, 
                            split="train", num=None, 
                            transform=transforms.Compose([
                            RandomGenerator(self.patch_size)
                            ]))
        self.dataset_val = BaseDataSets(base_dir=self.root_path, split="val")
    
    def initialize_network(self):
        def create_model(ema=False):
        # Network definition
            model = net_factory(net_type=self.backbone, in_chns=1,
                                class_num=self.num_classes)
            if ema:
                for param in model.parameters():
                    param.detach_()
            return model

        self.model = create_model()
        config = get_config(self.args)
        self.model2 = ViT_seg(config, img_size=self.patch_size,
                        num_classes=self.num_classes).cuda()
        self.model2.load_from(config)
    
    def __patients_to_slices(dataset, patiens_num):
        ref_dict = None
        if "ACDC" in dataset:
            ref_dict = {"3": 68, "7": 136,
                        "14": 256, "21": 396, "28": 512, "35": 664, "140": 1312}
        elif "Prostate" in dataset:
            ref_dict = {"2": 27, "4": 53, "8": 120,
                        "12": 179, "16": 256, "21": 312, "42": 623}
        elif "LA" in dataset:
            ref_dict = {"2": 27, "4": 53, "8": 704,
                        "12": 179, "16": 256, "21": 312, "42": 623}
        elif "BCV" in dataset:
            ref_dict = {"2": 27, "4": 569, "8": 704,
                        "12": 179, "16": 256, "21": 312, "42": 623}
        elif "MMWHS" in dataset:
            ref_dict = {"2": 567, "4": 569, "8": 704,
                        "12": 179, "16": 256, "21": 312, "42": 623}
        else:
            print("Error")
        return ref_dict[str(patiens_num)]
    
    def get_dataloader(self):
        total_slices = len(self.dataset)
        labeled_slice = self.__patients_to_slices(self.dataset_name, 
                                                  self.labeled_num)
        print("Total silices is: {}, labeled slices is: {}".format(
            total_slices, labeled_slice))
        labeled_idxs = list(range(0, labeled_slice))
        unlabeled_idxs = list(range(labeled_slice, total_slices))
        batch_sampler = TwoStreamBatchSampler(
            labeled_idxs, unlabeled_idxs, self.batch_size, 
            self.batch_size-self.labeled_bs)
        self.dataloader = DataLoader(self.dataset, batch_sampler=batch_sampler,
                             num_workers=4, pin_memory=True, 
                             worker_init_fn=self._worker_init_fn)

        self.dataloader_val = DataLoader(self.dataset_val, batch_size=1, 
                                         shuffle=False,num_workers=1)
    
        
    def _train_CTCT(self):
        """
        code for "Semi-Supervised Medical Image Segmentation via Cross Teaching 
        between CNN and Transformer"
        """
        print("================> Training CTCT<===============")
        iter_num = 0
        max_epoch = self.max_iterations // len(self.dataloader) + 1
        best_performance1 = 0.0
        best_performance2 = 0.0
        iterator = tqdm(range(max_epoch), ncols=70)
        for epoch_num in iterator:
            for i_batch, sampled_batch in enumerate(self.dataloader):
                volume_batch, label_batch = ( sampled_batch['image'], 
                                              sampled_batch['label'])
                volume_batch, label_batch = volume_batch.cuda(), label_batch.cuda()

                outputs1 = self.model(volume_batch)
                outputs_soft1 = torch.softmax(outputs1, dim=1)

                outputs2 = model2(volume_batch)
                outputs_soft2 = torch.softmax(outputs2, dim=1)
                consistency_weight = get_current_consistency_weight(
                    iter_num // 150)

                loss1 = 0.5 * (ce_loss(outputs1[:args.labeled_bs], label_batch[:args.labeled_bs].long()) + dice_loss(
                    outputs_soft1[:args.labeled_bs], label_batch[:args.labeled_bs].unsqueeze(1)))
                loss2 = 0.5 * (ce_loss(outputs2[:args.labeled_bs], label_batch[:args.labeled_bs].long()) + dice_loss(
                    outputs_soft2[:args.labeled_bs], label_batch[:args.labeled_bs].unsqueeze(1)))

                pseudo_outputs1 = torch.argmax(
                    outputs_soft1[args.labeled_bs:].detach(), dim=1, keepdim=False)
                pseudo_outputs2 = torch.argmax(
                    outputs_soft2[args.labeled_bs:].detach(), dim=1, keepdim=False)

                pseudo_supervision1 = dice_loss(
                    outputs_soft1[args.labeled_bs:], pseudo_outputs2.unsqueeze(1))
                pseudo_supervision2 = dice_loss(
                    outputs_soft2[args.labeled_bs:], pseudo_outputs1.unsqueeze(1))

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
                for param_group in optimizer1.param_groups:
                    param_group['lr'] = lr_
                for param_group in optimizer2.param_groups:
                    param_group['lr'] = lr_

                writer.add_scalar('lr', lr_, iter_num)
                writer.add_scalar(
                    'consistency_weight/consistency_weight', consistency_weight, iter_num)
                writer.add_scalar('loss/model1_loss',
                                model1_loss, iter_num)
                writer.add_scalar('loss/model2_loss',
                                model2_loss, iter_num)
                logging.info('iteration %d : model1 loss : %f model2 loss : %f' % (
                    iter_num, model1_loss.item(), model2_loss.item()))
                if iter_num % 50 == 0:
                    image = volume_batch[1, 0:1, :, :]
                    writer.add_image('train/Image', image, iter_num)
                    outputs = torch.argmax(torch.softmax(
                        outputs1, dim=1), dim=1, keepdim=True)
                    writer.add_image('train/model1_Prediction',
                                    outputs[1, ...] * 50, iter_num)
                    outputs = torch.argmax(torch.softmax(
                        outputs2, dim=1), dim=1, keepdim=True)
                    writer.add_image('train/model2_Prediction',
                                    outputs[1, ...] * 50, iter_num)
                    labs = label_batch[1, ...].unsqueeze(0) * 50
                    writer.add_image('train/GroundTruth', labs, iter_num)

                if iter_num > 800 and iter_num % 200 == 0:
                    model1.eval()
                    metric_list = 0.0
                    for i_batch, sampled_batch in enumerate(valloader):
                        metric_i = test_single_volume(
                            sampled_batch["image"], sampled_batch["label"], model1, classes=num_classes, patch_size=args.patch_size)
                        metric_list += np.array(metric_i)
                    metric_list = metric_list / len(db_val)
                    for class_i in range(num_classes-1):
                        writer.add_scalar('info/model1_val_{}_dice'.format(class_i+1),
                                        metric_list[class_i, 0], iter_num)
                        writer.add_scalar('info/model1_val_{}_hd95'.format(class_i+1),
                                        metric_list[class_i, 1], iter_num)

                    performance1 = np.mean(metric_list, axis=0)[0]

                    mean_hd951 = np.mean(metric_list, axis=0)[1]
                    writer.add_scalar('info/model1_val_mean_dice',
                                    performance1, iter_num)
                    writer.add_scalar('info/model1_val_mean_hd95',
                                    mean_hd951, iter_num)

                    if performance1 > best_performance1:
                        best_performance1 = performance1
                        save_mode_path = os.path.join(snapshot_path,
                                                    'model1_iter_{}_dice_{}.pth'.format(
                                                        iter_num, round(best_performance1, 4)))
                        save_best = os.path.join(snapshot_path,
                                                '{}_best_model1.pth'.format(args.model))
                        torch.save(model1.state_dict(), save_mode_path)
                        torch.save(model1.state_dict(), save_best)

                    logging.info(
                        'iteration %d : model1_mean_dice : %f model1_mean_hd95 : %f' % (iter_num, performance1, mean_hd951))
                    model1.train()

                    model2.eval()
                    metric_list = np.zeros((num_classes,2))
                    # for i_batch, sampled_batch in enumerate(valloader):
                    #     metric_i = test_single_volume(
                    #         sampled_batch["image"], sampled_batch["label"], model2, classes=num_classes, patch_size=args.patch_size)
                    #     metric_list += np.array(metric_i)
                    metric_list = metric_list / len(db_val)
                    for class_i in range(num_classes-1):
                        writer.add_scalar('info/model2_val_{}_dice'.format(class_i+1),
                                        metric_list[class_i, 0], iter_num)
                        writer.add_scalar('info/model2_val_{}_hd95'.format(class_i+1),
                                        metric_list[class_i, 1], iter_num)

                    performance2 = np.mean(metric_list, axis=0)[0]

                    mean_hd952 = np.mean(metric_list, axis=0)[1]
                    writer.add_scalar('info/model2_val_mean_dice',
                                    performance2, iter_num)
                    writer.add_scalar('info/model2_val_mean_hd95',
                                    mean_hd952, iter_num)

                    if performance2 > best_performance2:
                        best_performance2 = performance2
                        save_mode_path = os.path.join(snapshot_path,
                                                    'model2_iter_{}_dice_{}.pth'.format(
                                                        iter_num, round(best_performance2, 4)))
                        save_best = os.path.join(snapshot_path,
                                                '{}_best_model2.pth'.format(args.model))
                        torch.save(model2.state_dict(), save_mode_path)
                        torch.save(model2.state_dict(), save_best)

                    logging.info(
                        'iteration %d : model2_mean_dice : %f model2_mean_hd95 : %f' % (iter_num, performance2, mean_hd952))
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
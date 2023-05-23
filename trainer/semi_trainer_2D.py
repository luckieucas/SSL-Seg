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
from scipy.ndimage import zoom

from dataset.BCVData import BCVDataset, BCVDatasetCAC,DatasetSR
from dataset.dataset import DatasetSemi
from dataset.sampler import BatchSampler, ClassRandomSampler
from dataset.dataset import TwoStreamBatchSampler
from dataset.dataset_old import (BaseDataSets, RandomGenerator,
                                 TwoStreamBatchSampler)
from trainer.semi_trainer import SemiSupervisedTrainerBase
from networks.net_factory import net_factory
from networks.vision_transformer import SwinUnet as ViT_seg
from config import get_config
from arguments import Namespace



class SemiSupervisedTrainer2D(SemiSupervisedTrainerBase):
    def __init__(self, config, output_folder=None, logging=None, root_path=None) -> None:
        SemiSupervisedTrainerBase.__init__(self, config, output_folder, logging)
        self.root_path = root_path
        self.dataset_val = None
        self.dataloader_val = None
        parser = argparse.ArgumentParser()
        args = parser.parse_args()
        args.cfg = "./configs/swin_tiny_patch4_window7_224_lite.yaml"
        args.opts = None
        args.batch_size = self.batch_size
        args.zip = True
        args.resume = self.continue_training
        args.cache_mode = self.method_config['cache_mode']
        args.accumulation_steps = self.method_config['accumulation_steps']
        args.use_checkpoint = self.method_config['use_checkpoint']
        args.amp_opt_level = self.method_config['amp_opt_level']
        args.tag = self.method_config['tag']
        args.eval = self.method_config['eval']
        args.throughput = self.method_config['throughput']
        for key, value in Namespace(config).__dict__.items():
            vars(args)[key] = value
        self.args = args
    
    
    def load_checkpoint(self, fname, train=True):
        checkpoint = torch.load(fname)
        self.model.load_state_dict(checkpoint)
        self.model.to(self.device)
        
    
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
                                class_num=self.num_classes,
                                device=self.device)
            if ema:
                for param in model.parameters():
                    param.detach_()
            return model

        self.model = create_model()
        config = get_config(self.args)
        self.model2 = ViT_seg(config, img_size=self.patch_size,
                        num_classes=self.num_classes).to(self.device)
        self.model2.load_from(config)
    
    def __patients_to_slices(self,dataset, patiens_num):
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
    
    def _test_single_volume(self, net, image, label, classes, patch_size=[256, 256]):
        image, label = image.squeeze(0).cpu().detach(
        ).numpy(), label.squeeze(0).cpu().detach().numpy()
        prediction = np.zeros_like(label)
        for ind in range(image.shape[0]):
            slice = image[ind, :, :]
            x, y = slice.shape[0], slice.shape[1]
            slice = zoom(slice, (patch_size[0] / x, patch_size[1] / y), order=0)
            input = torch.from_numpy(slice).unsqueeze(
                0).unsqueeze(0).float().to(self.device)
            net.eval()
            with torch.no_grad():
                out = torch.argmax(torch.softmax(
                    net(input), dim=1), dim=1).squeeze(0)
                out = out.cpu().detach().numpy()
                pred = zoom(out, (x / patch_size[0], y / patch_size[1]), order=0)
                prediction[ind] = pred
        metric_list = []
        for i in range(1, classes):
            metric_list.append(self._calculate_metric(
                prediction == i, label == i, cal_asd=True))
        return metric_list
    
    def train(self):
        if self.method_name=="CTCT":
            self._train_CTCT()
    
    def evaluation(self, model,model_name="model"):
        metric_list = 0.0
        for i_batch, sampled_batch in enumerate(self.dataloader_val):
            volume_batch, label_batch = ( 
                    sampled_batch['image'], sampled_batch['label']
                )
            metric_i = self._test_single_volume(model,volume_batch, 
                                                label_batch, 
                                                classes=self.num_classes, 
                                                patch_size=self.patch_size)
            metric_list += np.array(metric_i)
        metric_list = metric_list / len(self.dataset_val)
        for class_i in range(self.num_classes-1):
            self.tensorboard_writer.add_scalar(
                f'info/{model_name}_val_{class_i+1}_dice',
                metric_list[class_i, 0], self.current_iter
            )
            self.tensorboard_writer.add_scalar(
                f'info/{model_name}_val_{class_i+1}_hd95',
                metric_list[class_i, 1], self.current_iter
            )
        performance = np.mean(metric_list, axis=0)[0]
        mean_hd95 = np.mean(metric_list,axis=0)[1]
        self.tensorboard_writer.add_scalar(f'mean/{model_name}_val_mean_dice',
                                  performance, self.current_iter)
        self.tensorboard_writer.add_scalar(f'mean/{model_name}_val_mean_hd',
                                  mean_hd95, self.current_iter)
        best_performance = self.best_performance
        if model_name=="model2":
            best_performance = self.best_performance2
        if performance > best_performance:
            if model_name == "model2":
                self.best_performance2 = performance
            else:
                self.best_performance = performance
            save_mode_path = os.path.join(self.output_folder,
                                          '{}_iter_{}_dice_{}.pth'.format(
                                            model_name,
                                            self.current_iter, 
                                            round(performance, 4)))
            save_best = os.path.join(self.output_folder,
                                     '{}_best_{}.pth'.format(
                                        self.backbone,model_name))
            torch.save(model.state_dict(), save_mode_path)
            torch.save(model.state_dict(), save_best)
            
        
    def _train_CTCT(self):
        """
        code for "Semi-Supervised Medical Image Segmentation via Cross Teaching 
        between CNN and Transformer"
        """
        print("================> Training CTCT<===============")
        iter_num = 0
        max_epoch = self.max_iterations // len(self.dataloader) + 1
        iterator = tqdm(range(max_epoch), ncols=70)
        for epoch_num in iterator:
            for i_batch, sampled_batch in enumerate(self.dataloader):
                self._adjust_learning_rate()
                self.model.train()
                self.model2.train()
                volume_batch, label_batch = ( 
                    sampled_batch['image'], sampled_batch['label']
                )
                volume_batch, label_batch = (
                    volume_batch.to(self.device), label_batch.to(self.device)
                )

                outputs1 = self.model(volume_batch)
                outputs_soft1 = torch.softmax(outputs1, dim=1)

                outputs2 = self.model2(volume_batch)
                outputs_soft2 = torch.softmax(outputs2, dim=1)
                self.consistency_weight = (
                    self._get_current_consistency_weight(self.current_iter//150)
                )

                loss1 = 0.5 * (self.ce_loss(
                                    outputs1[:self.labeled_bs], 
                                    label_batch[:self.labeled_bs].long()
                                ) + 
                               self.dice_loss(
                                   outputs_soft1[:self.labeled_bs], 
                                   label_batch[:self.labeled_bs].unsqueeze(1)
                                ))
                loss2 = 0.5 * (self.ce_loss(
                                    outputs2[:self.labeled_bs], 
                                            label_batch[:self.labeled_bs].long()
                                ) + self.dice_loss(
                                     outputs_soft2[:self.labeled_bs], 
                                     label_batch[:self.labeled_bs].unsqueeze(1))
                    )

                pseudo_outputs1 = torch.argmax(
                    outputs_soft1[self.labeled_bs:].detach(), dim=1, keepdim=False)
                pseudo_outputs2 = torch.argmax(
                    outputs_soft2[self.labeled_bs:].detach(), dim=1, keepdim=False)

                pseudo_supervision1 = self.dice_loss(
                    outputs_soft1[self.labeled_bs:], pseudo_outputs2.unsqueeze(1))
                pseudo_supervision2 = self.dice_loss(
                    outputs_soft2[self.labeled_bs:], pseudo_outputs1.unsqueeze(1))

                model1_loss = loss1 + self.consistency_weight * pseudo_supervision1
                model2_loss = loss2 + self.consistency_weight * pseudo_supervision2

                self.loss = model1_loss + model2_loss

                self.optimizer.zero_grad()
                self.optimizer2.zero_grad()

                self.loss.backward()

                self.optimizer.step()
                self.optimizer2.step()

                self.current_iter = self.current_iter + 1

                self._add_information_to_writer()
                if (self.current_iter > self.began_eval_iter and
                    self.current_iter % self.val_freq == 0
                ) or self.current_iter==20:
                    self.evaluation(model=self.model)
                    self.evaluation(model=self.model2, model_name="vit")
                if self.current_iter % self.save_checkpoint_freq == 0:
                    self._save_checkpoint()
                if self.current_iter >= self.max_iterations:
                    break
            if self.current_iter >= self.max_iterations:
                iterator.close()
                break
        self.tensorboard_writer.close()
        print("*"*10,"training done!","*"*10)
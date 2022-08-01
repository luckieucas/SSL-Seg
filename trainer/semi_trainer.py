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

from utils import losses,ramps
from dataset.BCVData import BCVDataset, BCVDatasetCAC
from dataset.dataset import DatasetSemi
from networks.net_factory_3d import net_factory_3d
from dataset.brats2019 import (BraTS2019, CenterCrop, RandomCrop,
                                   RandomRotFlip, ToTensor,
                                   TwoStreamBatchSampler)
from val_3D import test_all_case,test_all_case_BCV
from unet3d.losses import DiceLoss #test loss




class SemiSupervisedTrainer:
    def __init__(self, config, output_folder, logging) -> None:
        self.config = config
        self.device = torch.device(f"cuda:{config['gpu']}")
        self.output_folder = output_folder
        self.logging = logging
        self.exp = config['exp']
        self.FP16 = config['FP16']
        self.weight_decay = config['weight_decay']
        self.lr_scheduler_eps = config['lr_scheduler_eps']
        self.lr_scheduler_patience = config['lr_scheduler_patience']
        self.seed = config['seed']
        self.train_data_path = config['root_path']
        self.initial_lr = config['initial_lr']
        self.optimizer_type = config['optimizer_type']
        self.backbone = config['backbone']
        self.train_data_path = config['root_path']
        self.max_iterations = config['max_iterations']
        self.began_semi_iter = config['began_semi_iter']
        self.ema_decay = config['ema_decay']
        self.began_condition_iter = config['began_condition_iter']
        self.began_eval_iter = config['began_eval_iter']
        self.show_img_freq = config['show_img_freq']
        self.save_checkpoint_freq = config['save_checkpoint_freq']
        self.val_freq = config['val_freq']

        # config for training from checkpoint
        self.continue_training = config['continue_training']
        self.network_checkpoint = config['model_checkpoint']
        self.network2_checkpoint = config['model2_checkpoint'] # for CPS based methods

        # config for semi-supervised
        self.consistency_rampup = config['consistency_rampup']
        self.consistency = config['consistency']
        
        # config for dataset
        dataset = config['DATASET']
        self.patch_size = dataset['patch_size']
        self.labeled_num = dataset['labeled_num']

        self.batch_size = dataset['batch_size']
        self.labeled_bs = dataset['labeled_bs']
        self.cutout = dataset['cutout']
        self.affine_trans = dataset['affine_trans']
        self.random_rotflip = dataset['random_rotflip']
        self.edge_prob = dataset['edge_prob']
        self.dataset_name = config['dataset_name']
        
        dataset_config = dataset[self.dataset_name]
        self.num_classes = dataset_config['num_classes']
        self.training_data_num = dataset_config['training_data_num']
        self.testing_data_num = dataset_config['testing_data_num']
        self.train_list = dataset_config['train_list']
        self.test_list = dataset_config['test_list']
        self.cut_upper = dataset_config['cut_upper']
        self.cut_lower = dataset_config['cut_lower']

        # config for method
        self.method_name = config['method']
        self.method_config = config['METHOD'][self.method_name]
        self.use_CAC = config['use_CAC']

        self.experiment_name = None
        self.ce_loss = nn.CrossEntropyLoss()
        self.dice_loss = losses.DiceLoss(self.num_classes)
        self.dice_loss_con = losses.DiceLoss(2)
        self.dice_loss2 = DiceLoss(normalization='softmax')
        self.best_performance = 0.0
        self.best_performance2 = 0.0 # for CPS based methods
        self.current_iter = 0
        self.network = None
        self.ema_network = None # for MT based methods
        self.network2 = None # for CPS based methods
        self.scaler = None 
        self.scaler2 = None

        self.dataset = None
        self.dataloader = None
        self.wandb_logger = None
        self.tensorboard_writer = None
        self.labeled_idxs = None 
        self.unlabeled_idxs = None

        # generate by training process
        self.current_lr = self.initial_lr
        self.loss = None
        self.loss_ce = None 
        self.loss_dice = None
        self.loss_dice_con = None # for conditional network
        self.consistency_loss = None
        self.consistency_weight = None

    
    def initialize(self):
        self.experiment_name = f"{self.dataset_name}_{self.method_name}_"\
                               f"{self.backbone}_labeled{self.labeled_num}_"\
                               f"{self.optimizer_type}_{self.exp}"
        self.wandb_logger = wandb.init(name=self.experiment_name,
                                        project="semi-supervised-segmentation",
                                        config = self.config)
        wandb.tensorboard.patch(root_logdir=self.output_folder + '/log')
        self.tensorboard_writer = SummaryWriter(self.output_folder + '/log')
        self.load_dataset()
        self.initialize_optimizer_and_scheduler()
    
    def initialize_network(self):
        self.network = net_factory_3d(net_type=self.backbone,in_chns=1, 
                                      class_num=self.num_classes,
                                      model_config=self.config['model'],
                                      device=self.device)
        if self.continue_training:
            model_state_dict = torch.load(self.network_checkpoint)
            self.network.load_state_dict(model_state_dict)
            self.current_iter = self.config['current_iter_num']
            print(f"====>sucessfully load model from{self.network_checkpoint}")
        else:
            self._xavier_normal_init_weight()
        if self.method_name in ['MT','UAMT']:
            self.ema_network = net_factory_3d(net_type=self.backbone,in_chns=1, 
                                          class_num=self.num_classes,
                                          model_config=self.config['model'],
                                          device=self.device)
            for param in self.ema_network.parameters():
                param.detach_()
        elif self.method_name == 'CPS':
            self.network2 = net_factory_3d(net_type=self.backbone,in_chns=1, 
                                      class_num=self.num_classes,
                                      model_config=self.config['model'],
                                      device=self.device)
            if self.continue_training:
                model2_state_dict = torch.load(self.network2_checkpoint)
                self.network2.load_state_dict(model2_state_dict)
                print(f"sucessfully load model from{self.network2_checkpoint}")
            else:
                self._kaiming_normal_init_weight()
        elif self.method_name == 'C3PS':
            self.network2 = net_factory_3d("unet_3D_condition", in_chns=1, 
                                           class_num=2,device=self.device)
            if self.continue_training:
                model2_state_dict = torch.load(self.network2_checkpoint)
                self.network2.load_state_dict(model2_state_dict)
                print(f"sucessfully load model from{self.network2_checkpoint}")
            else:
                self._kaiming_normal_init_weight()

    def load_checkpoint(self, fname, train=True):
        pass

    def load_dataset(self):
        train_supervised = False
        if self.method_name == 'Baseline':
            train_supervised = True

        self.dataset = DatasetSemi(img_list_file=self.train_list, 
                                        cutout=self.cutout,
                                        affine_trans=self.affine_trans, 
                                        random_rotflip=self.random_rotflip,
                                        patch_size=self.patch_size, 
                                        num_class=self.num_classes, 
                                        edge_prob=self.edge_prob,
                                        upper=self.cut_upper,
                                        lower=self.cut_lower,
                                        labeled_num=self.labeled_num,
                                        train_supervised=train_supervised)
        if self.method_name == 'C3PS':
            self.dataset = BCVDatasetCAC(
                img_list_file=self.train_list,
                patch_size=self.patch_size,
                num_class=self.num_classes,
                stride=self.method_config['stride'],
                iou_bound=[self.method_config['iou_bound_low'],
                           self.method_config['iou_bound_high']],
                labeled_num=self.labeled_num,
                cutout=self.cutout,
                affine_trans=self.affine_trans,
                upper=self.cut_upper,
                lower=self.cut_lower
            )

    def get_dataloader(self):
        self.dataloader = DataLoader(self.dataset, batch_size=self.batch_size, 
                                     shuffle=True, num_workers=4, 
                                     pin_memory=True)

    def initialize_optimizer_and_scheduler(self):
        assert self.network is not None, "self.initialize_network must be called first"
        self.scaler = amp.GradScaler()
        if self.optimizer_type == "Adam":
            self.optimizer = torch.optim.Adam(self.network.parameters(), 
                                          self.initial_lr, 
                                          weight_decay=self.weight_decay,
                                          amsgrad=True)
        
        # self.optimizer = torch.optim.Adam(self.network.parameters(), lr=1e-3, 
        #                                   weight_decay=self.weight_decay)
        elif self.optimizer_type == 'SGD':
            self.optimizer = torch.optim.SGD(self.network.parameters(), 
                                            lr=self.initial_lr, momentum=0.9, 
                                            weight_decay=self.weight_decay)
        else:
            print("unrecognized optimizer, use Adam instead")
            self.optimizer = torch.optim.Adam(self.network.parameters(), 
                                          self.initial_lr, 
                                          weight_decay=self.weight_decay,
                                          amsgrad=True)
        
        if self.method_name in ['CPS', 'C3PS']:
            # self.optimizer2 = torch.optim.Adam(self.network2.parameters(),
            #                                    lr=1e-3, 
            #                                    weight_decay=self.weight_decay)
            self.scaler2 = amp.GradScaler()
            if self.optimizer_type == 'Adam':
                self.optimizer2 = torch.optim.Adam(
                    self.network2.parameters(), 
                    self.initial_lr, 
                    weight_decay=self.weight_decay,
                    amsgrad=True)
            elif self.optimizer_type == 'SGD':
                self.optimizer2 = torch.optim.SGD(self.network2.parameters(), 
                    lr=self.initial_lr, momentum=0.9, 
                    weight_decay=self.weight_decay)
            else:
                print("unrecognized optimizer type, use adam instead!")
                self.optimizer = torch.optim.Adam(
                    self.network2.parameters(), 
                    self.initial_lr, 
                    weight_decay=self.weight_decay,
                    amsgrad=True
                )
        self.lr_scheduler = lr_scheduler.ReduceLROnPlateau(
            self.optimizer, 
            mode='min', factor=0.2,
            patience=self.lr_scheduler_patience,
            verbose=True, threshold=self.lr_scheduler_eps,
            threshold_mode="abs")
    
    def train(self):
        self.labeled_idxs = list(range(0, self.labeled_num))
        self.unlabeled_idxs = list(range(self.labeled_num, self.training_data_num))
        if self.method_name == "Baseline":
            self.dataloader = DataLoader(self.dataset, 
                                         batch_size=self.batch_size,
                                         shuffle=True, num_workers=2,
                                         pin_memory=True)
            self.max_epoch = self.max_iterations // len(self.dataloader) + 1
            print(f"max epochs:{self.max_epoch}, max iterations:{self.max_iterations}")
            print(f"len dataloader:{len(self.dataloader)}")
            self._train_baseline_new()
        else:           
            batch_sampler = TwoStreamBatchSampler(self.labeled_idxs, 
                                            self.unlabeled_idxs,
                                            self.batch_size, 
                                            self.batch_size-self.labeled_bs)
            self.dataloader = DataLoader(self.dataset, batch_sampler=batch_sampler,
                            num_workers=4, pin_memory=True)
            self.max_epoch = self.max_iterations // len(self.dataloader) + 1
            if self.method_name == 'UAMT':
                self._train_UAMT()
            elif self.method_name == 'MT':
                self._train_MT()
            elif self.method_name == 'CPS':
                self._train_CPS()
            elif self.method_name == 'C3PS':
                if self.FP16:
                    self._train_C3PS_FP16()
                else:
                    self._train_C3PS()
            elif self.method_name == 'DAN':
                self._train_DAN()
            elif self.method_name == 'URPC':
                self._train_URPC()
            elif self.method_name == 'EM':
                self._train_EM()
            else:
                print(f"no such method {self.method_name}")
                sys.exit(0)
    
    def evaluation(self, model, do_condition=False):
        print("began evaluation!")
        model.eval()
        if do_condition:
            best_performance = self.best_performance2
            model_name = "model2"
        else:
            best_performance = self.best_performance
            model_name = "model"
        test_num = self.testing_data_num // 2
        if self.current_iter % 1000==0:
            test_num = self.testing_data_num

        avg_metric = test_all_case_BCV(model,
                                       test_list=self.test_list,
                                       num_classes=self.num_classes,
                                       patch_size=self.patch_size,
                                       stride_xy=64, stride_z=64,
                                       cut_upper=self.cut_upper,
                                       cut_lower=self.cut_lower,
                                       do_condition=do_condition,
                                       test_num=test_num,
                                       method=self.method_name.lower())
        if avg_metric[:, 0].mean() > best_performance:
            best_performance = avg_metric[:, 0].mean()
            save_mode_path = os.path.join(self.output_folder,
                                          '{}_iter_{}_dice_{}.pth'.format(
                                            model_name,
                                            self.current_iter, 
                                            round(best_performance, 4)))
            save_best = os.path.join(self.output_folder,
                                     '{}_best_{}.pth'.format(
                                        self.backbone,model_name))
            torch.save(model.state_dict(), save_mode_path)
            torch.save(model.state_dict(), save_best)

        self.tensorboard_writer.add_scalar(f'info/{model_name}_val_dice_score',
                        avg_metric[:, 0].mean(), self.current_iter)
        self.tensorboard_writer.add_scalar(f'info/{model_name}val_hd95',
                        avg_metric[:, 1].mean(), self.current_iter)
        self.logging.info(
            'iteration %d : %s_dice_score : %f %s_hd95 : %f' % (
                self.current_iter, 
                model_name,
                avg_metric[:, 0].mean(), 
                model_name,
                avg_metric[:, 1].mean()))
        return avg_metric

    def _train_baseline_new(self):
        print("================> Training Baseline New<===============")
        optimizer = torch.optim.SGD(self.network.parameters(), lr=0.01,
                        momentum=0.9, weight_decay=0.0001)
        iterator = tqdm(range(self.max_epoch), ncols=70)
        for epoch_num in iterator:
            dice_loss_list = []
            ce_loss_list = []
            for i_batch, sampled_batch in enumerate(self.dataloader):
                self.network.train()
                img, mask = (sampled_batch['image'], 
                                sampled_batch['label'].long())
                img = img.cuda()
                mask = mask.cuda()
                output = self.network(img)
                dice_loss = self.dice_loss2(output, mask)
            
                #ce_loss = CE_loss(output, mask)
                mask_argmax = torch.argmax(mask, dim=1)
                ce_loss = self.ce_loss(output, mask_argmax)
                loss =  dice_loss  + ce_loss
                dice_loss_list.append(dice_loss.item())
                ce_loss_list.append(ce_loss.item())
                #ce_loss_list.append(ce_loss.item())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                self.current_iter +=1
                lr_ = 0.01 * (1.0 - self.current_iter / 30000) ** 0.9
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr_
                print("iter:{}, lr:{}, dice loss:{}, ce loss:{}".format(
                    self.current_iter, lr_, dice_loss.item(),ce_loss.item())
                )
                if ((self.current_iter>1000 and self.current_iter % 200 == 0) or 
                    (self.current_iter<1000 and self.current_iter % 400 == 0)
                ):
                    self.network.eval()
                    avg_metric = test_all_case_BCV(
                        self.network,
                        test_list=self.test_list,
                        num_classes=self.num_classes,
                        patch_size=self.patch_size,
                        stride_xy=64,
                        stride_z=64,
                        cut_lower=self.cut_lower,
                        cut_upper=self.cut_upper
                    )
                    if avg_metric[:, 0].mean() > self.best_performance:
                        self.best_performance = avg_metric[:, 0].mean()
                        save_model_path = os.path.join(
                            self.output_folder,'model_iter_{}_dice_{}.pth'.format(
                                self.current_iter, round(self.best_performance, 4)
                            )
                        )
                        torch.save(self.network.state_dict(), save_model_path)
                    print(
                        f"iter:{self.current_iter},dice:{avg_metric[:, 0].mean()}"
                    )
                    self.wandb_logger.log(
                        {"iter:":self.current_iter,"dice": avg_metric[:, 0].mean()}
                    )
                    self.logging.info(f"iter:{self.current_iter},dice:{avg_metric[:, 0].mean()}")



    def _train_baseline(self):
        print("================> Training Baseline <===============")
        iterator = tqdm(range(self.max_epoch), ncols=70)
        for epoch_num in iterator:
            for i_batch, sampled_batch in enumerate(self.dataloader):
                self.network.train()
                volume_batch, label_batch = (sampled_batch['image'], 
                                             sampled_batch['label'].long())
                volume_batch, label_batch = (volume_batch.to(self.device), 
                                             label_batch.to(self.device))

                outputs = self.network(volume_batch)
                outputs_soft = torch.softmax(outputs, dim=1)

                label_batch = torch.argmax(label_batch, dim=1)
                self.loss_ce = self.ce_loss(outputs, label_batch.long())
                self.loss_dice = self.dice_loss(outputs_soft, 
                                                label_batch.unsqueeze(1))
                self.loss = 0.5 * (self.loss_dice + self.loss_ce)
                self.optimizer.zero_grad()
                self.loss.backward()
                self.optimizer.step()

                self._adjust_learning_rate()
                self.current_iter += 1
                self._add_information_to_writer()

                if self.current_iter % self.show_img_freq == 0:
                    image = volume_batch[0, 0:1, :, :, 20:61:10].permute(
                        3, 0, 1, 2).repeat(1, 3, 1, 1)
                    grid_image = make_grid(image, 5, normalize=True)
                    self.tensorboard_writer.add_image('train/Image', grid_image, 
                                                      self.current_iter)

                    image = outputs_soft[0, 1:2, :, :, 20:61:10].permute(
                        3, 0, 1, 2).repeat(1, 3, 1, 1)
                    grid_image = make_grid(image, 5, normalize=False)
                    self.tensorboard_writer.add_image('train/Predicted_label',
                                    grid_image, self.current_iter)

                    image = label_batch[0, :, :, 20:61:10].unsqueeze(
                        0).permute(3, 0, 1, 2).repeat(1, 3, 1, 1)
                    grid_image = make_grid(image, 5, normalize=False)
                    self.tensorboard_writer.add_image('train/Groundtruth_label',
                                    grid_image, self.current_iter)

                if (self.current_iter > self.began_eval_iter and 
                    self.current_iter % self.val_freq == 0):
                    self.evaluation(model=self.network)

                if self.current_iter % self.save_checkpoint_freq == 0:
                    self._save_checkpoint()

                if self.current_iter >= self.max_iterations:
                    break
            if self.current_iter>= self.max_iterations:
                iterator.close()
                break
        self.tensorboard_writer.close()
        print("*"*10,"training done!","*"*10)

    def _train_DAN(self):
        print("================> Training DAN <===============")
        self.network2 = net_factory_3d(net_type="DAN", class_num=self.num_classes)
        self.optimizer2 = torch.optim.Adam(
            self.network2.parameters(), lr=0.0001, betas=(0.9, 0.99)
        )
        iterator = tqdm(range(self.max_epoch), ncols=70)
        for epoch_num in iterator:
            for i_batch, sampled_batch in enumerate(self.dataloader):
                volume_batch, label_batch = (
                    sampled_batch['image'], sampled_batch['label']
                )
                volume_batch, label_batch = volume_batch.cuda(), label_batch.cuda()
                DAN_target = torch.tensor([1, 0]).cuda()
                self.network.train()
                self.network2.eval()

                outputs = self.network(volume_batch)
                outputs_soft = torch.softmax(outputs, dim=1)

                label_batch = torch.argmax(label_batch, dim=1)
                self.loss_ce = self.ce_loss(
                    outputs[:self.labeled_bs],label_batch[:self.labeled_bs]
                )
                self.loss_dice = self.dice_loss(
                    outputs_soft[:self.labeled_bs],
                    label_batch[:self.labeled_bs].unsqueeze(1)
                )
                supervised_loss = 0.5 * (self.loss_dice + self.loss_ce)

                self.consistency_weight = self._get_current_consistency_weight(
                    self.current_iter // 6
                )
                DAN_outputs = self.network2(
                    outputs_soft[self.labeled_bs:], 
                    volume_batch[self.labeled_bs:]
                )
                if self.current_iter > self.began_semi_iter:
                    self.consistency_loss = self.ce_loss(
                        DAN_outputs, (DAN_target[:self.labeled_bs]).long()
                    )
                else:
                    self.consistency_loss = torch.FloatTensor([0.0]).cuda()
                self.loss = supervised_loss + self.consistency_weight * \
                       self.consistency_loss
                self.optimizer.zero_grad()
                self.loss.backward()
                self.optimizer.step()

                self.network.eval()
                self.network2.train()
                with torch.no_grad():
                    outputs = self.network(volume_batch)
                    outputs_soft = torch.softmax(outputs, dim=1)
                
                DAN_outputs = self.network2(outputs_soft, volume_batch)
                DAN_loss = self.ce_loss(DAN_outputs, DAN_target.long())
                self.optimizer2.zero_grad()
                DAN_loss.backward()
                self.optimizer2.step()

                self._adjust_learning_rate()
                self.current_iter += 1
                self._add_information_to_writer()

                if self.current_iter % self.show_img_freq == 0:
                    image = volume_batch[0, 0:1, :, :, 20:61:10].permute(
                    3, 0, 1, 2).repeat(1, 3, 1, 1)
                    grid_image = make_grid(image, 5, normalize=True)
                    self.tensorboard_writer.add_image(
                        'train/Image', grid_image, self.current_iter
                    )

                    image = outputs_soft[0, 1:2, :, :, 20:61:10].permute(
                        3, 0, 1, 2).repeat(1, 3, 1, 1)
                    grid_image = make_grid(image, 5, normalize=False)
                    self.tensorboard_writer.add_image('train/Predicted_label',
                                    grid_image, self.current_iter)

                    image = label_batch[0, :, :, 20:61:10].unsqueeze(
                        0).permute(3, 0, 1, 2).repeat(1, 3, 1, 1)
                    grid_image = make_grid(image, 5, normalize=False)
                    self.tensorboard_writer.add_image('train/Groundtruth_label',
                                    grid_image, self.current_iter)
                if (
                    self.current_iter > self.began_eval_iter and 
                    self.current_iter % self.val_freq == 0
                ):
                    self.evaluation(model=self.network)
                if self.current_iter % self.save_checkpoint_freq == 0:
                    self._save_checkpoint()
                if self.current_iter >= self.max_iterations:
                    break 
            if self.current_iter >= self.max_iterations:
                iterator.close()
                break 
        self.tensorboard_writer.close()
        print("Training done!")



    def _train_URPC(self):
        print("================> Training URPC <===============")
        self.network = net_factory_3d(net_type='URPC', class_num=self.num_classes)
        iterator = tqdm(range(self.max_epoch), ncols=70)
        kl_distance = nn.KLDivLoss(reduction='none')
        for epoch_num in iterator:
            for i_batch, sampled_batch in enumerate(self.dataloader):
                volume_batch, label_batch = (
                    sampled_batch['image'], sampled_batch['label']
                )
                volume_batch, label_batch = volume_batch.cuda(), label_batch.cuda()
                label_batch = torch.argmax(label_batch, dim=1)
                outputs_aux1, outputs_aux2, outputs_aux3, outputs_aux4 = (
                    self.network(volume_batch)
                )
                outputs_aux1_soft = torch.softmax(outputs_aux1, dim=1)
                outputs_aux2_soft = torch.softmax(outputs_aux2, dim=1)
                outputs_aux3_soft = torch.softmax(outputs_aux3, dim=1)
                outputs_aux4_soft = torch.softmax(outputs_aux4, dim=1)

                loss_ce_aux1 = self.ce_loss(
                    outputs_aux1[:self.labeled_bs], label_batch[:self.labeled_bs]
                )
                loss_ce_aux2 = self.ce_loss(
                    outputs_aux2[:self.labeled_bs], label_batch[:self.labeled_bs]
                )
                loss_ce_aux3 = self.ce_loss(
                    outputs_aux3[:self.labeled_bs], label_batch[:self.labeled_bs]
                )
                loss_ce_aux4 = self.ce_loss(
                    outputs_aux4[:self.labeled_bs], label_batch[:self.labeled_bs]
                )

                loss_dice_aux1 = self.dice_loss(
                    outputs_aux1_soft[:self.labeled_bs], 
                    label_batch[:self.labeled_bs].unsqueeze(1)
                )
                loss_dice_aux2 = self.dice_loss(
                    outputs_aux2_soft[:self.labeled_bs], 
                    label_batch[:self.labeled_bs].unsqueeze(1)
                )
                loss_dice_aux3 = self.dice_loss(
                    outputs_aux3_soft[:self.labeled_bs], 
                    label_batch[:self.labeled_bs].unsqueeze(1)
                )
                loss_dice_aux4 = self.dice_loss(
                    outputs_aux4_soft[:self.labeled_bs], 
                    label_batch[:self.labeled_bs].unsqueeze(1)
                )
                
                self.loss_ce = (
                    loss_ce_aux1+loss_ce_aux2+loss_ce_aux3+loss_ce_aux4
                ) / 4.
                self.loss_dice = (
                    loss_dice_aux1+loss_dice_aux2+loss_dice_aux3+loss_dice_aux4
                ) / 4.
                supervised_loss = (
                    loss_ce_aux1+loss_ce_aux2+loss_ce_aux3+loss_ce_aux4+
                    loss_dice_aux1+loss_dice_aux2+loss_dice_aux3+loss_dice_aux4
                ) / 8.

                preds = (
                    outputs_aux1_soft + outputs_aux2_soft + outputs_aux3_soft +
                    outputs_aux4_soft
                ) / 4.

                variance_aux1 = torch.sum(
                    kl_distance(
                        torch.log(outputs_aux1_soft[self.labeled_bs:]),
                        preds[self.labeled_bs:]
                    ),
                    dim=1, keepdim=True
                )
                exp_variance_aux1 = torch.exp(-variance_aux1)

                variance_aux2 = torch.sum(
                    kl_distance(
                        torch.log(outputs_aux2_soft[self.labeled_bs:]),
                        preds[self.labeled_bs:]
                    ),
                    dim=1, keepdim=True
                )
                exp_variance_aux2 = torch.exp(-variance_aux2)

                variance_aux3 = torch.sum(
                    kl_distance(
                        torch.log(outputs_aux3_soft[self.labeled_bs:]),
                        preds[self.labeled_bs:]
                    ),
                    dim=1, keepdim=True
                )
                exp_variance_aux3 = torch.exp(-variance_aux3)

                variance_aux4 = torch.sum(
                    kl_distance(
                        torch.log(outputs_aux4_soft[self.labeled_bs:]),
                        preds[self.labeled_bs:]
                    ),
                    dim=1, keepdim=True
                )
                exp_variance_aux4 = torch.exp(-variance_aux4)

                self.consistency_weight = self._get_current_consistency_weight(
                    self.current_iter // 150
                )
                consistency_dist_aux1 = (
                    preds[self.labeled_bs:] - outputs_aux1_soft[self.labeled_bs:]
                ) ** 2
                consistency_loss_aux1 = torch.mean(
                    consistency_dist_aux1 * exp_variance_aux1
                ) / (torch.mean(exp_variance_aux1) + 1e-8) + torch.mean(variance_aux1)

                consistency_dist_aux2 = (
                    preds[self.labeled_bs:] - outputs_aux2_soft[self.labeled_bs:]
                ) ** 2
                consistency_loss_aux2 = torch.mean(
                    consistency_dist_aux2 * exp_variance_aux2
                ) / (torch.mean(exp_variance_aux2) + 1e-8) + torch.mean(variance_aux2)

                consistency_dist_aux3 = (
                    preds[self.labeled_bs:] - outputs_aux3_soft[self.labeled_bs:]
                ) ** 2
                consistency_loss_aux3 = torch.mean(
                    consistency_dist_aux3 * exp_variance_aux3
                ) / (torch.mean(exp_variance_aux3) + 1e-8) + torch.mean(variance_aux3)

                consistency_dist_aux4 = (
                    preds[self.labeled_bs:] - outputs_aux4_soft[self.labeled_bs:]
                ) ** 2
                consistency_loss_aux4 = torch.mean(
                    consistency_dist_aux4 * exp_variance_aux4
                ) / (torch.mean(exp_variance_aux4) + 1e-8) + torch.mean(variance_aux4)
                self.consistency_loss = (
                    consistency_loss_aux1 + consistency_loss_aux2 + 
                    consistency_loss_aux3 + consistency_loss_aux4
                ) / 4.

                if self.current_iter<self.began_semi_iter:
                    self.consistency_weight = 0.0
                self.loss = supervised_loss + self.consistency_weight * self.consistency_loss
                self.optimizer.zero_grad()
                self.loss.backward()
                self.optimizer.step()

                self._adjust_learning_rate()
                self.current_iter += 1
                self._add_information_to_writer()
                if self.current_iter % self.show_img_freq == 0:
                    image = volume_batch[0, 0:1, :, :, 20:61:10].permute(
                    3, 0, 1, 2).repeat(1, 3, 1, 1)
                    grid_image = make_grid(image, 5, normalize=True)
                    self.tensorboard_writer.add_image(
                        'train/Image', grid_image, self.current_iter
                    )

                    image = torch.argmax(outputs_aux1_soft, dim=1, keepdim=True)[0, 0:1, :, :, 20:61:10].permute(
                        3, 0, 1, 2).repeat(1, 3, 1, 1) * 100
                    grid_image = make_grid(image, 5, normalize=False)
                    self.tensorboard_writer.add_image(
                        'train/Predicted_label',grid_image, self.current_iter
                    )

                    image = label_batch[0, :, :, 20:61:10].unsqueeze(
                        0).permute(3, 0, 1, 2).repeat(1, 3, 1, 1) * 100
                    grid_image = make_grid(image, 5, normalize=False)
                    self.tensorboard_writer.add_image(
                        'train/Groundtruth_label',grid_image, self.current_iter
                    )
                if (
                        self.current_iter > self.began_eval_iter and 
                        self.current_iter % self.val_freq ==0
                ):
                    self.evaluation(model=self.network)
                if self.current_iter % self.save_checkpoint_freq == 0:
                    self._save_checkpoint()
                if self.current_iter >= self.max_iterations:
                    break
            if self.current_iter >= self.max_iterations:
                iterator.close()
                break 
        self.tensorboard_writer.close()
        print('Training Finished')
    
    
    def _train_MT(self):
        print("================> Training MT <===============")
        iterator = tqdm(range(self.max_epoch), ncols=70)
        for epoch_num in iterator:
            for i_batch, sampled_batch in enumerate(self.dataloader):
                volume_batch, label_batch = (sampled_batch['image'], 
                                             sampled_batch['label'])
                volume_batch, label_batch = (volume_batch.to(self.device), 
                                             label_batch.to(self.device))
                unlabeled_volume_batch = volume_batch[self.labeled_bs:]
                noise = torch.clamp(torch.randn_like(
                    unlabeled_volume_batch)*0.1, -0.2, 0.2)
                ema_inputs = unlabeled_volume_batch + noise
                ema_inputs = ema_inputs.cuda() 

                outputs = self.network(volume_batch)
                outputs_soft = torch.softmax(outputs, dim=1)
                with torch.no_grad():
                    ema_output = self.ema_network(ema_inputs)
                    ema_output_soft = torch.softmax(ema_output, dim=1)
                label_batch = torch.argmax(label_batch,dim=1)
                self.loss_ce = self.ce_loss(outputs[:self.labeled_bs],
                                            label_batch[:self.labeled_bs][:])
                self.loss_dice = self.dice_loss(outputs_soft[:self.labeled_bs],
                    label_batch[:self.labeled_bs].unsqueeze(1)
                    )
                supervised_loss = 0.5 * (self.loss_dice + self.loss_ce)

                self.consistency_weight = self._get_current_consistency_weight(
                    self.current_iter // 4
                )
                if self.current_iter > self.began_semi_iter:
                    self.consistency_loss = torch.mean(
                        (outputs_soft[self.labeled_bs:] - ema_output_soft)**2
                    )
                else:
                    self.consistency_loss = torch.FloatTensor([0]).to(self.device)
                self.loss = supervised_loss + self.consistency_weight * \
                            self.consistency_loss
                self.optimizer.zero_grad()
                self.loss.backward()
                self.optimizer.step()
                self._update_ema_variables()
                self._adjust_learning_rate()
                self.current_iter += 1
                self._add_information_to_writer()
                if self.current_iter % self.show_img_freq == 0:
                    image = volume_batch[0, 0:1, :, :, 20:61:10].permute(
                        3, 0, 1, 2).repeat(1, 3, 1, 1)
                    grid_image = make_grid(image, 5, normalize=True)
                    self.tensorboard_writer.add_image('train/Image', grid_image,
                                                      self.current_iter)
                    image = outputs_soft[0, 1:2, :, :, 20:61:10].permute(
                        3, 0, 1, 2).repeat(2, 3, 1, 1)
                    grid_image = make_grid(image, 5, normalize=False)
                    self.tensorboard_writer.add_image('train/Predicted_label',
                                                      grid_image,
                                                      self.current_iter)
                    image = label_batch[0, :, :, 20:61:10].unsqueeze(0).permute(
                        3, 0, 1, 2).repeat(1, 3, 1, 1)
                    grid_image = make_grid(image, 5, normalize=False)
                    self.tensorboard_writer.add_image('train/Groundtruth_label',
                                                      grid_image,
                                                      self.current_iter)
                if (self.current_iter > self.began_eval_iter and
                    self.current_iter % self.val_freq == 0):
                    self.evaluation(model=self.network)
                if self.current_iter % self.save_checkpoint_freq == 0:
                    self._save_checkpoint()
                if self.current_iter >= self.max_iterations:
                    break
            if self.current_iter >= self.max_iterations:
                iterator.close()
                break
        self.tensorboard_writer.close()
        print("*"*10,"training done!","*"*10)

    def _train_CPS(self):
        print("================> Training CPS <===============")
        self.network2.train()
        iterator = tqdm(range(self.max_epoch), ncols=70)
        for epoch_num in iterator:
            for i_batch, sampled_batch in enumerate(self.dataloader):
                volume_batch, label_batch = (
                    sampled_batch['image'], sampled_batch['label']
                )
                volume_batch, label_batch = (
                    volume_batch.to(self.device), label_batch.to(self.device)
                )
                outputs1 = self.network(volume_batch)
                outputs_soft1 = torch.softmax(outputs1, dim=1)

                outputs2 = self.network2(volume_batch)
                outputs_soft2 = torch.softmax(outputs2, dim=1)

                self.consistency_weight = self._get_current_consistency_weight(
                    self.current_iter//4
                )
                loss1 = 0.5 * (self.ce_loss(outputs1[:self.labeled_bs],
                                   label_batch[:][:self.labeled_bs].long()) + 
                               self.dice_loss(outputs_soft1[:self.labeled_bs], 
                                             label_batch[:self.labeled_bs].\
                                                unsqueeze(1)))
                loss2 = 0.5 * (self.ce_loss(outputs2[:self.labeled_bs],
                                   label_batch[:][:self.labeled_bs].long()) + 
                               self.dice_loss(outputs_soft2[:self.labeled_bs], 
                                             label_batch[:self.labeled_bs].\
                                                unsqueeze(1)))
                pseudo_outputs1 = torch.argmax(
                    outputs_soft1[self.labeled_bs:].detach(),
                    dim=1, keepdim=False
                )
                pseudo_outputs2 = torch.argmax(
                    outputs_soft2[self.labeled_bs:].detach(),
                    dim=1, keepdim=False
                )
                if self.current_iter < self.began_semi_iter:
                    pseudo_supervision1 = torch.FloatTensor([0]).to(self.device)
                    pseudo_supervision2 = torch.FloatTensor([0]).to(self.device)
                else:
                    pseudo_supervision1 = self.ce_loss(
                        outputs1[self.labeled_bs:],
                        pseudo_outputs2
                    )
                    pseudo_supervision2 = self.ce_loss(
                        outputs2[self.labeled_bs:],
                        pseudo_outputs1
                    )
                model1_loss = loss1 + self.consistency_weight *  \
                                      pseudo_supervision1
                model2_loss = loss2 + self.consistency_weight * \
                                      pseudo_supervision2
                loss = model1_loss + model2_loss 
                self.optimizer.zero_grad()
                self.optimizer2.zero_grad()

                loss.backward()
                self.optimizer.step()
                self.optimizer2.step()

                self.current_iter += 1

                self.tensorboard_writer.add_scalar('lr', self.current_lr, 
                                               self.current_iter)
                self.tensorboard_writer.add_scalar(
                'consistency_weight/consistency_weight',self.consistency_weight, 
                self.current_iter)
                self.tensorboard_writer.add_scalar('loss/model1_loss', model1_loss, 
                                               self.current_iter)
                self.tensorboard_writer.add_scalar('loss/model2_loss', model2_loss, 
                                               self.current_iter)
                self.logging.info(
                'iteration %d : model1 loss : %f model2 loss : %f' % (
                    self.current_iter,model1_loss.item(), 
                    model2_loss.item()))
            
                if self.current_iter % self.show_img_freq == 0:
                    image = volume_batch[0, 0:1, :, :, 20:61:10].permute(
                    3, 0, 1, 2).repeat(1, 3, 1, 1)
                    grid_image = make_grid(image, 5, normalize=True)
                    self.tensorboard_writer.add_image('train/Image', grid_image, 
                                                  self.current_iter)

                    image = outputs_soft1[0, 0:1, :, :, 20:61:10].permute(
                    3, 0, 1, 2).repeat(1, 3, 1, 1)
                    grid_image = make_grid(image, 5, normalize=False)
                    self.tensorboard_writer.add_image(
                    'train/Model1_Predicted_label',
                     grid_image, self.current_iter)

                    image = outputs_soft2[0, 0:1, :, :, 20:61:10].permute(
                    3, 0, 1, 2).repeat(1, 3, 1, 1)
                    grid_image = make_grid(image, 5, normalize=False)
                    self.tensorboard_writer.add_image('train/Model2_Predicted_label',
                                 grid_image, self.current_iter)

                    image = label_batch[0, :, :, 20:61:10].unsqueeze(
                    0).permute(3, 0, 1, 2).repeat(1, 3, 1, 1)
                    grid_image = make_grid(image, 5, normalize=False)
                    self.tensorboard_writer.add_image('train/Groundtruth_label',
                                 grid_image, self.current_iter)

                if (self.current_iter > self.began_eval_iter and
                    self.current_iter % self.val_freq == 0):
                    self.evaluation(model=self.network)
                    self.evaluation(model=self.network2)
                    self.network.train()
                    self.network2.train()
            
                if self.current_iter % self.save_checkpoint_freq == 0:
                    save_mode_path = os.path.join(
                    self.output_folder, 'model1_iter_' + \
                        str(self.current_iter) + '.pth')
                    torch.save(self.network.state_dict(), save_mode_path)
                    self.logging.info("save model1 to {}".format(save_mode_path))

                    save_mode_path = os.path.join(
                        self.output_folder, 
                        'model2_iter_' + str(self.current_iter) + '.pth')
                    torch.save(self.network2.state_dict(), save_mode_path)
                    self.logging.info("save model2 to {}".format(save_mode_path))
                if self.current_iter >= self.max_iterations:
                    break 
            if self.current_iter >= self.max_iterations:
                iterator.close()
                break
        self.tensorboard_writer.close()

    def _train_C3PS_FP16(self):
        print("================> Training C3PS FP16<===============")
        self.network2.train()
        iterator = tqdm(range(self.max_epoch), ncols=70)
        iter_each_epoch = len(self.dataloader)
        for epoch_num in iterator:
            for i_batch, sampled_batch in enumerate(self.dataloader):
                volume_batch, label_batch = (
                    sampled_batch['image'],sampled_batch['label']
                )
                volume_batch, label_batch = (volume_batch.to(self.device), 
                                             label_batch.to(self.device))
                
                                    #prepare input for condition net
                condition_batch = sampled_batch['condition'].type(torch.half)
                if self.use_CAC:
                    condition_batch = torch.cat(
                        [condition_batch, condition_batch],
                        dim=0
                    )
                    condition_batch = condition_batch.to(self.device)
                ul1, br1, ul2, br2 = [], [], [], []
                labeled_idxs_batch = torch.arange(0, self.labeled_bs)
                unlabeled_idx_batch = torch.arange(self.labeled_bs, 
                                                   self.batch_size)
                if self.use_CAC:
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
                        self.batch_size,
                        self.batch_size+self.labeled_bs
                    )
                    labeled_idxs1_batch = torch.arange(0,self.labeled_bs)
                    labeled_idxs_batch = torch.cat(
                        [labeled_idxs1_batch,labeled_idxs2_batch]
                    )
                    unlabeled_idxs1_batch = torch.arange(self.labeled_bs,
                                                         self.batch_size)
                    unlabeled_idxs2_batch = torch.arange(
                        self.batch_size+self.labeled_bs, 
                        2 * self.batch_size
                    )
                    unlabeled_idxs_batch = torch.cat(
                        [unlabeled_idxs1_batch,unlabeled_idxs2_batch]
                    )
                
                with amp.autocast():
                    outputs1 = self.network(volume_batch)
                    outputs_soft1 = torch.softmax(outputs1, dim=1)

                    outputs2 = self.network2(volume_batch, condition_batch)
                    outputs_soft2 = torch.softmax(outputs2, dim=1)
                    label_batch_con = (
                        label_batch==condition_batch.unsqueeze(-1).unsqueeze(-1)
                    ).long()

                    self.consistency_weight = self._get_current_consistency_weight(
                        self.current_iter//4
                    )
                    loss1 = 0.5 * (
                        self.ce_loss(
                            outputs1[labeled_idxs_batch],
                            label_batch[labeled_idxs_batch].long()
                        ) +
                        self.dice_loss(
                            outputs_soft1[labeled_idxs_batch],
                            label_batch[labeled_idxs_batch].unsqueeze(1)
                        )
                    )
                    loss2 = 0.5 * (
                        self.ce_loss(
                            outputs2[labeled_idxs_batch],
                            label_batch_con[labeled_idxs_batch].long()
                        ) + 
                        self.dice_loss_con(
                            outputs_soft2[labeled_idxs_batch],
                            label_batch_con[labeled_idxs_batch].unsqueeze(1)
                        )
                    )

                    if self.use_CAC:
                        overlap_soft1_list = []
                        overlap_soft2_list = []
                        overlap_outputs1_list = []
                        overlap_outputs2_list = []
                        for unlabeled_idx1, unlabeled_idx2 in zip(
                            unlabeled_idxs1_batch,
                            unlabeled_idxs2_batch
                        ):
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
                            assert overlap1_soft1.shape == overlap2_soft1.shape,(
                                "overlap region size must equal"
                            )

                            # overlap region pred by model1
                            overlap1_outputs1 = outputs1[
                                unlabeled_idx1,
                                :,
                                ul1[0][1]:br1[0][1],
                                ul1[1][1]:br1[1][1],
                                ul1[2][1]:br1[2][1]
                            ] # overlap batch1
                            overlap2_outputs1 = outputs1[
                                unlabeled_idx2,
                                :,
                                ul2[0][1]:br2[0][1],
                                ul2[1][1]:br2[1][1],
                                ul2[2][1]:br2[2][1]
                            ] # overlap batch2
                            assert overlap1_outputs1.shape == overlap2_outputs1.shape,(
                                "overlap region size must equal"
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
                            assert overlap1_soft2.shape == overlap2_soft2.shape,(
                                "overlap region size must equal"
                            )
                            
                            # overlap region outputs pred by model2
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
                            assert overlap1_outputs2.shape == overlap2_outputs2.shape,(
                                "overlap region size must equal"
                            )
                            overlap_outputs2_list.append(overlap1_outputs2.unsqueeze(0))
                            overlap_outputs2_list.append(overlap2_outputs2.unsqueeze(0))

                            # merge overlap region pred
                            overlap_soft1_tmp = (overlap1_soft1 + overlap2_soft1) / 2.
                            overlap_soft2_tmp = (overlap1_soft2 + overlap2_soft2) / 2.
                            overlap_soft1_list.append(overlap_soft1_tmp.unsqueeze(0))
                            overlap_soft2_list.append(overlap_soft2_tmp.unsqueeze(0))
                        overlap_soft1 = torch.cat(overlap_soft1_list, 0)
                        overlap_soft2 = torch.cat(overlap_soft2_list, 0)
                        overlap_outputs1 = torch.cat(overlap_outputs1_list, 0)
                        overlap_outputs2 = torch.cat(overlap_outputs2_list, 0)
                    if self.current_iter < self.began_condition_iter:
                        pseudo_supervision1 = torch.FloatTensor([0]).to(self.device)
                    else:
                        if self.use_CAC:
                            overlap_pseudo_outputs2 = torch.argmax(
                                overlap_soft2.detach(),
                                dim=1,
                                keepdim=False
                            )
                            overlap_pseudo_outputs2 = torch.cat(
                                [overlap_pseudo_outputs2, overlap_pseudo_outputs2]
                            )
                            pseudo_supervision1 = self._cross_entropy_loss_con(
                                overlap_outputs1,
                                overlap_pseudo_outputs2,
                                condition_batch[unlabeled_idx_batch]
                            )
                        else:
                            pseudo_outputs2 = torch.argmax(
                                outputs_soft2[self.labeled_bs:].detach(),
                                dim=1,
                                keepdim=False
                            )
                            pseudo_supervision1 = self._cross_entropy_loss_con(
                                outputs1[self.labeled_bs:],
                                pseudo_outputs2,
                                condition_batch[self.labeled_bs:]
                            )
                    if self.current_iter < self.began_semi_iter:
                        pseudo_supervision2 = torch.FloatTensor([0]).to(self.device)
                    else:
                        if self.use_CAC:
                            overlap_pseudo_outputs1 = torch.argmax(
                                overlap_soft1.detach(), 
                                dim=1, 
                                keepdim=False
                            )
                            overlap_pseudo_outputs1 = torch.cat(
                                [overlap_pseudo_outputs1, overlap_pseudo_outputs1]
                            )
                            pseudo_supervision2 = self.ce_loss(
                                overlap_outputs2, 
                                (
                                    overlap_pseudo_outputs1==condition_batch[unlabeled_idxs_batch].unsqueeze(-1).unsqueeze(-1)
                                ).long()
                            ) 
                        else:
                            pseudo_outputs1 = torch.argmax(
                                outputs_soft1[self.labeled_bs:].detach(), 
                                dim=1, 
                                keepdim=False
                            )
                            pseudo_supervision2 = self.ce_loss(
                                outputs2[self.labeled_bs:], 
                                (pseudo_outputs1==condition_batch[self.labeled_bs:].\
                                    unsqueeze(-1).unsqueeze(-1)).long()
                            )
                    

                    model1_loss = loss1 + self.consistency_weight * pseudo_supervision1
                    model2_loss = loss2 + self.consistency_weight * pseudo_supervision2

                    loss = model1_loss + model2_loss

                self.optimizer.zero_grad()
                self.optimizer2.zero_grad() 
                self.scaler.scale(model1_loss).backward()
                self.scaler2.scale(model2_loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler2.step(self.optimizer2)
                self.scaler.update()
                self.scaler2.update()
                # loss.backward()
                # self.optimizer.step()
                # self.optimizer2.step()   

                self.current_iter += 1
                self._adjust_learning_rate()  
                self.tensorboard_writer.add_scalar('lr', self.current_lr, 
                                                   self.current_iter)
                self.tensorboard_writer.add_scalar(
                    'consistency_weight/consistency_weight', 
                    self.consistency_weight, 
                    self.current_iter
                )
                self.tensorboard_writer.add_scalar('loss/model1_loss', 
                                                   model1_loss, 
                                                   self.current_iter)
                self.tensorboard_writer.add_scalar('loss/model2_loss', 
                                                   model2_loss, 
                                                   self.current_iter)
                self.tensorboard_writer.add_scalar(
                    'loss/pseudo_supervision1',
                    pseudo_supervision1, self.current_iter
                )
                self.tensorboard_writer.add_scalar(
                    'loss/pseudo_supervision2',
                    pseudo_supervision2, 
                    self.current_iter
                )
                self.logging.info(
                    'iteration %d :'
                    'model1 loss : %f' 
                    'model2 loss : %f' 
                    'pseudo_supervision1 : %f'
                    'pseudo_supervision2 : %f' % (
                        self.current_iter, model1_loss.item(), 
                        model2_loss.item(), 
                        pseudo_supervision1.item(), 
                        pseudo_supervision2.item()
                    )
                )
                if self.current_iter % self.show_img_freq == 0:
                    image = volume_batch[0, 0:1, :, :, 20:61:10].permute(
                        3, 0, 1, 2).repeat(1, 3, 1, 1)
                    grid_image = make_grid(image, 5, normalize=True)
                    self.tensorboard_writer.add_image('train/Image', grid_image, 
                                                      self.current_iter)

                    image = outputs_soft1[0, 0:1, :, :, 20:61:10].permute(
                        3, 0, 1, 2).repeat(1, 3, 1, 1)
                    grid_image = make_grid(image, 5, normalize=False)
                    self.tensorboard_writer.add_image(
                        'train/Model1_Predicted_label',
                        grid_image, 
                        self.current_iter
                    )

                    image = outputs_soft2[0, 0:1, :, :, 20:61:10].permute(
                        3, 0, 1, 2).repeat(1, 3, 1, 1)
                    grid_image = make_grid(image, 5, normalize=False)
                    self.tensorboard_writer.add_image(
                        'train/Model2_Predicted_label',
                        grid_image, 
                        self.current_iter
                    )

                    image = label_batch[0, :, :, 20:61:10].unsqueeze(0).permute(
                        3, 0, 1, 2).repeat(1, 3, 1, 1)
                    grid_image = make_grid(image, 5, normalize=False)
                    self.tensorboard_writer.add_image('train/Groundtruth_label',
                                                      grid_image, 
                                                      self.current_iter)
                if (self.current_iter > self.began_eval_iter and
                    self.current_iter % self.val_freq == 0
                ):
                    self.evaluation(model=self.network)
                    self.evaluation(model=self.network2, do_condition=True)
                    self.network.train()
                    self.network2.train()
                if self.current_iter % self.save_checkpoint_freq == 0:
                    save_model_path = os.path.join(
                        self.output_folder,
                        'model_iter_' + str(self.current_iter) + '.pth'
                    )
                    torch.save(self.network.state_dict(), save_model_path)
                    self.logging.info(f"save model to {save_model_path}")

                    save_model_path = os.path.join(
                        self.output_folder,
                        'model2_iter_' + str(self.current_iter) + '.pth'
                    )
                    torch.save(self.network2.state_dict(), save_model_path)
                    self.logging.info(f'save model2 to {save_model_path}')
                if self.current_iter >= self.max_iterations:
                    break
            if self.current_iter >= self.max_iterations:
                iterator.close()
                break
        self.tensorboard_writer.close()


    def _train_C3PS(self):
        print("================> Training C3PS<===============")
        
        iterator = tqdm(range(self.max_epoch), ncols=70)
        iter_each_epoch = len(self.dataloader)
        for epoch_num in iterator:
            for i_batch, sampled_batch in enumerate(self.dataloader):
                self.network.train()
                self.network2.train()
                volume_batch, label_batch = (
                    sampled_batch['image'],sampled_batch['label']
                )
                volume_batch, label_batch = (volume_batch.to(self.device), 
                                             label_batch.to(self.device))
                
                                    #prepare input for condition net
                condition_batch = sampled_batch['condition'].type(torch.half)
                if self.use_CAC:
                    condition_batch = torch.cat(
                        [condition_batch, condition_batch],
                        dim=0
                    )
                    condition_batch = condition_batch.to(self.device)
                ul1, br1, ul2, br2 = [], [], [], []
                labeled_idxs_batch = torch.arange(0, self.labeled_bs)
                unlabeled_idx_batch = torch.arange(self.labeled_bs, 
                                                   self.batch_size)
                if self.use_CAC:
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
                        self.batch_size,
                        self.batch_size+self.labeled_bs
                    )
                    labeled_idxs1_batch = torch.arange(0,self.labeled_bs)
                    labeled_idxs_batch = torch.cat(
                        [labeled_idxs1_batch,labeled_idxs2_batch]
                    )
                    unlabeled_idxs1_batch = torch.arange(self.labeled_bs,
                                                         self.batch_size)
                    unlabeled_idxs2_batch = torch.arange(
                        self.batch_size+self.labeled_bs, 
                        2 * self.batch_size
                    )
                    unlabeled_idxs_batch = torch.cat(
                        [unlabeled_idxs1_batch,unlabeled_idxs2_batch]
                    )
                
                outputs1 = self.network(volume_batch)
                outputs_soft1 = torch.softmax(outputs1, dim=1)

                outputs2 = self.network2(volume_batch, condition_batch)
                outputs_soft2 = torch.softmax(outputs2, dim=1)
                label_batch_con = (
                    label_batch==condition_batch.unsqueeze(-1).unsqueeze(-1)
                ).long()

                self.consistency_weight = self._get_current_consistency_weight(
                    self.current_iter//4
                )
                loss1 = 0.5 * (
                    self.ce_loss(
                        outputs1[labeled_idxs_batch],
                        label_batch[labeled_idxs_batch].long()
                    ) +
                    self.dice_loss(
                        outputs_soft1[labeled_idxs_batch],
                        label_batch[labeled_idxs_batch].unsqueeze(1)
                    )
                )
                loss2 = 0.5 * (
                    self.ce_loss(
                        outputs2[labeled_idxs_batch],
                        label_batch_con[labeled_idxs_batch].long()
                    ) + 
                    self.dice_loss_con(
                        outputs_soft2[labeled_idxs_batch],
                        label_batch_con[labeled_idxs_batch].unsqueeze(1)
                    )
                )

                if self.use_CAC:
                    overlap_soft1_list = []
                    overlap_soft2_list = []
                    overlap_outputs1_list = []
                    overlap_outputs2_list = []
                    for unlabeled_idx1, unlabeled_idx2 in zip(
                        unlabeled_idxs1_batch,
                        unlabeled_idxs2_batch
                    ):
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
                        assert overlap1_soft1.shape == overlap2_soft1.shape,(
                            "overlap region size must equal"
                        )

                        # overlap region pred by model1
                        overlap1_outputs1 = outputs1[
                            unlabeled_idx1,
                            :,
                            ul1[0][1]:br1[0][1],
                            ul1[1][1]:br1[1][1],
                            ul1[2][1]:br1[2][1]
                        ] # overlap batch1
                        overlap2_outputs1 = outputs1[
                            unlabeled_idx2,
                            :,
                            ul2[0][1]:br2[0][1],
                            ul2[1][1]:br2[1][1],
                            ul2[2][1]:br2[2][1]
                        ] # overlap batch2
                        assert overlap1_outputs1.shape == overlap2_outputs1.shape,(
                            "overlap region size must equal"
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
                        assert overlap1_soft2.shape == overlap2_soft2.shape,(
                            "overlap region size must equal"
                        )
                        
                        # overlap region outputs pred by model2
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
                        assert overlap1_outputs2.shape == overlap2_outputs2.shape,(
                            "overlap region size must equal"
                        )
                        overlap_outputs2_list.append(overlap1_outputs2.unsqueeze(0))
                        overlap_outputs2_list.append(overlap2_outputs2.unsqueeze(0))

                        # merge overlap region pred
                        overlap_soft1_tmp = (overlap1_soft1 + overlap2_soft1) / 2.
                        overlap_soft2_tmp = (overlap1_soft2 + overlap2_soft2) / 2.
                        overlap_soft1_list.append(overlap_soft1_tmp.unsqueeze(0))
                        overlap_soft2_list.append(overlap_soft2_tmp.unsqueeze(0))
                    overlap_soft1 = torch.cat(overlap_soft1_list, 0)
                    overlap_soft2 = torch.cat(overlap_soft2_list, 0)
                    overlap_outputs1 = torch.cat(overlap_outputs1_list, 0)
                    overlap_outputs2 = torch.cat(overlap_outputs2_list, 0)
                if self.current_iter < self.began_condition_iter:
                    pseudo_supervision1 = torch.FloatTensor([0]).to(self.device)
                else:
                    if self.use_CAC:
                        overlap_pseudo_outputs2 = torch.argmax(
                            overlap_soft2.detach(),
                            dim=1,
                            keepdim=False
                        )
                        overlap_pseudo_outputs2 = torch.cat(
                            [overlap_pseudo_outputs2, overlap_pseudo_outputs2]
                        )
                        pseudo_supervision1 = self._cross_entropy_loss_con(
                            overlap_outputs1,
                            overlap_pseudo_outputs2,
                            condition_batch[unlabeled_idx_batch]
                        )
                    else:
                        pseudo_outputs2 = torch.argmax(
                            outputs_soft2[self.labeled_bs:].detach(),
                            dim=1,
                            keepdim=False
                        )
                        pseudo_supervision1 = self._cross_entropy_loss_con(
                            outputs1[self.labeled_bs:],
                            pseudo_outputs2,
                            condition_batch[self.labeled_bs:]
                        )
                if self.current_iter < self.began_semi_iter:
                    pseudo_supervision2 = torch.FloatTensor([0]).to(self.device)
                else:
                    if self.use_CAC:
                        overlap_pseudo_outputs1 = torch.argmax(
                            overlap_soft1.detach(), 
                            dim=1, 
                            keepdim=False
                        )
                        overlap_pseudo_outputs1 = torch.cat(
                            [overlap_pseudo_outputs1, overlap_pseudo_outputs1]
                        )
                        pseudo_supervision2 = self.ce_loss(
                            overlap_outputs2, 
                            (
                                overlap_pseudo_outputs1==condition_batch[unlabeled_idxs_batch].unsqueeze(-1).unsqueeze(-1)
                            ).long()
                        ) 
                    else:
                        pseudo_outputs1 = torch.argmax(
                            outputs_soft1[self.labeled_bs:].detach(), 
                            dim=1, 
                            keepdim=False
                        )
                        pseudo_supervision2 = self.ce_loss(
                            outputs2[self.labeled_bs:], 
                            (pseudo_outputs1==condition_batch[self.labeled_bs:].\
                                unsqueeze(-1).unsqueeze(-1)).long()
                        )
                

                model1_loss = loss1 + self.consistency_weight * pseudo_supervision1
                model2_loss = loss2 + self.consistency_weight * pseudo_supervision2

                loss = model1_loss + model2_loss

                self.optimizer.zero_grad()
                self.optimizer2.zero_grad() 
                model1_loss.backward()
                model2_loss.backward()
                self.optimizer.step()
                self.optimizer2.step()

                self.current_iter += 1
                self._adjust_learning_rate()  
                for param_group in self.optimizer.param_groups:
                    self.current_lr = param_group['lr']
                self.tensorboard_writer.add_scalar('lr', self.current_lr, 
                                                   self.current_iter)
                self.tensorboard_writer.add_scalar(
                    'consistency_weight/consistency_weight', 
                    self.consistency_weight, 
                    self.current_iter
                )
                self.tensorboard_writer.add_scalar('loss/model1_loss', 
                                                   model1_loss, 
                                                   self.current_iter)
                self.tensorboard_writer.add_scalar('loss/model2_loss', 
                                                   model2_loss, 
                                                   self.current_iter)
                self.tensorboard_writer.add_scalar(
                    'loss/pseudo_supervision1',
                    pseudo_supervision1, self.current_iter
                )
                self.tensorboard_writer.add_scalar(
                    'loss/pseudo_supervision2',
                    pseudo_supervision2, 
                    self.current_iter
                )
                self.logging.info(
                    'iteration %d :'
                    'model1 loss : %f' 
                    'model2 loss : %f' 
                    'pseudo_supervision1 : %f'
                    'pseudo_supervision2 : %f' % (
                        self.current_iter, model1_loss.item(), 
                        model2_loss.item(), 
                        pseudo_supervision1.item(), 
                        pseudo_supervision2.item()
                    )
                )
                if self.current_iter % self.show_img_freq == 0:
                    image = volume_batch[0, 0:1, :, :, 20:61:10].permute(
                        3, 0, 1, 2).repeat(1, 3, 1, 1)
                    grid_image = make_grid(image, 5, normalize=True)
                    self.tensorboard_writer.add_image('train/Image', grid_image, 
                                                      self.current_iter)

                    image = outputs_soft1[0, 0:1, :, :, 20:61:10].permute(
                        3, 0, 1, 2).repeat(1, 3, 1, 1)
                    grid_image = make_grid(image, 5, normalize=False)
                    self.tensorboard_writer.add_image(
                        'train/Model1_Predicted_label',
                        grid_image, 
                        self.current_iter
                    )

                    image = outputs_soft2[0, 0:1, :, :, 20:61:10].permute(
                        3, 0, 1, 2).repeat(1, 3, 1, 1)
                    grid_image = make_grid(image, 5, normalize=False)
                    self.tensorboard_writer.add_image(
                        'train/Model2_Predicted_label',
                        grid_image, 
                        self.current_iter
                    )

                    image = label_batch[0, :, :, 20:61:10].unsqueeze(0).permute(
                        3, 0, 1, 2).repeat(1, 3, 1, 1)
                    grid_image = make_grid(image, 5, normalize=False)
                    self.tensorboard_writer.add_image('train/Groundtruth_label',
                                                      grid_image, 
                                                      self.current_iter)
                if (self.current_iter > self.began_eval_iter and
                    self.current_iter % self.val_freq == 0
                ):
                    with torch.no_grad():
                        self.evaluation(model=self.network)
                        self.evaluation(model=self.network2, do_condition=True)
                    self.network.train()
                    self.network2.train()
                if self.current_iter % self.save_checkpoint_freq == 0:
                    save_model_path = os.path.join(
                        self.output_folder,
                        'model_iter_' + str(self.current_iter) + '.pth'
                    )
                    torch.save(self.network.state_dict(), save_model_path)
                    self.logging.info(f"save model to {save_model_path}")

                    save_model_path = os.path.join(
                        self.output_folder,
                        'model2_iter_' + str(self.current_iter) + '.pth'
                    )
                    torch.save(self.network2.state_dict(), save_model_path)
                    self.logging.info(f'save model2 to {save_model_path}')
                if self.current_iter >= self.max_iterations:
                    break
            if self.current_iter >= self.max_iterations:
                iterator.close()
                break
        self.tensorboard_writer.close()

    def _train_EM(self):
        pass

    def _train_UAMT(self):
        print("================> Training UAMT<===============")
        iterator = tqdm(range(self.max_epoch), ncols=70)
        for epoch_num in iterator:
            for i_batch, sampled_batch in enumerate(self.dataloader):
                volume_batch, label_batch = (sampled_batch['image'], 
                                             sampled_batch['label'])
                volume_batch, label_batch = (volume_batch.to(self.device), 
                                             label_batch.to(self.device))
                unlabeled_volume_batch = volume_batch[self.labeled_bs:]

                
                label_batch = torch.argmax(label_batch, dim=1)
                noise = torch.clamp(torch.randn_like(
                    unlabeled_volume_batch)*0.1, -0.2, 0.2
                )
                ema_inputs = unlabeled_volume_batch + noise 

                outputs = self.network(volume_batch)
                outputs_soft = torch.softmax(outputs, dim=1)
                with torch.no_grad():
                    ema_output = self.ema_network(ema_inputs)
                T = 8
                _, _, d, w, h = unlabeled_volume_batch.shape
                volume_batch_r = unlabeled_volume_batch.repeat(2, 1, 1, 1, 1)
                stride = volume_batch_r.shape[0] // 2
                preds = torch.zeros([stride*T, 
                                     self.num_classes, d, w, h]).to(self.device)
                for i in range(T//2):
                    ema_inputs = volume_batch_r + torch.clamp(torch.randn_like(
                        volume_batch_r)*0.1, -0.2, 0.2)
                    with torch.no_grad():
                        preds[2*stride*i:2*stride*(i+1)] = self.ema_network(
                            ema_inputs)
                preds = torch.softmax(preds, dim=1)
                preds = preds.reshape(T, stride, self.num_classes, d, w, h)
                preds = torch.mean(preds, dim=0)
                uncertainty = -1.0 * torch.sum(preds*torch.log(preds+1e-6),
                                               dim=1, keepdim=True)
                self.loss_ce = self.ce_loss(outputs[:self.labeled_bs],
                                       label_batch[:self.labeled_bs])
                self.loss_dice = self.dice_loss(outputs_soft[:self.labeled_bs],
                                           label_batch[:self.labeled_bs].unsqueeze(1))
                supervised_loss = 0.5 * (self.loss_dice + self.loss_ce)
                self.consistency_weight = (
                    self._get_current_consistency_weight(self.current_iter//4)
                )
                consistency_dist = losses.softmax_dice_loss(
                    outputs[self.labeled_bs:], ema_output)
                if self.current_iter > self.began_semi_iter:
                    mask = (uncertainty < threshold).float()
                    self.consistency_loss = torch.sum(mask*consistency_dist)/(
                        2*torch.sum(mask)+1e-16
                    )
                else:
                    self.consistency_loss = torch.FloatTensor([0.0]).to(self.device)
                self.loss = supervised_loss + \
                    self.consistency_weight * self.consistency_loss
                self.optimizer.zero_grad()
                self.loss.backward()
                self.optimizer.step()
                self._update_ema_variables()
                self._adjust_learning_rate()
                self.current_iter += 1
                self._add_information_to_writer()
                if self.current_iter % self.show_img_freq == 0:
                    image = volume_batch[0, 0:1, :, :, 20:61:10].permute(
                        3, 0, 1, 2).repeat(1, 3, 1, 1)
                    grid_image = make_grid(image, 5, normalize=True)
                    self.tensorboard_writer.add_image('train/Image', grid_image,
                                                      self.current_iter)
                    image = outputs_soft[0, 1:2, :, :, 20:61:10].permute(
                        3, 0, 1, 2).repeat(2, 3, 1, 1)
                    grid_image = make_grid(image, 5, normalize=False)
                    self.tensorboard_writer.add_image('train/Predicted_label',
                                                      grid_image,
                                                      self.current_iter)
                    image = label_batch[0, :, :, 20:61:10].unsqueeze(0).permute(
                        3, 0, 1, 2).repeat(1, 3, 1, 1)
                    grid_image = make_grid(image, 5, normalize=False)
                    self.tensorboard_writer.add_image('train/Groundtruth_label',
                                                      grid_image,
                                                      self.current_iter)
                if (self.current_iter > self.began_eval_iter and
                    self.current_iter % self.val_freq == 0):
                    self.evaluation(model=self.network)
                if self.current_iter % self.save_checkpoint_freq == 0:
                    self._save_checkpoint()
                if self.current_iter >= self.max_iterations:
                    break
            if self.current_iter >= self.max_iterations:
                iterator.close()
                break
        self.tensorboard_writer.close()
        print("*"*10,"training done!","*"*10)


    
    def _get_current_consistency_weight(self, epoch):
        return self.consistency * ramps.sigmoid_rampup(epoch, 
                                                       self.consistency_rampup)
    
    def _update_ema_variables(self):
        # use the true average until the exponential average is more correct
        alpha = min(1-1/(self.current_iter + 1), self.ema_decay)
        for ema_param, param in zip(self.ema_network.parameters(),
                                    self.network.parameters()):
            ema_param.data.mul_(alpha).add_(1-alpha, param.data)
    
    def _worker_init_fn(self, worker_id):
        random.seed(self.seed + worker_id)
    
    def _save_checkpoint(self):
        save_checkpoint_path = os.path.join(
            self.output_folder, 'iter_' + str(self.current_iter)+ '.pth')
        torch.save(self.network.state_dict(), save_checkpoint_path)
        self.logging.info(f'save model to {save_checkpoint_path}')
    
    def _load_checkpoint(self):
        pass
    

    def _kaiming_normal_init_weight(self):
        for m in self.network2.modules():
            if isinstance(m, nn.Conv3d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


    def _xavier_normal_init_weight(self):
        for m in self.network.modules():
            if isinstance(m, nn.Conv3d):
                torch.nn.init.xavier_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
    def _cross_entropy_loss_con(self, output, target, condition):
        """
        cross entropy loss for conditional network
        """
        softmax = torch.softmax(output,dim=1)
        B,C,D,H,W = softmax.shape
        softmax_con = torch.zeros(B,2,D,H,W).to(self.device)
        softmax_con[:,1,...] = softmax[np.arange(B),condition.squeeze().long(),...] 
        softmax_con[:,0,...] = 1.0 - softmax_con[:,1,...]
        log = -torch.log(softmax_con.gather(1, target.unsqueeze(1)) + 1e-7)
        loss = log.mean()
        return loss

    def _adjust_learning_rate(self):
        if self.optimizer_type == 'Adam':
            return    # no need to adjust learning rate for adam optimizer   
        print("current learning rate: ",self.current_lr)
        self.current_lr = self.initial_lr * (
            1.0 - self.current_iter / self.max_iterations
        ) ** 0.9
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.current_lr
        
        if self.method_name in ['CPS','C3PS']:
            for param_group in self.optimizer2.param_groups:
                param_group['lr'] = self.current_lr
    
    def _add_information_to_writer(self):
        for param_group in self.optimizer.param_groups:
            self.current_lr = param_group['lr']
        self.tensorboard_writer.add_scalar('info/lr', self.current_lr, 
                                            self.current_iter)
        self.tensorboard_writer.add_scalar('info/total_loss', self.loss, 
                                            self.current_iter)
        self.tensorboard_writer.add_scalar('info/loss_ce', self.loss_ce, 
                                            self.current_iter)
        self.tensorboard_writer.add_scalar('info/loss_dice', self.loss_dice, 
                                            self.current_iter)
        self.logging.info(
            'iteration %d : loss : %f, loss_ce: %f, loss_dice: %f' %
            (self.current_iter, self.loss.item(), self.loss_ce.item(), 
                self.loss_dice.item()))
        self.tensorboard_writer.add_scalar('loss/loss', self.loss, 
                                            self.current_iter)
        if self.consistency_loss:
            self.tensorboard_writer.add_scalar('info/consistency_loss',
                                        self.consistency_loss, 
                                        self.current_iter)
        if self.consistency_weight:
            self.tensorboard_writer.add_scalar('info/consistency_weight',
                                                self.consistency_weight,
                                                self.current_iter)
    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.num_classes):
            temp_prob = input_tensor == i * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob)
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    

if __name__ == "__main__":
    # test semiTrainer
    pass
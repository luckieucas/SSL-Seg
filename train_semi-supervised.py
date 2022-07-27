import sys
import os
import shutil
import argparse
import logging
import time
import yaml
import random 
import numpy as np
from tqdm import tqdm

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F 
import torch.optim as optim
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import make_grid
from tensorboardX import SummaryWriter

from trainer.semi_trainer import SemiSupervisedTrainer
from utils.util import save_config







parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str,
                    default='train_config.yaml', help='training configuration')


if __name__ == "__main__":
    args = parser.parse_args()
    config = yaml.safe_load(open(args.config, 'r'))
    os.environ['CUDA_VISIBLE_DEVICES'] = config['gpu']
    save_config(config) # save config to yaml file with timestamp
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
    trainer = SemiSupervisedTrainer(config=config, output_folder=snapshot_path,
                                    logging=logging)   
    trainer.initialize_network()
    trainer.initialize()
    trainer.train()

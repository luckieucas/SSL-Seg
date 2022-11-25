'''
Descripttion: main func for all semi method, one can use this func to call any semi method
version: 
Author: Luckie
Date: 2022-01-10 13:36:49
LastEditors: Luckie
LastEditTime: 2022-01-13 14:21:40
'''
import os
import sys
import shutil
import logging
import torch
import argparse
import wandb
from omegaconf import OmegaConf

from dataset.BCVData import BCVDataset
from train_cross_pseudo_supervision_3D import train_cross_pseudo_3d

def main(args, snapshot_path):
    dataset_conf = args.DATASET
    if dataset_conf.data =='BCV':
        db_train = BCVDataset(img_list_file=dataset_conf.train_list,
                            patch_size=args.patch_size)
    
    train_cross_pseudo_3d(args,snapshot_path,db_train=db_train)


if __name__ == '__main__':
    #load config
    yaml = "/data/liupeng/semi-supervised_segmentation/SSL4MIS-master/code/configs/train_config.yaml"   
    args = OmegaConf.load(yaml)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    # create snap path
    snapshot_path = "../model/{}_{}_{}/{}".format(
    args.exp, args.labeled_num, args.dataset, args.model) 
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)
    if os.path.exists(snapshot_path + '/code'):
        shutil.rmtree(snapshot_path + '/code')
    shutil.copytree('.', snapshot_path + '/code',
                    shutil.ignore_patterns(['.git','__pycache__']))
    
    # set logging
    logging.basicConfig(filename=snapshot_path+"/log.txt", level=logging.INFO,
    format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%s')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))

    main(args, snapshot_path)
'''
Descripttion: test model
version: 
Author: Luckie
Date: 2021-04-19 13:09:02
LastEditors: Luckie
LastEditTime: 2021-06-09 15:00:23
'''
import os 
import shutil
import torch
import torch.nn.functional as F
from pathlib import Path
import math
import numpy as np
import SimpleITK as sitk
from medpy import metric
from tqdm import tqdm
from random import shuffle
import yaml
import argparse

from unet3d.model import get_model
from unet3d.config import load_config
from networks.net_factory_3d import net_factory_3d

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str,
                    default='test_config_new.yaml', help='training configuration')


task_name_id_dict={"full":0,"spleen":1,"kidney":2,"liver":4,"pancreas":5}
def test_single_case(net, image, stride_xy, stride_z, patch_size, num_classes=1, 
                     condition=-1, method='regular'):
    w, h, d = image.shape

    # if the size of image is less than patch_size, then padding it
    add_pad = False
    if w < patch_size[0]:
        w_pad = patch_size[0]-w
        add_pad = True
    else:
        w_pad = 0
    if h < patch_size[1]:
        h_pad = patch_size[1]-h
        add_pad = True
    else:
        h_pad = 0
    if d < patch_size[2]:
        d_pad = patch_size[2]-d
        add_pad = True
    else:
        d_pad = 0
    wl_pad, wr_pad = w_pad//2, w_pad-w_pad//2
    hl_pad, hr_pad = h_pad//2, h_pad-h_pad//2
    dl_pad, dr_pad = d_pad//2, d_pad-d_pad//2
    if add_pad:
        image = np.pad(image, [(wl_pad, wr_pad), (hl_pad, hr_pad),
                               (dl_pad, dr_pad)], mode='constant', constant_values=0)
    ww, hh, dd = image.shape

    sx = math.ceil((ww - patch_size[0]) / stride_xy) + 1
    sy = math.ceil((hh - patch_size[1]) / stride_xy) + 1
    sz = math.ceil((dd - patch_size[2]) / stride_z) + 1
    print(f"img shape:{image.shape}, sx:{sx}, sy:{sy}, sz:{sz}")
    score_map = np.zeros((num_classes, ) + image.shape).astype(np.float32)
    cnt = np.zeros(image.shape).astype(np.float32)

    for x in range(0, sx):
        xs = min(stride_xy*x, ww-patch_size[0])
        for y in range(0, sy):
            ys = min(stride_xy * y, hh-patch_size[1])
            for z in range(0, sz):
                zs = min(stride_z * z, dd-patch_size[2])
                test_patch = image[xs:xs+patch_size[0],
                                   ys:ys+patch_size[1], zs:zs+patch_size[2]]
                test_patch = np.expand_dims(np.expand_dims(
                    test_patch, axis=0), axis=0).astype(np.float32)
                test_patch = torch.from_numpy(test_patch).cuda()
        
                with torch.no_grad():
                    y = net(test_patch)
                    # ensemble
                    y = torch.softmax(y, dim=1)
                y = y.cpu().data.numpy()
                y = y[0, :, :, :, :]
                score_map[:, xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] \
                    = score_map[:, xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] + y
                cnt[xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] \
                    = cnt[xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] + 1
    score_map = score_map/np.expand_dims(cnt, axis=0)
    label_map = np.argmax(score_map, axis=0)

    if add_pad:
        label_map = label_map[wl_pad:wl_pad+w,
                              hl_pad:hl_pad+h, dl_pad:dl_pad+d]
        score_map = score_map[:, wl_pad:wl_pad +
                              w, hl_pad:hl_pad+h, dl_pad:dl_pad+d]
    return label_map


def cal_metric(gt, pred, cal_hd95=False, spacing=None):
    if pred.sum() > 0 and gt.sum() > 0:
        dice = metric.binary.dc(pred, gt)
        if cal_hd95:
            hd95 = metric.binary.hd95(pred, gt, voxelspacing=spacing)
        else:
            hd95 = 0.0
        return np.array([dice, hd95])
    else:
        return np.zeros(2)



def test_all_case_BCV(net, test_list="full_test.list", num_classes=4, 
                      patch_size=(48, 160, 160), stride_xy=32, stride_z=24, 
                      condition=-1, method="regular", cal_hd95=False,
                      cut_lower=-68, cut_upper=200, save_prediction=False,
                      prediction_save_path='./'):
    with open(test_list, 'r') as f:
        image_list = [img.replace('\n','') for img in f.readlines()]
    total_metric = np.zeros((num_classes-1, 2))
    print("Validation begin")
    condition_list = [i for i in range(1,num_classes)]
    shuffle(condition_list)
    img_num = np.zeros((num_classes-1,1))
    for i, image_path in enumerate(tqdm(image_list)):
        print(f"=============>processing {image_path}")
        if len(image_path.strip().split()) > 1:
            image_path, mask_path = image_path.strip().split()
        else: 
            mask_path = image_path.replace('img','label')
        assert os.path.isfile(mask_path),"invalid mask path error"
        
        """get task name and task id"""
        _,img_name = os.path.split(image_path)

        image_sitk = sitk.ReadImage(image_path)
        spacing = image_sitk.GetSpacing()
        image = sitk.GetArrayFromImage(image_sitk)
        label = sitk.GetArrayFromImage(sitk.ReadImage(mask_path))
        np.clip(image,cut_lower,cut_upper,out=image)
        image = (image - image.mean()) / image.std()
        prediction = test_single_case(
            net, image, stride_xy, stride_z, patch_size, 
            num_classes=num_classes, condition=condition, method=method)
        
        for i in range(1, num_classes):
            metrics = cal_metric(
                label == i, prediction == i, cal_hd95=cal_hd95, spacing=spacing
            )
            print(f"class:{i}, metric:{metrics}")
            img_num[i-1]+=1
            total_metric[i-1, :] += metrics

        if save_prediction: 
            pred_itk = sitk.GetImageFromArray(prediction.astype(np.uint8))
            pred_itk.SetSpacing(spacing)
            _,image_name = os.path.split(image_path)
            sitk.WriteImage(pred_itk, prediction_save_path+image_name.replace(".nii.gz","_pred.nii.gz"))
    print("Validation end")
    print(f"img_num:{img_num}")
    return total_metric / img_num

if __name__ == '__main__':
    args = parser.parse_args()
    config = yaml.safe_load(open(args.config, 'r'))
    os.environ['CUDA_VISIBLE_DEVICES'] = config['gpu']
    # model = get_model(config['model'])
    best_dice = 0
    best_dice_list = 0
    best_epoch = 200
    dataset_name = config['dataset_name']
    dataset_config = config['DATASET'][dataset_name]
    cut_upper = dataset_config['cut_upper']
    cut_lower = dataset_config['cut_lower']
    for epoch in range(357,358):
        print(f"-----test epoch:{epoch}------")
        #model_path= f"/data/liupeng/semi-supervised_segmentation/3D_U-net/models/train_multi_organ_all_data_baseline_0601/epoch_{epoch}.pth"
        #model_path = "../model/BCV_4_C3PS_test/unet_3D/model1_iter_8800_dice_0.5924.pth"
        model_path = "/data/liupeng/semi-supervised_segmentation/3D_U-net_baseline/models/train_MMWHS_all_data_baseline_sgd/best_iter1600_dice0.6124.pth"
        model_path = model_path
        save_path,_ = os.path.split(model_path)
        prediction_save_path = "{}/Prediction/".format(save_path)
        if os.path.exists(prediction_save_path):
            shutil.rmtree(prediction_save_path)
        os.makedirs(prediction_save_path)
        model = net_factory_3d(net_type='unet_3D',in_chns=1, 
                                      class_num=dataset_config['num_classes'],
                                      model_config=config['model'])
        model.load_state_dict(torch.load(model_path, map_location="cuda:0"))
        try:
            epoch = int(model_path.split("_")[-1].replace(".pth",""))
        except ValueError:
            print('epoch value error')
            epoch = 0
        
        epoch = 0
        test_list = dataset_config['test_list']
        patch_size = config['DATASET']['patch_size']
        model = model.cuda()
        model.eval()
        avg_metric = test_all_case_BCV(
                            model,
                            test_list=test_list,
                            num_classes=dataset_config['num_classes'], 
                            patch_size=patch_size,
                            stride_xy=64, 
                            stride_z=64,
                            cal_hd95=True,
                            cut_upper=cut_upper,
                            cut_lower=cut_lower,
                            save_prediction=True,
                            prediction_save_path=prediction_save_path
                        )
        print(avg_metric)
        print(avg_metric[:, 0].mean(),avg_metric[:,1].mean())
        if best_dice < avg_metric[:, 0].mean():
            best_dice = avg_metric[:, 0].mean()
            best_epoch = epoch
            best_dice_list = avg_metric
    print(f"best_dice:{best_dice}, best_epoch:{best_epoch}, best_dice_list:{avg_metric}")
import os
import math
from glob import glob

import h5py
import numpy as np
import SimpleITK as sitk
import torch
import torch.nn.functional as F
from medpy import metric
from tqdm import tqdm
from random import shuffle
import random


def test_single_case(net, image, stride_xy, stride_z, patch_size, num_classes=1, condition=-1, method='regular', return_scoremap=False):
    w, h, d = image.shape

    # if the size of image is less than patch_size, then padding it
    device = next(net.parameters()).device
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
    if condition>0:
        score_map = np.zeros((2, ) + image.shape).astype(np.float32) 
    else:
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
                test_patch = torch.from_numpy(test_patch).to(device)
                

                with torch.no_grad():
                    if condition>0:
                        condition = torch.tensor([condition],dtype=torch.long, device=device)
                        pred1 = net(test_patch, condition)
                    else:
                        pred1 = net(test_patch)
                        if len(pred1)>0 and isinstance(pred1, (tuple, list)):
                            pred1 = pred1[0]
                    # ensemble
                    pred = torch.softmax(pred1, dim=1)
                pred = pred.cpu().data.numpy()
                pred = pred[0, :, :, :, :]
                score_map[:, xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] \
                    = score_map[:, xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] + pred
                cnt[xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] \
                    = cnt[xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] + 1
    score_map = score_map/np.expand_dims(cnt, axis=0)
    label_map = np.argmax(score_map, axis=0)

    if add_pad:
        label_map = label_map[wl_pad:wl_pad+w,
                              hl_pad:hl_pad+h, dl_pad:dl_pad+d]
        score_map = score_map[:, wl_pad:wl_pad +
                              w, hl_pad:hl_pad+h, dl_pad:dl_pad+d]
    
    if return_scoremap:
        return label_map, score_map
    else:
        return label_map


def calculate_metric(gt, pred, cal_hd95=False):
    if pred.sum() > 0 and gt.sum() > 0:
        dice = metric.binary.dc(pred, gt)
        if cal_hd95:
            hd95 = metric.binary.hd95(pred, gt)
        else:
            hd95 = 0.0
        return np.array([dice, hd95])
    else:
        return np.zeros(2)


def test_all_case(net, base_dir, test_list="full_test.list", num_classes=4, patch_size=(48, 160, 160), stride_xy=32, stride_z=24):
    with open(base_dir + '/{}'.format(test_list), 'r') as f:
        image_list = f.readlines()
    image_list = [base_dir + "/data/{}.h5".format(
        item.replace('\n', '').split(",")[0]) for item in image_list]
    total_metric = np.zeros((num_classes-1, 2))
    print("Validation begin")
    for image_path in tqdm(image_list):
        h5f = h5py.File(image_path, 'r')
        image = h5f['image'][:]
        label = h5f['label'][:]
        prediction = test_single_case(
            net, image, stride_xy, stride_z, patch_size, num_classes=num_classes)
        for i in range(1, num_classes):
            total_metric[i-1, :] += calculate_metric(label == i, prediction == i)
    print("Validation end")
    return total_metric / len(image_list)


def test_all_case_BCV(net, test_list="full_test.list", num_classes=4, 
                        patch_size=(48, 160, 160), stride_xy=32, stride_z=24, 
                        do_condition=False, method="regular",
                        cal_metric=True,
                        save_prediction=False,
                        prediction_save_path='./',
                        test_num=2,
                        cut_upper=200,
                        cut_lower=-68,
                        con_list=None,
                        normalization='Zscore'):
    if os.path.isdir(test_list):
        image_list = glob(test_list+"*.nii.gz")
    else:
        with open(test_list, 'r') as f:
            image_list = [img.replace('\n','') for img in f.readlines()]
    print("Total test images:",len(image_list))
    total_metric = np.zeros((num_classes-1, 2))
    print("Validation begin")
    if con_list:
        condition_list = con_list # for condition learning
    else:
        condition_list = [i for i in range(1,num_classes)]
    #shuffle(condition_list)
    shuffle(image_list)
    if not do_condition:
        test_num = len(image_list)
    for i, image_path in enumerate(tqdm(image_list)):
        if i>test_num-1 and do_condition:
            break
        if len(image_path.strip().split()) > 1:
            image_path, mask_path = image_path.strip().split()
        else: 
            mask_path = image_path.replace('img','label')
        if cal_metric:
            assert os.path.isfile(mask_path),"invalid mask path error"
        image_sitk = sitk.ReadImage(image_path)
        image = sitk.GetArrayFromImage(image_sitk)
        
        if cal_metric: # whether calculate metrics
            label = sitk.GetArrayFromImage(sitk.ReadImage(mask_path))
        else:
            label = np.zeros_like(image)
        if "heartMR" in image_path or normalization=='MinMax':
            min_val_1p=np.percentile(image,1)
            max_val_99p=np.percentile(image,99)
            # min-max norm on total 3D volume
            print('min max norm')
            image=(image-min_val_1p)/(max_val_99p-min_val_1p)
            np.clip(image, 0.0, 1.0, out=image)
        else:
            np.clip(image,cut_lower,cut_upper,out=image)
            image = (image - image.mean()) / image.std()
        if do_condition:
            print(f"===>test image:{image_path}")
            for condition in condition_list:
                prediction = test_single_case(
                    net, image, stride_xy, stride_z, patch_size, 
                    num_classes=2, condition=condition, method=method)
                if cal_metric:
                    metric = calculate_metric(label==condition, prediction)
                    print(f"condition:{condition}, metric:{metric}")
                    total_metric[condition-1, :] += metric
        else:
            prediction = test_single_case(
                net, image, stride_xy, stride_z, patch_size, 
                num_classes=num_classes, condition=-1, method=method)
            if cal_metric:
                for i in range(1, num_classes):
                    total_metric[i-1, :] += calculate_metric(label == i, prediction == i)
        
        # save prediction
        if save_prediction: 
            spacing = image_sitk.GetSpacing()
            pred_itk = sitk.GetImageFromArray(prediction.astype(np.uint8))
            pred_itk.SetSpacing(spacing)
            _,image_name = os.path.split(image_path)
            sitk.WriteImage(pred_itk, prediction_save_path+image_name.replace(".nii.gz","_pred.nii.gz"))

    print("Validation end")
    if con_list:
        return total_metric[[con-1 for con in con_list]] / test_num
    else:
        return total_metric / test_num



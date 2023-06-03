'''
Descripttion: test model
version: 
Author: Luckie
Date: 2021-04-19 13:09:02
LastEditors: Luckie
LastEditTime: 2021-06-09 15:00:23
'''
from email.policy import default
import os 
import shutil
import torch
import math
import numpy as np
import SimpleITK as sitk
from medpy import metric
from tqdm import tqdm
from random import shuffle
from typing import OrderedDict
import yaml
import argparse
from glob import glob
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric
from monai.transforms import Compose,LoadImage,ToTensor

from networks.net_factory_3d import net_factory_3d
from batchgenerators.utilities.file_and_folder_operations import save_json

task_name_id_dict={"full":0,"spleen":1,"kidney":2,"liver":4,"pancreas":5,
                   "heart":0}
parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str,
                    default='test_config_new.yaml', 
                    help='training configuration'
)
parser.add_argument('--gpu', type=str, default='2',help='gpu id for testing')
parser.add_argument(
    '--model_path', type=str,
    default='/data/liupeng/semi-supervised_segmentation/SSL4MIS-master/model/BCV_500_CVCL_partial_test_vnet_cvcl_SGD_SGD/vnet/model_iter_18000_dice_0.8605.pth',
    help='model path for testing'
)


class_id_name_dict = {
    'MMWHS':['MYO', 'LA', 'LV', 'RA', 'AA', 'PA', 'RV'],
    'BCV':['Spleen', 'Right Kidney', 'Left Kidney','Liver','Pancreas'],
    'LA':['LA'],
    'FLARE':['Liver','Kidney','Spleen','Pancreas']
}

def cal_metric(gt, pred, num_classes, total_metric, res_metric, img_num, 
               class_name_list, cal_hd95=False, cal_asd=False, spacing=None):
    each_metric = np.zeros((num_classes-1, 3))
    for i in range(1, num_classes):
        binary_gt = gt==i
        binary_pred = pred==i
        if binary_pred.sum() > 0 and binary_gt.sum() > 0: 
            dice = metric.binary.dc(binary_pred,binary_gt)     
            if cal_hd95:
                hd95 = metric.binary.hd95(binary_pred,binary_gt)  
            else:
                hd95 = 0.0
            if cal_asd:
                asd = metric.binary.asd(binary_pred,binary_gt)  
            else:
                asd = 0.0
            metrics = np.array([dice, hd95, asd])
        else:
            metrics = np.array([0.0, 150, 150])
        print(f"class:{class_name_list[i-1]}, metric:{metrics}")
        res_metric[class_name_list[i-1]] = {
            'Dice':metrics[0],'HD95':metrics[1],'ASD':metrics[2]
        }
        img_num[i-1]+=1
        total_metric[i-1, :] += metrics
        each_metric[i-1, :] += metrics
    res_metric['Mean'] = {'Dice':each_metric[:,0].mean(),
                          'HD95':each_metric[:,1].mean(),
                          'ASD':each_metric[:,2].mean()
                }
    return each_metric



def process_fn(seg_prob_tuple, window_data, importance_map_):
    """seg_prob_tuple, importance_map = 
    process_fn(seg_prob_tuple, window_data, importance_map_)
    """
    if len(seg_prob_tuple)>0 and isinstance(seg_prob_tuple, (tuple, list)):
        seg_prob = torch.softmax(seg_prob_tuple[0],dim=1)
        return tuple(seg_prob.unsqueeze(0))+seg_prob_tuple[1:],importance_map_
    else:
        seg_prob = torch.softmax(seg_prob_tuple,dim=1)
        return seg_prob,importance_map_


def test_all_case_monai(net, test_list="full_test.list", num_classes=4, 
                      patch_size=(48, 160, 160),overlap=0.5,
                      condition=-1,method="regular",cal_hd95=False,
                      cal_asd=False,cut_lower=-68, cut_upper=200, 
                      save_prediction=False,prediction_save_path='./',
                      class_name_list=[]):
    with open(test_list, 'r') as f:
        image_list = [img.replace('\n','') for img in f.readlines()]
    total_metric = np.zeros((num_classes-1, 3))
    all_scores = OrderedDict() # for save as json
    all_scores['all'] = []
    all_scores['mean'] = OrderedDict()
    print("***************************validation begin************************")
    condition_list = [i for i in range(1,num_classes)]
    shuffle(condition_list)
    img_num = np.zeros((num_classes-1,1))
    transform = Compose([ToTensor(), LoadImage(image_only=True)])
    dice_metric = DiceMetric(False,"mean")
    with torch.no_grad():
        for i, image_path in enumerate(tqdm(image_list)):
            res_metric = OrderedDict()
            if len(image_path.strip().split()) > 1:
                image_path, mask_path = image_path.strip().split()
            else: 
                mask_path = image_path.replace('img','label')
            assert os.path.isfile(mask_path),"invalid mask path error"
            _,img_name = os.path.split(image_path)
            print(f"=============>processing {img_name}")
            
            # use for save json results
            res_metric['image_path'] = image_path 
            res_metric['mask_path'] = mask_path
            
            

            image_sitk = sitk.ReadImage(image_path)
            spacing = image_sitk.GetSpacing()
            image = sitk.GetArrayFromImage(image_sitk)
            label = sitk.GetArrayFromImage(sitk.ReadImage(mask_path))
            task_name = 'full'
            if len(img_name.split("_")) >1:
                task_name = img_name.split("_")[0]
            task_id = task_name_id_dict[task_name]
            print("task_id:",task_id)
            image = sitk.GetArrayFromImage(sitk.ReadImage(image_path))
            label = sitk.GetArrayFromImage(sitk.ReadImage(mask_path))
            if task_id > 0 and task_id !=2:
                label[label!=0] = task_id
            if task_id == 2:
                label[label==2] = 3
                label[label==1] = 2
            np.clip(image,cut_lower,cut_upper,out=image)
            image = (image - image.mean()) / image.std()
            image = torch.from_numpy(image).unsqueeze(0).unsqueeze(0).cuda()
            prediction,_ = sliding_window_inference(
                image.float(),patch_size,2,net,overlap=overlap,
                mode='gaussian',process_fn=process_fn
            )       
            prediction = torch.argmax(prediction,dim=1).squeeze().cpu().numpy()
            cal_metric(label,prediction,num_classes,total_metric,res_metric, 
                       img_num,class_name_list)
            all_scores['all'].append(res_metric)
            if save_prediction: 
                pred_itk = sitk.GetImageFromArray(prediction.astype(np.uint8))
                pred_itk.SetSpacing(spacing)
                sitk.WriteImage(
                    pred_itk, 
                    prediction_save_path+img_name.replace(".nii.gz","_pred.nii.gz")
                )
    
    mean_metric = total_metric / img_num
    for i in range(1, num_classes):
        all_scores['mean'][class_name_list[i-1]] = {
            'Dice': mean_metric[i-1][0],
            'HD95': mean_metric[i-1][1],
            'ASD': mean_metric[i-1][2]
        }
    all_scores['mean']['mean']={
        'Dice':mean_metric[:,0].mean(),
        'HD95':mean_metric[:,1].mean(),
        'ASD':mean_metric[:,2].mean()
    }
    save_json(all_scores,prediction_save_path+"/Results.json")
    print("***************************validation end**************************")
    return mean_metric


if __name__ == '__main__':
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    model_path = args.model_path

    root_path,_ = os.path.split(model_path)
    config_file_list =  glob(root_path+"/*yaml")
    sorted(config_file_list)
    config_file = config_file_list[-1]
    print(f"===========> using config file: {os.path.split(config_file)[1]}")
    config = yaml.safe_load(open(config_file, 'r'))
    method_name = config['method']
    method = 'regular'
    if method_name == 'URPC':
        config['backbone'] = 'URPC'
        method = 'urpc'

    dataset_name = config['dataset_name']
    class_name_list = class_id_name_dict[dataset_name]
    dataset_config = config['DATASET'][dataset_name]
    cut_upper = dataset_config['cut_upper']
    cut_lower = dataset_config['cut_lower']

    pred_save_path = "{}/Prediction_full_monai/".format(root_path)
    if os.path.exists(pred_save_path):
        shutil.rmtree(pred_save_path)
    os.makedirs(pred_save_path)
    model = net_factory_3d(net_type=config['backbone'],in_chns=1, 
                                class_num=dataset_config['num_classes'],
                                model_config=config['model'])
    model.load_state_dict(torch.load(model_path, map_location="cuda:0"))  
    #test_list = dataset_config['test_list']
    test_list = '/data/liupeng/semi-supervised_segmentation/3D_U-net_baseline/datasets/test_full.txt'
    patch_size = config['DATASET']['patch_size']
    model = model.cuda()
    model.eval()
    avg_metric = test_all_case_monai(
                        model,
                        test_list=test_list,
                        num_classes=dataset_config['num_classes'], 
                        patch_size=patch_size,
                        overlap=0.5,
                        cal_hd95=False,
                        cal_asd=True,
                        cut_upper=cut_upper,
                        cut_lower=cut_lower,
                        save_prediction=True,
                        prediction_save_path=pred_save_path,
                        class_name_list=class_name_list,
                        method = method
                    )
    print(avg_metric)
    print(avg_metric[:, 0].mean(),avg_metric[:,1].mean(), avg_metric[:,2].mean())

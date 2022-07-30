"test conditional model"
import os 
import argparse
import torch
from glob import glob  
import SimpleITK as sitk 
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool
import yaml

from networks.net_factory_3d import net_factory_3d
from val_3D import test_single_case,calculate_metric

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str,
                    default='test_config_new.yaml', help='model_name')
parser.add_argument('--test_list', type=str, 
                    default='../data/BCV/test.txt')
parser.add_argument('--checkpoint', type=str,
                    default='../model/BCV_4_C3PS_test/unet_3D/'\
                            'model2_iter_15800_dice_0.7323.pth')

def predict_multiprocess():
    pass

def test_all_case_condition(net, test_list="full_test.list", num_classes=4, 
                        patch_size=(48, 160, 160), stride_xy=32, stride_z=24, 
                        condition=-1, method="regular",
                        cal_metric=True,
                        save_prediction=False,
                        prediction_save_path='./',
                        cut_upper=1000,
                        cut_lower=-1000):
    if os.path.isdir(test_list):
        image_list = glob(test_list+"*.nii.gz")
    else:
        with open(test_list, 'r') as f:
            image_list = [img.replace('\n','') for img in f.readlines()]
    print("Total test images:",len(image_list))
    total_metric = np.zeros((num_classes-1, 2))
    print("Validation begin")
    condition_list = [i for i in range(1,num_classes)]
    for i, image_path in enumerate(tqdm(image_list)):
        print(f"===========>processing {image_path}")
        if len(image_path.strip().split()) > 1:
            image_path, mask_path = image_path.strip().split()
        else: 
            mask_path = image_path.replace('img','label')
        if cal_metric:
            assert os.path.isfile(mask_path),"invalid mask path error"
        image_sitk = sitk.ReadImage(image_path)
        image = sitk.GetArrayFromImage(image_sitk)
        shape = image.shape
        
        if cal_metric: # whether calculate metrics
            label = sitk.GetArrayFromImage(sitk.ReadImage(mask_path))
        else:
            label = np.zeros_like(image)
        np.clip(image,cut_lower,cut_upper,out=image)
        image = (image - image.mean()) / image.std()

        prob_map = np.zeros((num_classes,shape[0],shape[1],shape[2]))
        for con_index in condition_list:
            pred_con, prob = test_single_case(net, image, stride_xy, stride_z, 
                                               patch_size, 
                                               num_classes=num_classes, 
                                               condition=con_index, 
                                               method=method,
                                               return_scoremap=True)
            prob_map[0,:] += prob[0]
            prob_map[con_index,:] = prob[1]
            metric = calculate_metric(label == con_index, pred_con)
            print(f"con:{con_index}, metric:{metric}")
        prob_map[0] /= len(condition_list)
        prediction = np.argmax(prob_map,0)
        if cal_metric:
            for i in range(1, num_classes):
                total_metric[i-1, :] += calculate_metric(label == i, 
                                                         prediction == i)
        
        # save prediction
        if save_prediction: 
            spacing = image_sitk.GetSpacing()
            pred_itk = sitk.GetImageFromArray(prediction.astype(np.uint8))
            pred_itk.SetSpacing(spacing)
            _,image_name = os.path.split(image_path)
            sitk.WriteImage(pred_itk, prediction_save_path+image_name.replace(".nii.gz","_pred.nii.gz"))

    print("Validation end")
    return total_metric / len(image_list)
def main(args, config):
    model = net_factory_3d("unet_3D_condition", in_chns=1, class_num=2).cuda()
    model_state_dict = torch.load(config['model_checkpoint'])
    model.load_state_dict(model_state_dict)
    dataset_name = config['dataset_name']
    dataset_config = config['DATASET'][dataset_name]
    cut_upper = dataset_config['cut_upper']
    cut_lower = dataset_config['cut_lower']
    test_list = dataset_config['test_list']
    patch_size = (96,160,160)
    model.eval()
    metrics = test_all_case_condition(
                        model,
                        test_list=test_list,
                        num_classes=8, 
                        patch_size=patch_size,
                        stride_xy=64, 
                        stride_z=64,
                        condition=1,
                        cut_lower=cut_lower,
                        cut_upper=cut_upper
                    )
    print(metrics)


if __name__ == "__main__":
    args = parser.parse_args()
    config = yaml.safe_load(open(args.config, 'r'))
    main(args, config)

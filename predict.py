import os
import SimpleITK as sitk
import numpy as np 
import random
from random import shuffle
from tqdm import tqdm
import argparse

from val_3D import test_single_case


parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str,
                    default='../data/BraTS2019', help='Name of Experiment')
parser.add_argument('--exp', type=str,
                    default='BraTS2019/Interpolation_Consistency_Training_25', 
                    help='experiment_name')
parser.add_argument('--model', type=str,
                    default='unet_3D', help='model_name')
parser.add_argument('--test_list',type=str,
                    default='../data/Flare/Flare_test.txt',help='test_list')
parser.add_argument('--checkpoint',type=str,
                    default='../model/fully_supervised_Flare_baseline'\
                    '_labeledNum289/unet_3D/unet_3D_best_model.pth',
                    help='test model')


def predict_all_case(net, test_list="full_test.list", num_classes=4, 
                        patch_size=(48, 160, 160), stride_xy=32, stride_z=24, 
                        condition=-1, method="regular",
                        cal_metric=True,
                        save_prediction=False,
                        prediction_save_path='./'):
    """
    predict case without ground truth
    """
    with open(test_list, 'r') as f:
        image_list = [img.replace('\n','') for img in f.readlines()]
    total_metric = np.zeros((num_classes-1, 2))
    print("Validation begin")
    condition_list = [i for i in range(1,num_classes)]
    shuffle(condition_list)
    for i, image_path in enumerate(tqdm(image_list)):
        if len(image_path.strip().split()) > 1:
            image_path, mask_path = image_path.strip().split()
        else: 
            mask_path = image_path.replace('img','label')
        assert os.path.isfile(mask_path),"invalid mask path error"
        image_sitk = sitk.ReadImage(image_path)
        image = sitk.GetArrayFromImage(image_sitk)
        label = sitk.GetArrayFromImage(sitk.ReadImage(mask_path))
        if condition>0:
            if i< len(condition_list):
                condition = condition_list[i]
            else:
                condition = random.choice(condition_list)
            label = (label==condition).astype(np.uint8)
        np.clip(image,-1000,1000,out=image)
        image = (image - image.mean()) / image.std()
        prediction = test_single_case(
            net, image, stride_xy, stride_z, patch_size, 
            num_classes=num_classes, condition=condition, method=method)
        for i in range(1, num_classes):
            total_metric[i-1, :] += cal_metric(label == i, prediction == i)
        
        # save prediction
        if save_prediction: 
            spacing = image_sitk.GetSpacing()
            pred_itk = sitk.GetImageFromArray(prediction.astype(np.uint8))
            pred_itk.SetSpacing(spacing)
            _,image_name = os.path.split(image_path)
            sitk.WriteImage(pred_itk, prediction_save_path+image_name.replace(".nii.gz","_pred.nii.gz"))

    print("Validation end")
    return total_metric / len(image_list)



def Inference(args):
    num_classes = 5
    model_path = args.checkpoint
    save_path,_ = os.path.split(model_path)
    test_save_path = "{}/Prediction/".format(save_path)
    if os.path.exists(test_save_path):
        shutil.rmtree(test_save_path)
    os.makedirs(test_save_path)
    #net = unet_3D(n_classes=num_classes, in_channels=1).cuda()
    net = net_factory_3d(net_type=args.model, in_chns=1, class_num=num_classes)
    net.load_state_dict(torch.load(model_path))
    print("init weight from {}".format(model_path))
    net = net.cuda()
    net.eval()
    avg_metric = test_all_case_BCV(net, 
                                    test_list= args.test_list,
                                    num_classes=num_classes, 
                                    patch_size=(96,160,160),
                                    stride_xy=64, 
                                    stride_z=64,save_prediction=True,
                                    prediction_save_path=test_save_path)
    return avg_metric


if __name__ == '__main__':
    args = parser.parse_args()
    metric = Inference(args)
    print(metric)
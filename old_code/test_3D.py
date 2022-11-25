import argparse
import os
import shutil
from glob import glob

import torch
from networks.net_factory_3d import net_factory_3d
from networks.unet_3D import unet_3D
from val_3D import test_all_case_BCV

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
                    default='../model/fully_supervised_Flare_baseline_'\
                    'labeledNum289/unet_3D/unet_3D_best_model.pth',
                    help='test model')



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
                                    stride_z=64,
                                    cal_metric=False,
                                    save_prediction=True,
                                    prediction_save_path=test_save_path)
    return avg_metric


if __name__ == '__main__':
    args = parser.parse_args()
    metric = Inference(args)
    print(metric)

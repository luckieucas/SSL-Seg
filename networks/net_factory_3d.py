'''
Descripttion: 
version: 
Author: Luckie
Date: 2021-12-29 11:07:34
LastEditors: Luckie
LastEditTime: 2022-01-08 18:58:10
'''
from torch import nn
from dynamic_network_architectures.architectures.unet import PlainConvUNet
from dynamic_network_architectures.building_blocks.helper import get_matching_instancenorm, convert_dim_to_conv_op
from networks.unet_3D import unet_3D
from networks.unet_3D_condition import unet_3D_Condition,Unet3DConditionDecoder,Unet3DConditionBottom
from networks.vnet import VNet
from networks.VoxResNet import VoxResNet
from networks.attention_unet import Attention_UNet
from networks.discriminator import FC3DDiscriminator
from networks.unet_3D_dv_semi import unet_3D_dv_semi
from networks.nnunet import initialize_network
from unet3d.model import get_model
from .McNet import MCNet3d_v2
from networks.unet_3D_cl import unet_3D_cl 
from networks.unet_3D_sr import unet_3D_sr
from networks.resnet3D import generate_resnet3d
from networks.DenseNet3D import SinglePathDenseNet
from networks.DenseVox3D import DenseVoxelNet
from monai.networks.nets import UNet
from networks.PlaninUNet_sr import PlainConvUNetSR

def net_factory_3d(net_type="unet_3D", in_chns=1, class_num=2, 
                   model_config=None, device=None, condition_noise=False):
    if net_type == "unet_3D":
        model_config['out_channels'] = class_num
        net = get_model(model_config)
    elif net_type == 'unet_3D_old':
        net = unet_3D(n_classes=class_num, in_channels=in_chns)
        print(f"net:{net}")
    elif net_type == "unet_3D_condition":
        net = unet_3D_Condition(
            n_classes=class_num, in_channels=in_chns
        ).to(device)
    elif net_type == "unet_3D_condtion_decoder":
        net = Unet3DConditionDecoder(
            n_classes=class_num, in_channels=in_chns, condition_noise=condition_noise
        ).to(device)
    elif net_type == "Unet3DConditionBottom":
        net = Unet3DConditionBottom(
            n_classes=class_num, in_channels=in_chns
        ).to(device)
    elif net_type == "DAN":
        net = FC3DDiscriminator(num_classes=class_num).to(device)
    elif net_type == 'URPC':
        net = unet_3D_dv_semi(n_classes=class_num, in_channels=1).cuda()
    elif net_type == "attention_unet":
        net = Attention_UNet(n_classes=class_num, in_channels=in_chns).cuda()
    elif net_type == "voxresnet":
        net = VoxResNet(in_chns=in_chns, feature_chns=64,
                        class_num=class_num).cuda()
    elif net_type == "vnet":
        net = VNet(n_channels=in_chns, n_classes=class_num,
                   normalization='batchnorm', has_dropout=False).to(device)
    elif net_type == "nnUNet":
        net = initialize_network(num_classes=class_num).cuda()
    elif net_type == 'McNet':
        net = MCNet3d_v2(
            n_channels=in_chns, n_classes=class_num, normalization='batchnorm', 
            has_dropout=True
        ).to(device)
    elif net_type == 'unet_3D_cl':
        net = unet_3D_cl(
            feature_scale=4, n_classes=class_num, is_deconv=True, 
            in_channels=1, is_batchnorm=True
        ).to(device)
    elif net_type == 'unet_3D_sr':
        net = unet_3D_sr(feature_scale=4, n_classes=class_num, is_deconv=True, 
                         in_channels=1, is_batchnorm=True).to(device)
    elif net_type == 'resnet_3D_cvcl':
        net = generate_resnet3d(in_channels=1,
                                model_depth=10,
                                classes=class_num).to(device)
    elif net_type == 'densenet_3D_cvcl':
        #net = SinglePathDenseNet(in_channels=1,classes=class_num).to(device)
        net = DenseVoxelNet(in_channels=1, classes=class_num).to(device)
    else:
        net = None
    return net

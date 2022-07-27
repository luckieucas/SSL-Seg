'''
Descripttion: 
version: 
Author: Luckie
Date: 2021-04-12 16:20:15
LastEditors: Luckie
LastEditTime: 2021-07-28 09:38:49
'''

import os
import random
import torch 
from torch.utils.data import Dataset as dataset
from torch.utils.data import DataLoader as dataloader 
import numpy as np 
from glob import glob
import SimpleITK as sitk
from .data_augmentation import rotation, affine_transformation, random_cutout

#from .sampler import BatchSampler, ClassRandomSampler
#from data_augmentation import rotation, affine_transformation, random_cutout
#from sampler import BatchSampler, ClassRandomSampler #for test



task_name_id_dict={"full":0,"spleen":1,"kidney":2,"liver":4,"pancreas":5}



class Dataset(dataset):
    def __init__(self, img_list_file, patch_size=(48, 256, 256),
                cutout=False, affine_trans=False, 
                num_class=2, edge_prob=0., upper=1000, lower=-1000):
        self.patch_size = patch_size
        self.cutout = cutout
        self.affine_trans = affine_trans
        self.num_class = num_class
        self.edge_prob = edge_prob
        self.upper = upper
        self.lower = lower
        with open(img_list_file, 'r') as f:
            self.img_list = [img.replace("\n","") for img in f.readlines()]

    def __getitem__(self, index):
        img_path = self.img_list[index]
        mask_path = img_path.replace("img","label")
        assert os.path.isfile(mask_path),"invalid mask path error!"

        """read image and mask"""
        image = sitk.ReadImage(img_path)
        mask = sitk.ReadImage(mask_path)
        img_array = sitk.GetArrayFromImage(image) 
        mask_array = sitk.GetArrayFromImage(mask)
        mask_array = mask_array.astype(np.uint8)
        # 将灰度值在阈值之外的截断掉
        upper = self.upper
        lower = self.lower
        img_array[img_array > upper] = upper
        img_array[img_array < lower] = lower
        """Normalize the image"""
        img_array = (img_array - img_array.mean())/img_array.std()
        img_shape = img_array.shape
        
        """get image patch"""
        margin_len_x = self.patch_size[0]//2
        margin_len_y = self.patch_size[1]//2
        margin_len_z = self.patch_size[2]//2

        prob = random.random()
        
        # center_x = img_shape[0]-margin_len_x
        # if prob > self.edge_prob:
        center_x = random.randint(margin_len_x, img_shape[0]-margin_len_x)
        center_y = random.randint(margin_len_y, img_shape[1]-margin_len_y)
        center_z = random.randint(margin_len_z, img_shape[2]-margin_len_z)
        img_array = img_array[center_x-margin_len_x:center_x+margin_len_x,
                    center_y-margin_len_y:center_y+margin_len_y,\
                        center_z-margin_len_z:center_z+margin_len_z]
        mask_array = mask_array[center_x-margin_len_x:center_x+margin_len_x,\
                                center_y-margin_len_y:center_y+margin_len_y,\
                                center_z-margin_len_z:center_z+margin_len_z]
                #convert mask to one hot
        # do random cutout
        if self.cutout:
            mask_array = random_cutout(mask_array)
        img_array = torch.FloatTensor(img_array).unsqueeze(0)
        mask_array = torch.FloatTensor(mask_array).unsqueeze(0)
        gt_onehot = torch.zeros((self.num_class, mask_array.shape[1], mask_array.shape[2],mask_array.shape[3]))
        gt_onehot.scatter_(0, mask_array.long(), 1)
        mask_array = gt_onehot
        #mask_array = torch.FloatTensor(mask_array).unsqueeze(0)
        # do transformation
        
        if self.affine_trans:
            angle_x = random.uniform(-0.08,0.08)
            angle_y = random.uniform(-0.08,0.08)
            angle_z = random.uniform(-0.08,0.08)
            scale_x = random.uniform(0.8,1.2)
            scale_y = random.uniform(0.8,1.2)
            scale_z = random.uniform(0.8,1.2)     
            img = affine_transformation(img_array[np.newaxis,:], 
                                        radius=(angle_x, angle_y, angle_z), 
                                        translate=(0, 0, 0),
                                        scale=(scale_x, scale_y, scale_z),
                                        bspline_order=0, border_mode="nearest", 
                                        constant_val=0, is_reverse=False)
            mask = affine_transformation(mask_array[np.newaxis,:], 
                                        radius=(angle_x, angle_y, angle_z), 
                                        translate=(0, 0, 0),
                                        scale=(scale_x, scale_y, scale_z),
                                        bspline_order=0, border_mode="nearest",
                                        constant_val=0, is_reverse=False)
            img_array = img[0,:]
            mask_array = mask[0,:]
        return img_array,mask_array

    def __len__(self):
        return len(self.img_list)


class DatasetSemi(dataset):
    def __init__(self, img_list_file, patch_size=(48, 224, 224),
                cutout=False, affine_trans=False, 
                num_class=2, edge_prob=0., upper=1000, lower=-1000,
                labeled_num=4, train_supervised=False):
        self.patch_size = patch_size
        self.cutout = cutout
        self.affine_trans = affine_trans
        self.num_class = num_class
        self.edge_prob = edge_prob
        self.upper = upper
        self.lower = lower
        self.labeled_num = labeled_num
        self.train_supervised = train_supervised
        with open(img_list_file, 'r') as f:
            self.img_list = [img.replace("\n","") for img in f.readlines()]
        if self.train_supervised:
            self.img_list = self.img_list[:self.labeled_num]
    def __getitem__(self, index):
        if len(self.img_list[index].strip().split()) > 1:
            img_path, mask_path = self.img_list[index].strip().split()
        else:
            img_path = self.img_list[index]
            mask_path = img_path.replace("img","label")
        if index < self.labeled_num:
            assert os.path.isfile(mask_path),f"invalid mask path: {mask_path},index:{index}!"

        """get task name"""
        _,img_name = os.path.split(img_path)
        """read image and mask"""
        image = sitk.ReadImage(img_path)
        mask = sitk.ReadImage(mask_path)
        img_array = sitk.GetArrayFromImage(image) 
        mask_array = sitk.GetArrayFromImage(mask)
        mask_array = mask_array.astype(np.uint8)
        img_shape = img_array.shape
        
        if img_shape[0]< self.patch_size[0]:
            #need to extend data
            gap = self.patch_size[0]-img_shape[0]
            img_array_extend = np.zeros((self.patch_size[0], img_shape[1], img_shape[2]))
            mask_array_extend = np.zeros((self.patch_size[0], img_shape[1], img_shape[2]))
            img_array_extend[gap//2:gap//2+img_shape[0],:,:] = img_array
            mask_array_extend[gap//2:gap//2+img_shape[0],:,:] = mask_array
            img_array = img_array_extend
            mask_array = mask_array_extend
        if img_shape[1]< self.patch_size[1]:
                #need to extend data
            gap = self.patch_size[1]-img_shape[1]
            img_array_extend = np.zeros(( img_shape[0], self.patch_size[1],img_shape[2]))
            mask_array_extend = np.zeros(( img_shape[0], self.patch_size[1], img_shape[2]))
            img_array_extend[:,gap//2:gap//2+img_shape[0],:] = img_array
            mask_array_extend[:,gap//2:gap//2+img_shape[0],:] = mask_array
            img_array = img_array_extend
            mask_array = mask_array_extend
        if img_shape[2]< self.patch_size[2]:
                    #need to extend data
            gap = self.patch_size[2]-img_shape[2]
            img_array_extend = np.zeros(( img_shape[0],img_shape[1], self.patch_size[2]))
            mask_array_extend = np.zeros(( img_shape[0], img_shape[1], self.patch_size[2]))
            img_array_extend[:,:,gap//2:gap//2+img_shape[0]] = img_array
            mask_array_extend[:,:,gap//2:gap//2+img_shape[0]] = mask_array
            img_array = img_array_extend
            mask_array = mask_array_extend
        #print("mask array shape:", mask_array.shape)
  
        # 将灰度值在阈值之外的截断掉
        upper = self.upper
        lower = self.lower
        img_array[img_array > upper] = upper
        img_array[img_array < lower] = lower
        """Normalize the image"""
        img_array = (img_array - img_array.mean())/img_array.std()
        img_shape = img_array.shape
        
        """get image patch"""
        margin_len_x = self.patch_size[0]//2
        margin_len_y = self.patch_size[1]//2
        margin_len_z = self.patch_size[2]//2

        prob = random.random()
        
        center_x = img_shape[0]-margin_len_x
        if prob > self.edge_prob:
            center_x = random.randint(margin_len_x, img_shape[0]-margin_len_x)
        center_y = random.randint(margin_len_y, img_shape[1]-margin_len_y)
        center_z = random.randint(margin_len_z, img_shape[2]-margin_len_z)
        img_array = img_array[center_x-margin_len_x:center_x+margin_len_x,
                    center_y-margin_len_y:center_y+margin_len_y,\
                        center_z-margin_len_z:center_z+margin_len_z]
        mask_array = mask_array[center_x-margin_len_x:center_x+margin_len_x,\
                                center_y-margin_len_y:center_y+margin_len_y,\
                                center_z-margin_len_z:center_z+margin_len_z]
                #convert mask to one hot
        # do random cutout
        if self.cutout:
            mask_array = random_cutout(mask_array)
        img_array = torch.FloatTensor(img_array).unsqueeze(0)
        mask_array = torch.FloatTensor(mask_array).unsqueeze(0)
        gt_onehot = torch.zeros((self.num_class, mask_array.shape[1], mask_array.shape[2],mask_array.shape[3]))
        gt_onehot.scatter_(0, mask_array.long(), 1)
        mask_array = gt_onehot
        #mask_array = torch.FloatTensor(mask_array).unsqueeze(0)
        # do transformation
        
        if self.affine_trans:
            angle_x = random.uniform(-0.08,0.08)
            angle_y = random.uniform(-0.08,0.08)
            angle_z = random.uniform(-0.08,0.08)
            scale_x = random.uniform(0.8,1.2)
            scale_y = random.uniform(0.8,1.2)
            scale_z = random.uniform(0.8,1.2)     
            img = affine_transformation(img_array[np.newaxis,:], 
                                        radius=(angle_x, angle_y, angle_z), 
                                        translate=(0, 0, 0),
                                        scale=(scale_x, scale_y, scale_z),
                                        bspline_order=0, border_mode="nearest", 
                                        constant_val=0, is_reverse=False)
            mask = affine_transformation(mask_array[np.newaxis,:], 
                                        radius=(angle_x, angle_y, angle_z), 
                                        translate=(0, 0, 0),
                                        scale=(scale_x, scale_y, scale_z),
                                        bspline_order=0, border_mode="nearest",
                                        constant_val=0, is_reverse=False)
            img_array = img[0,:]
            mask_array = mask[0,:]
        sample = {'image': img_array, 
                  'label': mask_array.long()}
        return sample

    def __len__(self):
        return len(self.img_list)



def get_train_loaders(config):
    """
    Returns dictionary containing the training and validation loaders (torch.utils.data.DataLoader).

    :param config: a top level configuration object containing the 'loaders' key
    :return: dict {
        'train': <train_loader>
        'val': <val_loader>
    }
    """
    assert 'loaders' in config, 'Could not find data loaders configuration'
    loaders_config = config['loaders']

    print('Creating training and validation set loaders...')

    # get dataset class


    assert set(loaders_config['train']['file_paths']).isdisjoint(loaders_config['val']['file_paths']), \
        "Train and validation 'file_paths' overlap. One cannot use validation data for training!"

    
    train_dataset = Dataset(config['train_list_file'], patch_size=tuple(config['patch_size']))
    val_dataset = Dataset(config['val_list_file'], patch_size=tuple(config['patch_size']))
    
    num_workers = loaders_config.get('num_workers', 1)
    print(f'Number of workers for train/val dataloader: {num_workers}')
    batch_size = loaders_config.get('batch_size', 1)
    # if torch.cuda.device_count() > 1 and not config['device'].type == 'cpu':
    #     logger.info(
    #         f'{torch.cuda.device_count()} GPUs available. Using batch_size = {torch.cuda.device_count()} * {batch_size}')
    #     batch_size = batch_size * torch.cuda.device_count()

    print(f'Batch size for train/val loader: {batch_size}')
    # when training with volumetric data use batch_size of 1 due to GPU memory constraints
    return {
        'train': dataloader(train_dataset, batch_size=batch_size, shuffle=True,
                            num_workers=num_workers),
        # don't shuffle during validation: useful when showing how predictions for a given batch get better over time
        'val': dataloader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    }

if __name__ == '__main__':
    # test generic dataset
    # img_list_file = "/data/liupeng/hip_knee_segmentation/datasets/CTAndLabels_resample/train.txt"
    # test_dataset = Dataset(img_list_file,cutout=True,affine_trans=True)
    # test_dataloader = dataloader(test_dataset,batch_size=1)
    # for i,(img_array, mask_array) in enumerate(test_dataloader):
    #     print("step:{}, img_array shape:{}, mask array shape:{}".format(i, img_array.shape, mask_array.shape))
    
    #test generic dataset
    img_list_file = "/media/gdp/data/liupeng/3D_U-net/datasets/train.txt"
    test_dataset = DatasetSemi(img_list_file,num_class=6,cutout=True,affine_trans=True)
    #test_dataloader = dataloader(test_dataset,batch_size=2,shuffle=True)
    # dataloader with sampler
    # test_dataloader = dataloader(test_dataset, batch_sampler = BatchSampler(ClassRandomSampler(test_dataset), 2, True), num_workers=2, pin_memory=True)
    # #for i,(img_array, mask_array, task_name,ul1,br1,ul2,br2) in enumerate(test_dataloader):
    # for i,(img_array, mask_array, task_name) in enumerate(test_dataloader):
    #     print("step:{}, img_array shape:{}, mask array shape:{}, task_name:{}".format(i, img_array.shape, mask_array.shape, task_name))
    #     #print("step:{}, img_array shape:{}, mask array shape:{}, task_name:{}, ul1:{}, ul2:{}, br1:{}, br2:{}".format(i, img_array.shape, mask_array.shape, task_name,ul1,ul2,br1,br2))
    #     if torch.sum(task_name) > 0:
    #         new_mask_array= mask_array[:,[0,task_name[0]],:,:,:]
    #         #new_mask_array= mask_array[:,:,[0,task_name[0]],:,:,:]
    #         print("test")

        
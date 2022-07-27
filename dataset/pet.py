'''
Descripttion: 
version: 
Author: Luckie
Date: 2022-01-06 14:33:13
LastEditors: Luckie
LastEditTime: 2022-01-07 19:55:40
'''
import torch
import torch.nn.functional as F
import torchio as tio
from glob import glob
import SimpleITK as sitk
from torchvision import transforms, datasets 
from torch.utils.data import Dataset, DataLoader 
import numpy as np 
import os 
import random
from .data_augmentation import rotation,affine_transformation



def resize3d(image, shape, mode='trilinear', data_type='numpy'):
    """
        resize 3d image
    """
    if data_type == 'numpy':
        image = torch.tensor(image)[None,None,:,:,:]
    image = F.interpolate(torch.tensor(image), size=shape, mode=mode)
    image = image[0,0,:,:,:].numpy()
    return image

class PETDataset(Dataset):
    def __init__(self, image_list, transforms=None, affine=False):
        '''
        label: image label
        imageset: dataset
        image_list: txt file of image list
        '''
        # get the img list
        self.img_list = []
        with open(image_list,'r') as f:
            for line in f:
                if os.path.exists(line.replace("\n","")):
                    self.img_list.append(line.replace("\n",""))
        
        self.transforms = transforms
        self.affine = affine

    def __getitem__(self, index):
        img_file = self.img_list[index]
        img_path,img_name = os.path.split(img_file)
        # get the image label eg. t30_1.nii.gz
        label = img_name.split("_")[1][0]
        # get the img mask 
        has_gt_mask = 0 # whether traianing data has ground truth mask
        mask_file = img_file.replace(".nii","_seg.nii")
        # convert the img from nii to numpy
        img = sitk.GetArrayFromImage(sitk.ReadImage(img_file)) / 1.0 # divide 1.0 for type convert
        mask = np.zeros(img.shape)
        if os.path.exists(mask_file):
            has_gt_mask = 1
            mask = sitk.GetArrayFromImage(sitk.ReadImage(mask_file)) / 1.0


        ## TODO: image augmentation; tansform; resize the img
        # resize image
        #img = img * mask # filter the unrelevent region
        img = resize3d(img, (112, 64, 64))
        mask = resize3d(mask, (112, 64, 64),mode='nearest')
        #print("mask unique:",np.unique(mask))
        # random rotation

        #normalize image
        
        img = img / 50.0
        img[img>1.0] = 1.0
        img[img<0.0] = 0.0
        #img = (img-img.mean()) /img.std()
        #img = img[np.newaxis, :].repeat(2, axis=0)
        img = img[np.newaxis, :]
        # mask = mask[np.newaxis, :].repeat(3, axis=0)
        #img[1, :] = mask
        mask_0 = 1.0 - mask
        mask_1 = mask
        mask = mask[np.newaxis, :].repeat(2, axis=0)
        mask[0] = mask_0
        mask[1] = mask_1

        #rotate image and mask
        angle_x = random.uniform(-0.08,0.08)
        angle_y = random.uniform(-0.15,0.15)
        angle_z = random.uniform(-0.08,0.08)
        scale_x = random.uniform(0.8,1.2)
        scale_y = random.uniform(0.8,1.2)
        scale_z = random.uniform(0.8,1.2)

        # do affine transformation
        if self.affine:
            img = affine_transformation(img[np.newaxis,:], 
                                        radius=(angle_x, angle_y, angle_z), 
                                        translate=(0, 0, 0),
                                        scale=(scale_x, scale_y, scale_z),
                                        bspline_order=0, border_mode="nearest", 
                                        constant_val=0, is_reverse=False)
            mask = affine_transformation(mask[np.newaxis,:], 
                                        radius=(angle_x, angle_y, angle_z), 
                                        translate=(0, 0, 0),
                                        scale=(scale_x, scale_y, scale_z),
                                        bspline_order=0, border_mode="nearest",
                                        constant_val=0, is_reverse=False)
            img = img[0,:]
            mask = mask[0,:]
        sample = {'image': img[0], 'label': mask[1,:].astype(np.uint8),'img_name':img_name,'has_gt_mask':has_gt_mask}
        if self.transforms:
            sample = self.transforms(sample)
        return sample
    
    def __len__(self):
        return len(self.img_list)

class PETAugDataset(Dataset):
    def __init__(self, image_path, transforms=None):
        '''
        label: image label
        imageset: dataset
        '''
        # get the img list
        self.img_list = []
        with open(image_path,'r') as f:
            for line in f:
                self.img_list.append(line.replace("\n",""))
        
        self.transforms = transforms

    def __getitem__(self, index):
        img_file = self.img_list[index]
        img_path,img_name = os.path.split(img_file)
        # get the image label eg. t30_1.nii.gz
        label = img_name.split("_")[1][0]
        # get the img mask 
        mask_file = img_file.replace(".nii","_seg.nii")
        # convert the img from nii to numpy
        img = sitk.GetArrayFromImage(sitk.ReadImage(img_file)) / 1.0 # divide 1.0 for type convert
        mask = sitk.GetArrayFromImage(sitk.ReadImage(mask_file)) / 1.0






        ## TODO: image augmentation; tansform; resize the img
        # resize image

        #normalize image
        #img = img * mask # filter the unrelevent region
        #img = (img - img.min()) / (img.max() - img.min()) * 1.0
        #img = img.astype(np.float)
        img = img / 50.0
        img[img>1.0] = 1.0
        img[img<0.0] = 0.0
        #img = img[np.newaxis, :].repeat(2, axis=0)
        img = img[np.newaxis, :]
        # mask = mask[np.newaxis, :].repeat(3, axis=0)
        #img[1, :] = mask
        mask = mask[np.newaxis, :]

        #data augmentation
        if self.transforms:
            img = self.transforms(img)
            mask = self.transforms(mask)
        #img = img.repeat(2, axis=0)
        #img[0,:] = mask[0]
        #print("img shape:",img.shape)


        return img, torch.from_numpy(np.array(int(label))), mask 
    
    def __len__(self):
        return len(self.img_list)

# add data augmentation 
def get_PET_aug_dataset(image_path,transform,is_test=False):
    # get data subjects
    img_list = []
    with open(image_path,'r') as f:
        for line in f:
            img_list.append(line.replace("\n",""))
    subjects = []
    for img_file in img_list:
        label = int(img_file.split("_")[-1][0])
        file_path, img_name = os.path.split(img_file)
        mask_file = img_file.replace(".nii","_seg.nii")
        img = sitk.GetArrayFromImage(sitk.ReadImage(img_file)) / 1.0 
        mask = sitk.GetArrayFromImage(sitk.ReadImage(mask_file)) / 1.0
        #normalize image
        img = img / 50.0
        img[img>1.0] = 1.0
        img[img<0.0] = 0.0
        x,y,z = img.shape
        if is_test:
            for i in range(0,10):
                #random crop
                x_r = np.random.randint(0,x-224)
                z_r = np.random.randint(0,z-64)
                img_tmp = img[np.newaxis, x_r:x_r+224, :,z_r:z_r+64]
                #mask_tmp = mask[np.newaxis, x_r:x_r+224, :,z_r:z_r+64]
                mask_tmp = np.zeros([1,224,64,64])
                subject = tio.Subject(
                    name=img_name,
                    pet=tio.ScalarImage(tensor=img_tmp),
                    mask=tio.LabelMap(tensor=mask_tmp),
                    label=label
                )
                subjects.append(subject)
        else:
            #random crop
            x_r = np.random.randint(0, x-224)
            z_r = np.random.randint(0, z-64)
            img = img[np.newaxis, x_r:x_r+224, :,z_r:z_r+64]
            mask = mask[np.newaxis, x_r:x_r+224, :,z_r:z_r+64]
            subject = tio.Subject(
                name=img_name,
                pet=tio.ScalarImage(tensor=img),
                mask=tio.LabelMap(tensor=mask),
                label=label
            )
            subjects.append(subject)
        

    training_set = tio.SubjectsDataset(
    subjects, transform=transform)
    return training_set
    



    


# test the dataset class
# train_datasets = HipDataset("../3dHipData/test.txt")

# train_dataloader = DataLoader(train_datasets, batch_size=1, shuffle=False)
if __name__ == "__main__":
    dataset = get_PET_aug_dataset("../pet_additional/fold{}_test.txt".format(str(0)), validation_transform)
    dataloader = DataLoader(dataset,batch_size=2)
    for i,subjects_batch in enumerate(dataloader):
        inputs = subjects_batch['pet'][tio.DATA]
        targets = subjects_batch['label']
        mask = subjects_batch['mask'][tio.DATA]
    # for img, label, mask in train_dataloader:
    #     print("img shape:{},mask shape:{}".format(img.shape,mask.shape))
    #     print("img:",img[0,0,:])
    #     break




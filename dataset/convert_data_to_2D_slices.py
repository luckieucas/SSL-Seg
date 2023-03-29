import glob
import os

import h5py
import numpy as np
import SimpleITK as sitk
import argparse
from tqdm import tqdm


parser = argparse.ArgumentParser()
parser.add_argument("--img_list",type=str,
                    default='/data/liupeng/semi-supervised_segmentation/SSL4MIS-master/data/BCV/train.txt', 
                    help='image file list txt file')

def convert_h5_file(args):
    with open(args.img_list, 'r') as f:
        img_list = [line.replace("\n","").strip() for line in f.readlines()]
    slice_num = 0
    
    for case in tqdm(img_list):
        if len(case.split())>1:
            image_path = case.split()[0]
            mask_path = case.split()[1]
        else:
            image_path = case 
            mask_path = case.replace("img","label")
        root_path,_ = os.path.split(image_path)
        img_itk = sitk.ReadImage(image_path)
        origin = img_itk.GetOrigin()
        spacing = img_itk.GetSpacing()
        direction = img_itk.GetDirection()
        image = sitk.GetArrayFromImage(img_itk)
        if os.path.exists(mask_path):
            msk_itk = sitk.ReadImage(mask_path)
            mask = sitk.GetArrayFromImage(msk_itk)
            image = (image - image.min()) / (image.max() - image.min())
            print(image.shape)
            image = image.astype(np.float32)
            item = case.split("/")[-1].split(".")[0].split("_")[0]
            if image.shape != mask.shape:
                print("Error")
            f = h5py.File(
                '{}/{}.h5'.format(args.save_path,item), 'w')
            f.create_dataset(
                'image', data=image, compression="gzip")
            f.create_dataset('label', data=mask, compression="gzip")
            f.close()
    print("Converted all ACDC volumes to h5 file")
    print("Total {} slices".format(slice_num))

def convert_2d_slices(args):
    with open(args.img_list, 'r') as f:
        img_list = [line.replace("\n","").strip() for line in f.readlines()]
    slice_num = 0
    
    for case in tqdm(img_list):
        if len(case.split())>1:
            image_path = case.split()[0]
            mask_path = case.split()[1]
        else:
            image_path = case 
            mask_path = case.replace("img","label")
        root_path,_ = os.path.split(image_path)
        img_itk = sitk.ReadImage(image_path)
        origin = img_itk.GetOrigin()
        spacing = img_itk.GetSpacing()
        direction = img_itk.GetDirection()
        image = sitk.GetArrayFromImage(img_itk)
        if os.path.exists(mask_path):
            msk_itk = sitk.ReadImage(mask_path)
            mask = sitk.GetArrayFromImage(msk_itk)
            image = (image - image.min()) / (image.max() - image.min())
            print(image.shape)
            image = image.astype(np.float32)
            item = case.split("/")[-1].split(".")[0].split("_")[0]
            print(item)
            if image.shape != mask.shape:
                print("Error")
            for slice_ind in range(image.shape[0]):
                f = h5py.File(
                    '{}/{}_slice_{}.h5'.format(args.save_path, item, slice_ind), 'w')
                f.create_dataset(
                    'image', data=image[slice_ind], compression="gzip")
                f.create_dataset('label', data=mask[slice_ind], compression="gzip")
                f.close()
                slice_num += 1
    print("Converted all ACDC volumes to 2D slices")
    print("Total {} slices".format(slice_num))

def generate_train_test_list_file(args,is_train=True):
    with open(args.img_list, 'r') as f:
        img_list = [line.replace("\n","").strip() for line in f.readlines()]
    slice_path = args.save_path
    with open("train.list",'w') as f:
        for case in tqdm(img_list):
            if len(case.split())>1:
                image_path = case.split()[0]
                mask_path = case.split()[1]
            else:
                image_path = case 
                mask_path = case.replace("img","label")
            print(mask_path)
            item = image_path.split("/")[-1].split("_")[0]
            print(item)
            if is_train:
                slice_list = glob.glob(slice_path+f"/{item}_slice_*")
                f.writelines(f"{os.path.split(line)[1]}\n" for line in slice_list)
                print(slice_list)
            else:
                f.writelines(f"{item}\n")
           

if __name__=="__main__":
    args = parser.parse_args()
    args.save_path = "/data/liupeng/semi-supervised_segmentation/nnUNetFrame/DATASET/nnUNet_raw/nnUNet_raw_data/Task11_BCV/Training/slices/"
    #main(args)
    generate_train_test_list_file(args,True)
    #convert_h5_file(args)
    #convert_2d_slices(args)

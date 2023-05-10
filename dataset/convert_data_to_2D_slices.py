import glob
import os

import h5py
import numpy as np
import SimpleITK as sitk
import argparse
from tqdm import tqdm


parser = argparse.ArgumentParser()
parser.add_argument("--img_list",type=str,
                    default='/data1/liupeng/semi-supervised_segmentation/SSL4MIS-master/data/MMWHS/MMWHS_test.txt', 
                    help='image file list txt file')

def convert_h5_file(args,cut_lower=None,cut_upper=None):
    with open(args.img_list, 'r') as f:
        img_list = [line.replace("\n","").strip() for line in f.readlines()]
    slice_num = 0
    
    for case in tqdm(img_list):
        print(f"processing:{case}")
        if len(case.split())>1:
            image_path = case.split()[0]
            mask_path = case.split()[1]
        else:
            image_path = case 
            mask_path = case.replace("img","label")
        image_path = image_path.replace("_crop","")
        mask_path = mask_path.replace("_crop","")
        root_path,_ = os.path.split(image_path)
        img_itk = sitk.ReadImage(image_path)
        image = sitk.GetArrayFromImage(img_itk)
        if os.path.exists(mask_path):
            msk_itk = sitk.ReadImage(mask_path)
            mask = sitk.GetArrayFromImage(msk_itk)
            h,w,d = mask.shape
            boud_h, boud_w, boud_d = np.where(mask >= 1)
            bbx_h_min = boud_h.min()
            bbx_h_max = boud_h.max()
            bbx_w_min = boud_w.min()
            bbx_w_max = boud_w.max()
            bbx_d_min = boud_d.min()
            bbx_d_max = boud_d.max()
            mask = mask[
                max(bbx_h_min,0):min(bbx_h_max,h),
                16:464,#max(bbx_w_min-60,0):min(bbx_w_max+60,w),
                5:465 #max(bbx_d_min-50,0):min(bbx_d_max+50,d)
            ]
            image = image[
                max(bbx_h_min,0):min(bbx_h_max,h),
                16:464,#max(bbx_w_min-60,0):min(bbx_w_max+60,w),
                5:465 #max(bbx_d_min-50,0):min(bbx_d_max+50,d)
            ]
            if cut_lower:
                np.clip(image,cut_lower,cut_upper,out=image)
            image = (image - image.mean()) / image.std()
            print(f"image shape:{image.shape}")
            image = image.astype(np.float32)
            mask = mask.astype(np.int32)
            item = case.split("/")[-1].split(".")[0].split("_crop")[0]
            print("item:",item)
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

def convert_2d_slices(args,cut_lower=None,cut_upper=None):
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
        image_path = image_path.replace("_crop","")
        mask_path = mask_path.replace("_crop","")
        root_path,_ = os.path.split(image_path)
        img_itk = sitk.ReadImage(image_path)
        image = sitk.GetArrayFromImage(img_itk)
        if os.path.exists(mask_path):
            msk_itk = sitk.ReadImage(mask_path)
            mask = sitk.GetArrayFromImage(msk_itk)
            h,w,d = mask.shape
            boud_h, boud_w, boud_d = np.where(mask >= 1)
            bbx_h_min = boud_h.min()
            bbx_h_max = boud_h.max()
            bbx_w_min = boud_w.min()
            bbx_w_max = boud_w.max()
            bbx_d_min = boud_d.min()
            bbx_d_max = boud_d.max()
            mask = mask[
                max(bbx_h_min,0):min(bbx_h_max,h),
                16:464,#max(bbx_w_min-60,0):min(bbx_w_max+60,w),
                5:465 #max(bbx_d_min-50,0):min(bbx_d_max+50,d)
            ]
            image = image[
                max(bbx_h_min,0):min(bbx_h_max,h),
                16:464,#max(bbx_w_min-60,0):min(bbx_w_max+60,w),
                5:465 #max(bbx_d_min-50,0):min(bbx_d_max+50,d)
            ]
            print(f"image shape:{image.shape}")
            if cut_lower:
                np.clip(image,cut_lower,cut_upper,out=image)
            image = (image - image.mean()) / image.std()
            image = image.astype(np.float32)
            item = case.split("/")[-1].split(".")[0].split("_crop")[0]
            print(item)
            assert image.shape == mask.shape, "Image and mask shape mismatch"
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
    with open("test.list",'w') as f:
        for case in tqdm(img_list):
            if len(case.split())>1:
                image_path = case.split()[0]
                mask_path = case.split()[1]
            else:
                image_path = case 
                mask_path = case.replace("img","label")
            print(mask_path)
            item = case.split("/")[-1].split(".")[0].split("_crop")[0]
            print(item)
            if is_train:
                slice_list = glob.glob(slice_path+f"/{item}_slice_*")
                f.writelines(f"{os.path.split(line)[1]}\n" for line in slice_list)
                print(slice_list)
            else:
                f.writelines(f"{item}\n")
           

if __name__=="__main__":
    args = parser.parse_args()
    args.save_path = "/data1/liupeng/semi-supervised_segmentation/dataset/Task012_Heart/data/"
    cut_lower = -200
    cut_upper = 300
    #main(args)
    #generate_train_test_list_file(args,is_train=False)
    convert_h5_file(args,-200,300)
    #convert_2d_slices(args,cut_lower,cut_upper)

# processing the dataset
import numpy as np 
import SimpleITK as sitk 
from tqdm import tqdm



def main(img_file):
    with open(img_file, 'r') as f:
        img_list = [img.replace("\n","") for img in f.readlines()]
    for img_mask_path in tqdm(img_list):
        #img_path, mask_path = img_mask_path.strip().split()
        if len(img_mask_path.strip().split()) > 1:
            img_path, mask_path = img_mask_path.strip().split()
        else:
            img_path = img_mask_path
            mask_path = img_path.replace('img','label')
        print(f"=====>processing:{img_path}")
        img_itk = sitk.ReadImage(img_path)
        spacing = img_itk.GetSpacing()
        direction = img_itk.GetDirection()
        origin = img_itk.GetOrigin()
        mask_np = sitk.GetArrayFromImage(sitk.ReadImage(mask_path))
        image_np = sitk.GetArrayFromImage(img_itk)
        h,w,d = mask_np.shape
        boud_h, boud_w, boud_d = np.where(mask_np >= 1)
        print(boud_h)
        bbx_h_min = boud_h.min()
        bbx_h_max = boud_h.max()
        bbx_w_min = boud_w.min()
        bbx_w_max = boud_w.max()
        bbx_d_min = boud_d.min()
        bbx_d_max = boud_d.max()
        bbx_mask = mask_np[
            0:h,#max(bbx_h_min-25,0):min(bbx_h_max+25,h),
            max(bbx_w_min-60,0):min(bbx_w_max+60,w),
            max(bbx_d_min-50,0):min(bbx_d_max+50,d)
        ]
        bbx_image = image_np[
            0:h, #max(bbx_h_min-25,0):min(bbx_h_max+25,h),
            max(bbx_w_min-60,0):min(bbx_w_max+60,w),
            max(bbx_d_min-50,0):min(bbx_d_max+50,d)
        ]
        bbx_mask  = np.flip(bbx_mask,axis=1)
        bbx_image  = np.flip(bbx_image,axis=1)
        assert bbx_mask.shape == bbx_image.shape, "shape must equal after crop"
        print(f"before crop shape:{mask_np.shape}, after crop shape:{bbx_mask.shape}")
        bbx_mask_itk = sitk.GetImageFromArray(bbx_mask)
        bbx_mask_itk.SetOrigin(origin)
        bbx_mask_itk.SetSpacing(spacing)
        bbx_mask_itk.SetDirection(direction)
        sitk.WriteImage(bbx_mask_itk,mask_path.replace(".nii.gz","_crop.nii.gz"))
        
        bbx_image_itk = sitk.GetImageFromArray(bbx_image)
        bbx_image_itk.SetOrigin(origin)
        bbx_image_itk.SetSpacing(spacing)
        bbx_image_itk.SetDirection(direction)
        sitk.WriteImage(bbx_image_itk,img_path.replace(".nii.gz","_crop.nii.gz"))

if __name__ == "__main__":
    print("test")
    img_file = "/data/liupeng/semi-supervised_segmentation/SSL4MIS-master/data/Pancreas/pancreas_test.txt"
    main(img_file)
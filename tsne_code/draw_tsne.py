import SimpleITK as sitk 
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn import manifold
import pandas as pd 
import seaborn as sns
from batchgenerators.utilities.file_and_folder_operations import load_json

def draw_tsne_save(X, y, save_name="fea_vis.png"):
    '''t-SNE'''
    
    tsne = manifold.TSNE(n_components=2, init='pca', random_state=501)
    print(f"label unique: {np.unique(y)}")
    X_tsne = tsne.fit_transform(X)
    print(X.shape)
    print(X_tsne.shape)
    print(y.shape)
    color = ['m','r','g','b','y','c'] # color for differen class
    print("Org data dimension is {}.Embedded data dimension is {}".format(X.shape[-1], X_tsne.shape[-1]))

    '''embedding space visualization'''
    x_min, x_max = X_tsne.min(0), X_tsne.max(0)
    X_tsne = (X_tsne - x_min) / (x_max -   x_min)  # norm
    X_norm = X_tsne
    X_tsne_df = pd.DataFrame(X_tsne).rename(columns={0:'dim1',1:'dim2'})
    y_df = pd.DataFrame(y[:,np.newaxis]).rename(columns={0:'class'})
    data_tsne = pd.concat([X_tsne_df,y_df], axis=1)
    plt.figure(figsize=(8, 8))
    sns.scatterplot(data=data_tsne, hue='class',x='dim1',y='dim2')
    print("data tsne shape: ", data_tsne.shape)
    plt.savefig("img0039_tsne_all_baseline.png")
    
    # plt.figure(figsize=(8, 8))
    # sns.scatterplot(data=data_tsne[15810:], hue='class',x='dim1',y='dim2')
    # print("data tsne shape: ", data_tsne.shape)
    # plt.savefig("tsne_2.png")


def draw_tsne():
    feature_large_patch = np.load("../data/BCV/img0039_feature_small_patch_baseline_1.npy")
    feature_small_patch = np.load("../data/BCV/img0039_feature_small_patch_baseline_2.npy")
    print(f"feature large shape: {feature_large_patch.shape}")
    print(f"feature small shape: {feature_small_patch.shape}")
    
    # Read mask
    label_large_patch = sitk.GetArrayFromImage(sitk.ReadImage("../data/BCV/overlap_patches/label0039_crop_96_160_160_1.nii.gz"))
    label_small_patch = sitk.GetArrayFromImage(sitk.ReadImage("../data/BCV/overlap_patches/label0039_crop_96_160_160_2.nii.gz"))
    
    coords = load_json("../data/BCV/overlap_patches/img0039_crop_large96_small96_coords.json")
    #label_large_patch = label_large_patch[32:128,48:208,64:224]
    print(f"large label shape: {label_large_patch.shape}")
    print(f"small label shape: {label_small_patch.shape}")
    
    # down sample label
    label_arr_large = nn.functional.interpolate(torch.from_numpy(label_large_patch).unsqueeze(0).unsqueeze(0).type(torch.float32),
                                          size=(48,80,80),mode='nearest').numpy()
    label_arr_small = nn.functional.interpolate(torch.from_numpy(label_small_patch).unsqueeze(0).unsqueeze(0).type(torch.float32),
                                          size=(48,80,80),mode='nearest').numpy()
    # down sample small patch feature
    # feature_small_patch = nn.functional.interpolate(torch.from_numpy(feature_small_patch),
    #                                       size=(36,61,44)).numpy()
    # label_arr_small = nn.functional.interpolate(torch.from_numpy(label_arr_small),
    #                                       size=(36,61,44),mode='nearest').numpy()
    
    # large_overlap_x_l, large_overlap_x_h = 16, 48
    # large_overlap_y_l, large_overlap_y_h = 0, 58
    # large_overlap_z_l, large_overlap_z_h = 0, 36
    # small_overlap_x_l, small_overlap_x_h = 0, 32
    # small_overlap_y_l, small_overlap_y_h = 3, 61
    # small_overlap_z_l, small_overlap_z_h = 8, 44
    ul1 = [int(i/2) for i in coords['ul1']]
    br1 = [int(i/2) for i in coords['br1']]
    ul2 = [int(i/2) for i in coords['ul2']]
    br2 = [int(i/2) for i in coords['br2']]
    
    large_overlap_x_l, large_overlap_x_h = ul1[0], br1[0]
    large_overlap_y_l, large_overlap_y_h = ul1[1], br1[1]
    large_overlap_z_l, large_overlap_z_h = ul1[2], br1[2]
    small_overlap_x_l, small_overlap_x_h = ul2[0], br2[0]
    small_overlap_y_l, small_overlap_y_h = ul2[1], br2[1]
    small_overlap_z_l, small_overlap_z_h = ul2[2], br2[2]
    
    print(f"feature_small_patch shape: {feature_small_patch.shape}")
    feature_large_patch_overlap = feature_large_patch[0,:,large_overlap_x_l:large_overlap_x_h,
                                                      large_overlap_y_l:large_overlap_y_h,
                                                      large_overlap_z_l:large_overlap_z_h]
    feature_small_patch_overlap = feature_small_patch[0,:,small_overlap_x_l:small_overlap_x_h,
                                                      small_overlap_y_l:small_overlap_y_h,
                                                      small_overlap_z_l:small_overlap_z_h]
    print(f"feature large patch overlap: {feature_large_patch_overlap.shape}")
    print(f"feature small patch overlap: {feature_small_patch_overlap.shape}")
    label_arr_large_overlap = label_arr_large[0,0,large_overlap_x_l:large_overlap_x_h,
                                      large_overlap_y_l:large_overlap_y_h,
                                      large_overlap_z_l:large_overlap_z_h]
    label_arr_small_overlap = label_arr_small[0,0,small_overlap_x_l:small_overlap_x_h,
                                              small_overlap_y_l:small_overlap_y_h,
                                              small_overlap_z_l:small_overlap_z_h]
    print(f"label_arr_large_overlap : {label_arr_large_overlap.shape}")
    print(f"label_arr_small_overlap : {label_arr_small_overlap.shape}")
    print(f"different label sum: {(label_arr_small_overlap==label_arr_large_overlap).sum()}")
    print(f"forground sum: {(label_arr_small_overlap>0).sum()}")
    print(np.unique(label_arr_large))
    
    # flatten feature
    feature_large_patch_overlap_flattened = np.transpose(feature_large_patch_overlap.reshape(32,-1),(1,0))
    feature_small_patch_overlap_flattened = np.transpose(feature_small_patch_overlap.reshape(32,-1),(1,0))
    print(f"feature_large_patch_overlap_flattened shape: {feature_large_patch_overlap_flattened.shape}")
    print(f"feature_small_patch_overlap_flattened shape: {feature_small_patch_overlap_flattened.shape}")
    
    #flatten label
    label_arr_large_overlap_flattened = label_arr_large_overlap.flatten()
    label_arr_small_overlap_flattened = label_arr_small_overlap.flatten()
    print(f"label_arr_large_overlap_flattened shape: {label_arr_large_overlap_flattened.shape}")
    print(f"label_arr_small_overlap_flattened shape: {label_arr_small_overlap_flattened.shape}")
    
    #filter background 
    feature_large_patch_overlap_flattened = feature_large_patch_overlap_flattened[label_arr_large_overlap_flattened>0]
    print(f"feature_large_patch_overlap_flattened after filter shape: {feature_large_patch_overlap_flattened.shape}")
    feature_small_patch_overlap_flattened = feature_small_patch_overlap_flattened[label_arr_small_overlap_flattened>0]
    
    feature_all = np.concatenate((feature_large_patch_overlap_flattened,
                                  feature_small_patch_overlap_flattened), axis=0)
    label_all = np.concatenate((label_arr_large_overlap_flattened[label_arr_large_overlap_flattened>0],
                                label_arr_small_overlap_flattened[label_arr_small_overlap_flattened>0]),axis=0)
    print(f"feature all shape:{feature_all.shape}")
    print(f"label all shape:{label_all.shape}")
    draw_tsne_save(feature_all, label_all)
    # draw_tsne_save(feature_small_patch_overlap_flattened,
    #               label_arr_small_overlap_flattened[label_arr_small_overlap_flattened>0])
    

if __name__ == '__main__':
    draw_tsne()
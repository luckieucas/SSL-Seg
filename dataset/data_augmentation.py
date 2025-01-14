'''
Descripttion: 
version: 
Author: Luckie
Date: 2022-01-06 14:34:34
LastEditors: Luckie
LastEditTime: 2022-01-06 14:34:35
'''
import numpy as np
import torch
from batchgenerators.augmentations.utils import interpolate_img, create_zero_centered_coordinate_mesh
from medpy.io import load, save
from math import ceil
import random


def create_matrix_rotation_z_3d(angle, matrix=None):
    rotation_z = np.array([[np.cos(angle), -np.sin(angle), 0],
                           [np.sin(angle), np.cos(angle), 0],
                           [0, 0, 1]])
    if matrix is None:
        return rotation_z

    return np.dot(matrix, rotation_z)


def create_matrix_rotation_x_3d(angle, matrix=None):
    rotation_x = np.array([[1, 0, 0],
                           [0, np.cos(angle), -np.sin(angle)],
                           [0, np.sin(angle), np.cos(angle)]])
    if matrix is None:
        return rotation_x

    return np.dot(matrix, rotation_x)


def create_matrix_rotation_y_3d(angle, matrix=None):
    rotation_y = np.array([[np.cos(angle), 0, -np.sin(angle)],
                           [0, 1, 0],
                           [np.sin(angle), 0, np.cos(angle)]])
    if matrix is None:
        return rotation_y

    return np.dot(matrix, rotation_y)


def batch_crop(volume, patch_size, seg=None, patch_ids=None):
    """
    # TODO: optimizing this function, it has many bugs!!!
    :param volume: tensor (batch, channel, h, w, d)
    :param patch_size: (h, w, d)
    :param seg:
    :param repeats
    :param patch_ids: None for random crop, tuple for structure crop
    :return: patch_volume, (patch_seg if seg), bounds

    """
    # done: validating this function whether output right region
    shape = volume.shape[2:]
    if patch_ids is None:

        x_range = shape[0] - patch_size[0]
        y_range = shape[1] - patch_size[1]
        z_range = shape[2] - patch_size[2]
        if x_range == 0:
            x_random = int(0)
        else:
            x_random = np.random.randint(0, x_range, dtype=np.int)
        if y_range == 0:
            y_random = int(0)
        else:
            y_random = np.random.randint(0, y_range, dtype=np.int)
        if z_range == 0:
            z_random = int(0)
        else:
            z_random = np.random.randint(0, z_range, dtype=np.int)
        if seg is not None:
            return volume[..., x_random:x_random + patch_size[0], y_random:y_random + patch_size[1],
                   z_random:z_random + patch_size[2]], \
                   seg[..., x_random:x_random + patch_size[0], y_random:y_random + patch_size[1],
                   z_random:z_random + patch_size[2]], \
                   [x_random, x_random + patch_size[0], y_random, y_random + patch_size[1],
                    z_random, z_random + patch_size[2]]
        else:
            return volume[..., x_random:x_random + patch_size[0], y_random:y_random + patch_size[1],
                   z_random:z_random + patch_size[2]], \
                   [x_random, x_random + patch_size[0], y_random, y_random + patch_size[1],
                    z_random, z_random + patch_size[2]]
    else:
        # DONE implement overlay patch crop
        start_lists = []
        repeats = []
        for i, j in zip(shape, patch_size):
            if i == j:
                repeats.append(1)
            else:
                repeats.append(int(i // j + 1))
        for rep, vs, ps in zip(repeats, shape, patch_size):
            final_index = vs - ps
            if rep == 1:
                step = final_index
            else:
                step = final_index // (rep - 1)
            if final_index == 0:
                start_list = [0]
            else:
                start_list = [start_point for start_point in range(0, final_index, step)]
            if len(start_list) == rep:
                start_list[-1] = final_index
            else:
                start_list += [final_index]
            start_lists.append(start_list)
        # repeat_x, repeat_y, repeat_z = ceil(shape[0] / patch_size[0]), ceil(
        #     shape[1] / patch_size[1]), ceil(shape[2] / patch_size[2])
        # if patch_ids[0] != repeat_x - 1:
        #     bound_x = patch_ids[0] * patch_size[0]
        # else:
        #     bound_x = shape[0] - patch_size[0]
        # if patch_ids[1] != repeat_y - 1:
        #     bound_y = patch_ids[1] * patch_size[1]
        # else:
        #     bound_y = shape[1] - patch_size[1]
        # if patch_ids[2] != repeat_z - 1:
        #     bound_z = patch_ids[2] * patch_size[2]
        # else:
        #     bound_z = shape[2] - patch_size[2]
        _s = start_lists
        _p = patch_ids
        if seg is not None:
            return volume[..., _s[0][_p[0]]:_s[0][_p[0]] + patch_size[0], _s[1][_p[1]]:_s[1][_p[1]] + patch_size[1],
                   _s[2][_p[2]]:_s[2][_p[2]] + patch_size[2]], \
                   seg[..., _s[0][_p[0]]:_s[0][_p[0]] + patch_size[0], _s[1][_p[1]]:_s[1][_p[1]] + patch_size[1],
                   _s[2][_p[2]]:_s[2][_p[2]] + patch_size[2]], \
                   [_s[0][_p[0]], _s[0][_p[0]] + patch_size[0], _s[1][_p[1]], _s[1][_p[1]] + patch_size[1],
                    _s[2][_p[2]], _s[2][_p[2]] + patch_size[2]]
        else:
            return volume[..., _s[0][_p[0]]:_s[0][_p[0]] + patch_size[0], _s[1][_p[1]]:_s[1][_p[1]] + patch_size[1],
                   _s[2][_p[2]]:_s[2][_p[2]] + patch_size[2]], \
                   [_s[0][_p[0]], _s[0][_p[0]] + patch_size[0], _s[1][_p[1]], _s[1][_p[1]] + patch_size[1],
                    _s[2][_p[2]], _s[2][_p[2]] + patch_size[2]]


def affine_transformation(vol, radius, translate, scale, bspline_order, border_mode, constant_val, is_reverse):
    """
    # done: debug multi class mask (one hot and no one hot), no error
    forward: scale -> rotation -> translation
    backward: translation -> rotation -> scale
    :param vol: np.ndarray or torch.tensor(batch, channel, h, w, d)
    :param radius: tuple
    :param translate: tuple
    :param scale: tuple
    :param bspline_order: 0~5 0:nearest, 1:bilinear, 2~5 bspline, 3 is common used.
    :param border_mode: "constant", "nearest"
    :param constant_val: contant value
    :param is_reverse: bool
    :return:
    """
    tensor_flag = False
    _device = torch.device("cpu")
    if isinstance(vol, torch.Tensor):
        _device = vol.device
        vol = vol.cpu().detach().numpy()
        tensor_flag = True
    shape = vol.shape[2:]
    dim = len(vol.shape) - 2
    center = tuple((i - 1) / 2. for i in shape)
    tmp = tuple([np.arange(i) for i in shape])
    coords = np.array(np.meshgrid(*tmp, indexing='ij')).astype(float)
    for d in range(len(shape)):
        coords[d] -= center[d]  # centered coords

    if is_reverse:
        # translation first
        for i in range(dim):
            coords[i] -= translate[i]
        # rotation
        rot_matrix = np.identity(len(coords))
        rot_matrix = create_matrix_rotation_z_3d(radius[2], rot_matrix)
        rot_matrix = create_matrix_rotation_y_3d(radius[1], rot_matrix)
        rot_matrix = create_matrix_rotation_x_3d(radius[0], rot_matrix)
        coords = np.dot(coords.reshape(len(coords), -1).transpose(), rot_matrix).transpose().reshape(coords.shape)
        # scale
        if isinstance(scale, (tuple, list, np.ndarray)):  # scale axis individual
            assert len(scale) == len(coords)
            for i in range(len(scale)):
                coords[i] *= scale[i]
        else:
            coords *= scale  # scale axis both
    else:
        # scale first
        if isinstance(scale, (tuple, list, np.ndarray)):  # scale axis individual
            assert len(scale) == len(coords)
            for i in range(len(scale)):
                coords[i] *= scale[i]
        else:
            coords *= scale  # scale axis both
        # rotation
        rot_matrix = np.identity(len(coords))
        rot_matrix = create_matrix_rotation_x_3d(radius[0], rot_matrix)
        rot_matrix = create_matrix_rotation_y_3d(radius[1], rot_matrix)
        rot_matrix = create_matrix_rotation_z_3d(radius[2], rot_matrix)
        coords = np.dot(coords.reshape(len(coords), -1).transpose(), rot_matrix).transpose().reshape(coords.shape)
        # translate
        for i in range(dim):
            coords[i] -= translate[i]

    for i in range(dim):
        coords[i] += center[i]
    data_result = np.zeros_like(vol, dtype=np.float32)
    for sample_id in range(vol.shape[0]):
        for channel_id in range(vol.shape[1]):
            data_result[sample_id, channel_id] = interpolate_img(vol[sample_id, channel_id], coords,
                                                                 order=bspline_order,
                                                                 mode=border_mode, cval=constant_val)
    if tensor_flag:
        data_result = torch.from_numpy(data_result)
        data_result = data_result.to(_device)
    return data_result


def rotate_coords_3d(coords, angle_x, angle_y, angle_z, is_reverse):
    """reverse rotate data, so the rotating order is z, y, x"""
    rot_matrix = np.identity(len(coords))
    if is_reverse:
        rot_matrix = create_matrix_rotation_z_3d(angle_z, rot_matrix)
        rot_matrix = create_matrix_rotation_y_3d(angle_y, rot_matrix)
        rot_matrix = create_matrix_rotation_x_3d(angle_x, rot_matrix)
    else:
        rot_matrix = create_matrix_rotation_x_3d(angle_x, rot_matrix)
        rot_matrix = create_matrix_rotation_y_3d(angle_y, rot_matrix)
        rot_matrix = create_matrix_rotation_z_3d(angle_z, rot_matrix)
    coords = np.dot(coords.reshape(len(coords), -1).transpose(), rot_matrix).transpose().reshape(coords.shape)
    return coords


def translate_coords_3d(coords, t_x, t_y, t_z):
    coords[0] -= t_x
    coords[1] -= t_y
    coords[2] -= t_z
    return coords


def shift_center(coords, patch_size):
    for d in range(3):
        ctr = ((np.array(patch_size).astype(float) - 1) / 2.)[d]
        coords[d] += ctr
    return coords


def batch_deform_3d(vol, coords, bspline_order, border_mode, constant_val):
    data_result = np.zeros_like(vol, dtype=np.float32)
    for sample_id in range(vol.shape[0]):
        for channel_id in range(vol.shape[1]):
            data_result[sample_id, channel_id] = interpolate_img(vol[sample_id, channel_id], coords,
                                                                 order=bspline_order,
                                                                 mode=border_mode, cval=constant_val)
    return data_result

def rotation(i_volume,shape,angle):
    p = np.pi
    i_volume = i_volume[np.newaxis, :]
    coords = create_zero_centered_coordinate_mesh(shape)
    #print("initial coords:", coords)
    coords = rotate_coords_3d(coords, angle_x=angle[0] * p, angle_y=angle[1] * p, angle_z=angle[2] * p, is_reverse=False)
    #print("forward rotate coords", coords)
    coords = shift_center(coords, shape)
    new_volume = batch_deform_3d(i_volume, coords, bspline_order=0, border_mode="nearest",
                                    constant_val=0)
    return new_volume[0,:]

def random_rotate_flip(img_array, mask_array):
    k = np.random.randint(0, 4)
    image = np.rot90(img_array, k)
    label = np.rot90(mask_array, k)
    axis = np.random.randint(0, 2)
    img_array = np.flip(image, axis=axis).copy()
    mask_array = np.flip(label, axis=axis).copy()
    return img_array, mask_array

def random_cutout(mask_array, num=1, size=(5,5,5),):
    patch_size = mask_array.shape
    a,b,c = np.where(mask_array>0)
    if len(a) < 125:
        return mask_array
    for i in range(num):
        idx = random.randint(1, len(a)-1)
        x,y,z = a[idx], b[idx], c[idx]
        mask_array[x:min(x+size[0], patch_size[0]),y:min(y+size[1],patch_size[1]), z:min(z+size[2],patch_size[2])] = 0
    return mask_array

if __name__ == "__main__":
    is_translate = False
    is_rotate = False
    is_affine = False
    is_volume = True
    if is_translate:
        i_volume = torch.zeros((1, 1, 50, 50, 50))
        i_volume[0, 0, 10:20, 10:20, 20:30] = 1
        shape = (50, 50, 50)
        coords = create_zero_centered_coordinate_mesh(shape)
        print("initial coords:", coords)
        coords = translate_coords_3d(coords, t_x=5.3, t_y=6.2, t_z=3.1)
        print("forward translate coords", coords)
        coords = shift_center(coords, shape)
        print("forward shifted coords", coords)
        recovered_coords = create_zero_centered_coordinate_mesh(shape)
        recovered_coords = translate_coords_3d(recovered_coords, t_x=-5.3, t_y=-6.2, t_z=-3.1)
        print("backward translate coords", recovered_coords)
        recovered_coords = shift_center(recovered_coords, shape)
        print("backward shifted coords", recovered_coords)
        new_volume = batch_deform_3d(i_volume.detach().numpy(), coords, bspline_order=0, border_mode="nearest",
                                     constant_val=0)
        recovered_volume = batch_deform_3d(new_volume, recovered_coords, bspline_order=0, border_mode="nearest",
                                           constant_val=0)
        save(i_volume.squeeze().detach().numpy(), filename="./in.nii.gz")
        save(new_volume.squeeze(), filename="./out.nii.gz")
        save(recovered_volume.squeeze(), filename="./rec.nii.gz")
        print("origin volume size:", i_volume.sum().item())
        print("forward volume size:", new_volume.sum())
        print("backward volume size:", recovered_volume.sum())
    if is_rotate:
        p = np.pi
        i_volume = torch.zeros((1, 1, 201, 201, 201))
        i_volume[0, 0, 40:161, 40:161, 40:161] = 1
        shape = (201, 201, 201)
        coords = create_zero_centered_coordinate_mesh(shape)
        print("initial coords:", coords)
        coords = rotate_coords_3d(coords, angle_x=0.25 * p, angle_y=0.33 * p, angle_z=0.66 * p, is_reverse=False)
        print("forward rotate coords", coords)
        coords = shift_center(coords, shape)
        print("forward shifted coords", coords)
        recovered_coords = create_zero_centered_coordinate_mesh(shape)
        recovered_coords = rotate_coords_3d(recovered_coords, angle_x=-0.25 * p, angle_y=-0.33 * p,
                                            angle_z=-0.66 * p,
                                            is_reverse=True)
        print("backward rotate coords", recovered_coords)
        recovered_coords = shift_center(recovered_coords, shape)
        print("backward shifted coords", recovered_coords)
        new_volume = batch_deform_3d(i_volume.detach().numpy(), coords, bspline_order=0, border_mode="nearest",
                                     constant_val=0)
        recovered_volume = batch_deform_3d(new_volume, recovered_coords, bspline_order=0, border_mode="nearest",
                                           constant_val=0)
        save(i_volume.squeeze().detach().numpy(), filename="./in.nii.gz")
        save(new_volume.squeeze(), filename="./out.nii.gz")
        save(recovered_volume.squeeze(), filename="./rec.nii.gz")
        print("origin volume size:", i_volume.sum().item())
        print("forward volume size:", new_volume.sum())
        print("backward volume size:", recovered_volume.sum())
    if is_affine:
        p = np.pi
        i_volume = torch.zeros((1, 1, 201, 201, 201))
        i_volume[0, 0, 60:141, 60:141, 60:141] = 1
        out_volume = affine_transformation(i_volume, radius=(0.25 * p, 0.33 * p, 0.16 * p), translate=(2.1, 0, 5.7),
                                           scale=(2, 2, 2),
                                           bspline_order=0, border_mode="nearest", constant_val=0, is_reverse=False)
        save(i_volume.squeeze().detach().numpy(), filename="./affine_in.nii.gz")
        save(out_volume.squeeze().detach().numpy(), filename="./affine_out.nii.gz")
        rec_volume = affine_transformation(out_volume, radius=(-0.25 * p, -0.33 * p, -0.16 * p),
                                           translate=(-2.1, 0, -5.7), scale=(0.5, 0.5, 0.5),
                                           bspline_order=0, border_mode="nearest", constant_val=0, is_reverse=True)
        save(rec_volume.squeeze().detach().numpy(), filename="./affine_rec.nii.gz")
        delta = rec_volume - i_volume
        print("delta voxel number:", delta.sum().item())
    if is_volume:
        p = np.pi
        i_volume, h1 = load("../pet0113/nii_files_crop/t111_1.nii.gz")
        i_mask, h2 = load("../pet0113/nii_files_crop/t111_1_seg.nii.gz")
        i_volume = i_volume.reshape(1, 1, i_volume.shape[0], i_volume.shape[1], i_volume.shape[2])
        i_mask = i_mask.reshape(1, 1, i_mask.shape[0], i_mask.shape[1], i_mask.shape[2])
        radius = np.array((0.05 * p, 0.08 * p, 0.05 * p))
        translate = np.array((0., 0., 0.))
        scale = np.array((0.9, 0.9, 0.9))
        out_volume = affine_transformation(i_volume, radius=radius, translate=translate, scale=scale,
                                           bspline_order=3,
                                           border_mode="constant", constant_val=0, is_reverse=False)
        out_mask = affine_transformation(i_mask, radius=radius, translate=translate, scale=scale, bspline_order=0,
                                         border_mode="nearest", constant_val=0, is_reverse=False)
        rec_volume = affine_transformation(out_volume, radius=-radius, translate=-translate, scale=scale,
                                           bspline_order=3, border_mode="constant", constant_val=0, is_reverse=True)
        rec_mask = affine_transformation(out_mask, radius=-radius, translate=-translate, scale=scale,
                                         bspline_order=0,
                                         border_mode="nearest", constant_val=0, is_reverse=True)
        delta_v = np.abs(i_volume - rec_volume)
        delta_m = np.abs(i_mask - rec_mask)
        print(delta_v.sum(), delta_m.sum())
        out_volume = (out_volume - out_volume.max()).squeeze().astype(np.float)
        rec_volume = (rec_volume - rec_volume.max()).squeeze().astype(np.float)
        out_mask = out_mask.squeeze().astype(np.uint8)
        rec_mask = rec_mask.squeeze().astype(np.uint8)
        delta_v = delta_v.squeeze().astype(np.uint16)
        delta_m = delta_m.squeeze().astype(np.uint8)
        save(out_volume, filename="./affine_result/out_vol.nii.gz", hdr=h1)
        save(rec_volume, filename="./affine_result/rec_vol.nii.gz", hdr=h1)
        save(out_mask, filename="./affine_result/out_mask.nii.gz", hdr=h2)
        save(rec_mask, filename="./affine_result/rec_mask.nii.gz", hdr=h2)
        save(delta_v, filename="./affine_result/delta_v.nii.gz", hdr=h1)
        save(delta_m, filename="./affine_result/delta_m.nii.gz", hdr=h2)
        save(i_volume.squeeze(), filename="./affine_result/in_vol.nii.gz", hdr=h1)
        save(i_mask.squeeze(), filename="./affine_result/in_msk.nii.gz", hdr=h2)
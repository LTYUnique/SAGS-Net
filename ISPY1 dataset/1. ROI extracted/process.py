#!/home/wanghong428/anaconda3/bin/python

import os
import natsort
import numpy as np
import SimpleITK as sitk

def find_edge(label, axis):
    s = np.sum(label, axis=axis)  # axis: 1-行, 0-列
    edge1 = 0
    for i in s:
        if i == 0:
            edge1 += 1
        else:
            break
    edge2 = len(s) - 1
    for i in reversed(s):
        if i == 0:
            edge2 -= 1
        else:
            break
    return edge1, edge2


def find_edges(label):
    up, down = find_edge(label, 1)
    left, right = find_edge(label, 0)
    return up, down, left, right



def find_window_location(label3D):
    alist = []
    for slice_index in range(len(label3D)):
        up, down, left, right = find_edges(label3D[slice_index])
        if up >= down or left >= right:
            continue

        alist.append([up, down, left, right])

    # 找到长宽最大值及其索引
    a = np.array(alist)
    up_max, down_max, left_max, right_max = np.min(a[:, 0]), np.max(a[:, 1]), np.min(a[:, 2]), np.max(a[:, 3])
    # print('window:', (down_max - up_max + 1), ',', (right_max - left_max + 1), '(', up_max, down_max, left_max,
    #       right_max, ')')
    return up_max, down_max, left_max, right_max


def extract(image, up, down, left, right, margin):
    up = (up - margin) if up > margin else 0
    down = (down + margin) if (down + margin) < image.shape[0] else image.shape[0] - 1
    left = (left - margin) if left > margin else 0
    right = (right + margin) if (right + margin) < image.shape[1] else image.shape[1] - 1

    width = right - left + 1
    height = down - up + 1
    # print(height, width)
    new_image = np.empty((height, width))
    # new_image = image[up:down+1][left:right+1]
    for new_row, o_row in enumerate(range(up, down + 1)):
        for new_col, o_col in enumerate(range(left, right + 1)):
            new_image[new_row][new_col] = image[o_row][o_col]
    return new_image



def find_tumor_side(label3D):
    s = np.sum(label3D, axis=(1, 2))
    edge1 = 0
    for i in s:
        if i == 0:
            edge1 += 1
        else:
            break
    edge2 = len(s) - 1
    for i in reversed(s):
        if i == 0:
            edge2 -= 1
        else:
            break
    return edge1, edge2  # 都包含


def extract_and_save(image3D, up, down, left, right, margin, begin, end, save_path_name):
    alist = []
    for slice_index in range(begin, end+1):
        new_image = extract(image3D[slice_index], up, down, left, right, margin)
        alist.append(new_image)

    sitk.WriteImage(sitk.GetImageFromArray(np.array(alist)), save_path_name)


def division(image3D_path, label3D_path, save_path, margin):
    """
    image3D_path, label3D_path : 要分割的图
    根据label3D_path来分割，label可含多个值，值取在label_values里面的值
    label_values：范围在label3D_refer之内，type : list
    """
    folder_list = natsort.natsorted(os.listdir(image3D_path))
    os.makedirs(save_path, exist_ok=True)
    
    d = {'0000': 'pre', '0001': 'post1','0002': 'post2'}

    for folder in folder_list:
        file_list = natsort.natsorted(os.listdir(image3D_path + '/' + folder))
        label3D = sitk.ReadImage(label3D_path + '/' + folder + '.nii.gz')
        label3D = sitk.GetArrayFromImage(label3D)
        up_max, down_max, left_max, right_max = find_window_location(label3D)
        begin, end = find_tumor_side(label3D)
        # print(folder, begin, end)

        for file3D_name in file_list:
            image3D = sitk.ReadImage(image3D_path + '/' + folder + '/' + file3D_name)
            image3D = sitk.GetArrayFromImage(image3D)
            extract_and_save(image3D, up_max, down_max, left_max, right_max, margin, begin, end,
                             save_path + '/' + folder + d[file3D_name.split('_')[3]] + '.nii')


if __name__ == '__main__':
	division(
		'images_bias-corrected_resampled_zscored_nifti',
		'masks_stv_manual',
		'processed',
		0
	)

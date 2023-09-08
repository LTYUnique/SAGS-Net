import os

import SimpleITK
import numpy as np
import pandas as pd
import pydicom


def get_dcm_data(file_name):
    names_dcm = os.listdir(file_name)
    names_dcm = sorted(names_dcm)
    imgs = []
    for n in names_dcm:
        img = pydicom.read_file(file_name + '/' + n).pixel_array
        imgs.append(img)
    imgs = np.array(imgs)
    return imgs


if __name__ == '__main__':
    path_box = 'Duke-Breast-Cancer-MRI/Annotation_Boxes.xlsx'

    bboxs = pd.read_excel(path_box)
    names = np.array(bboxs)[:, 0]
    xzy = np.array(bboxs)[:, 1:]
    L = dict(zip(names, xzy))

    path = '/media/ubuntu/HDD/TCIA_dataset/MRI/pre_data_MRI'
    path_out = '/media/ubuntu/HDD/TCIA_dataset/MRI/MRI_ROI'
    patients = os.listdir(path)
    patients.sort(key=lambda x: int(x.split('_')[-1]))
    for file in patients:
        if os.path.exists(path_out + '/' + file.split('_')[-1] + '.nii'):
            continue
        print(file)
        data = get_dcm_data(path + '/' + file + '/DCM')

        s = L[file]
        x = [s[0], s[1]]
        y = [s[2], s[3]]
        z = [s[4], s[5]]

        print(data.shape)
        SimpleITK.WriteImage(SimpleITK.GetImageFromArray(data[z[0]: z[1] + 1, x[0]: x[1] + 1, y[0]: y[1] + 1]),
                              path_out + '/' + file.split('_')[-1] + '.nii')
    #     # break

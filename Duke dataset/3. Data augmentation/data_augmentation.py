
import os
import numpy as np
import SimpleITK as sitk
import pandas as pd
import cv2 as cv
import copy


def data_augmentation(img, tag):
    # tag: 1:水平翻转, 2:垂直翻转 , 3-9:旋转 per 45°, 10-16:水平+旋转, 0: img itself
    if tag == 0:
        return img

    size = img.shape  # 获得图像的形状
    h = size[0]
    w = size[1]

    if tag == 1:  # 水平翻转
        iLR = copy.deepcopy(img)  # 获得一个和原始图像相同的图像，注意这里要使用深度复制
        for i in range(h):  # 元素循环
            for j in range(w):
                iLR[i, w - 1 - j] = img[i, j]
        return iLR
    if tag == 2:  # 垂直翻转
        jLR = copy.deepcopy(img)  # 获得一个和原始图像相同的图像，注意这里要使用深度复制
        for i in range(w):  # 元素循环
            for j in range(h):
                jLR[h - 1 - j, i] = img[j, i]
        return jLR
    if tag <= 9 and tag >= 3:  # 旋转
        angle = (tag - 2) * 45
        # 抓取旋转矩阵(应用角度的负值顺时针旋转)。参数1为旋转中心点;参数2为旋转角度,正的值表示逆时针旋转;参数3为各向同性的比例因子
        M = cv.getRotationMatrix2D((w / 2, h / 2), -angle, 1.0)
        newW = int((h * np.abs(M[0, 1])) + (w * np.abs(M[0, 0])))
        newH = int((h * np.abs(M[0, 0])) + (w * np.abs(M[0, 1])))
        # 调整旋转矩阵以考虑平移
        M[0, 2] += (newW - w) / 2
        M[1, 2] += (newH - h) / 2
        # 执行实际的旋转并返回图像
        return cv.warpAffine(img, M, (newW, newH))  # borderValue 缺省，默认是黑色
    if tag <= 16 and tag >= 10:  # 水平+旋转
        iLR = copy.deepcopy(img)  # 获得一个和原始图像相同的图像，注意这里要使用深度复制
        for i in range(h):  # 元素循环
            for j in range(w):
                iLR[i, w - 1 - j] = img[i, j]

        angle = (tag - 9) * 45
        # 抓取旋转矩阵(应用角度的负值顺时针旋转)。参数1为旋转中心点;参数2为旋转角度,正的值表示逆时针旋转;参数3为各向同性的比例因子
        M = cv.getRotationMatrix2D((w / 2, h / 2), -angle, 1.0)
        newW = int((h * np.abs(M[0, 1])) + (w * np.abs(M[0, 0])))
        newH = int((h * np.abs(M[0, 0])) + (w * np.abs(M[0, 1])))
        # 调整旋转矩阵以考虑平移
        M[0, 2] += (newW - w) / 2
        M[1, 2] += (newH - h) / 2
        # 执行实际的旋转并返回图像
        return cv.warpAffine(iLR, M, (newW, newH))  # borderValue 缺省，默认是黑色
    print('error in data_augmentation')
    return None  # error


def data_aug_3D(img_3D, tag):  # image_3D: np.array
    new_image = []
    for j in range(img_3D.shape[0]):
        new_image.append(data_augmentation(img_3D[j], tag))
    new_image = np.array(new_image)
    return new_image


tags_value = {  # 0: '',  # 原始图像
    1: 'a',  # 水平翻转
    2: 'b',  # 垂直翻转
    3: 'c',  # 旋转45°
    4: 'd',  # 旋转90°
    5: 'e',  # 旋转135°
    6: 'f',  # 旋转180°
    7: 'g',  # 旋转225°
    8: 'h',  # 旋转270°
    9: 'i',  # 旋转315°
    10: 'j',  # 水平翻转+旋转45°
    11: 'k',  # 水平翻转+旋转90°
    12: 'l',  # 水平翻转+旋转135°
    13: 'm',  # 水平翻转+旋转180°
    14: 'n',  # 水平翻转+旋转225°
    15: 'o',  # 水平翻转+旋转270°
    16: 'p'  # 水平翻转+旋转315°
}

if __name__ == '__main__':
    # 从excel获取文件列表
    data_list = pd.read_excel("label pCR (tumor response).xlsx")
    # names = np.array(data_list)[0:288, 0]
    names = np.array(data_list)[:, 0]
    names = [names[i].split('_')[-1] for i in range(len(names))]

    cols = []
    # 读取文件
    file_path = 'Neoadjuvant Therapy or not(all patients 286 post 2 ROI)'
    for name in names:

        nii_image = sitk.ReadImage(os.path.join(file_path, name) + '.nii')
        nii_image = sitk.GetArrayFromImage(nii_image)
        print(name, nii_image.shape)
        # 数据增广
        # tag: 1:水平翻转, 2:垂直翻转 , 3-9:旋转 90 180 270, 10-16:水平+旋转, 0: img itself
        rows = []
        for tag in tags_value:
            img_aug = data_aug_3D(nii_image, tag)

            # 存储图像
            save_path = 'post2/aug' + tags_value[tag]
            os.makedirs(save_path, exist_ok=True)
            sitk.WriteImage(sitk.GetImageFromArray(img_aug),
                            save_path + '/'  + name + 'post2'+ tags_value[tag] + '.nii')
            rows.append('Breast_MRI_' + name + 'post2' + tags_value[tag])
        cols.append(rows)

    # 存储excel
    cols = np.array(cols)
    for index, tag in enumerate(tags_value):
        data_list[tags_value[tag]] = cols[:, index]
    writer = pd.ExcelWriter("post2.xlsx", engine="xlsxwriter")
    data_list.to_excel(writer, index=False)
    writer.save()
    writer.close()

import os
import SimpleITK
import cv2
import numpy as np
from keras.utils.all_utils import to_categorical
import pandas as pd

np.random.seed(0)


def normalize(image):
    image -= np.min(image)
    image = image / np.max(image)
    return image


def resize(image):
    image = image.transpose(1, 2, 0)
    image = cv2.resize(image, (64, 64)).transpose(2, 0, 1)
    image = cv2.resize(image, (64, 64))
    return image


def DataSet(k_folds, k, training=True, classes=2):
    if k == k_folds:
        val_k = 0
    else:
        val_k = k

    # path_data = 'pCR to NAC_new/pre_post1_post2_858'
    # path_label = 'pCR to NAC_new/label_pre_post1_post2_858.xlsx'
    path_data = '1. Breast Cancer Datasets/3. ISPY1/NIfTI-Files/processed'
    path_label = '1. Breast Cancer Datasets/3. ISPY1/NIfTI-Files/extracted_label_2.xlsx'

    features = pd.read_excel(path_label)
    e = np.array(features)[0:, 1]
    Id = np.array(features)[0:, 0]
    # Id = [Id[i].split('_')[-1] for i in range(len(Id))] #ISPY1 dataset need to common
    list_id_label = dict(zip(Id, e))
    # print(list_id_label)
    files = np.array(os.listdir(path_data))
    files_k_folds = np.array_split(files, k_folds)

    test_data = np.array([normalize(resize(SimpleITK.GetArrayFromImage
                                           (SimpleITK.ReadImage(path_data + '/' + files_k_folds[k - 1][i]))))
                          for i in range(len(files_k_folds[k - 1]))])
    test_data = np.concatenate([test_data[i] for i in range(test_data.shape[0])], axis=0)
    test_label = np.array([list_id_label[files_k_folds[k - 1][i].split('.')[0]]
                           for i in range(len(files_k_folds[k - 1])) for _ in range(64)])
    test_label = to_categorical(test_label, classes)

    val_data = np.array([normalize(resize(SimpleITK.GetArrayFromImage
                                          (SimpleITK.ReadImage(path_data + '/' + files_k_folds[val_k][i]))))
                         for i in range(len(files_k_folds[val_k]))])
    val_data = np.concatenate([val_data[i] for i in range(val_data.shape[0])], axis=0)
    val_label = np.array([list_id_label[files_k_folds[val_k][i].split('.')[0]]
                          for i in range(len(files_k_folds[val_k])) for _ in range(64)])
    val_label = to_categorical(val_label, classes)

    train_files = np.concatenate([files_k_folds[i] for i in range(k_folds) if (i != val_k) and (i != k - 1)])

    train_data = np.array([normalize(resize(SimpleITK.GetArrayFromImage
                                            (SimpleITK.ReadImage(path_data + '/' + train_files[i]))))
                           for i in range(len(train_files))])
    train_data = np.concatenate([train_data[i] for i in range(train_data.shape[0])], axis=0)
    train_label = np.array([list_id_label[train_files[i].split('.')[0]]
                            for i in range(len(train_files)) for _ in range(64)])
    train_label = to_categorical(train_label, classes)

    if training:
        return [train_data[..., None], train_label, train_files], [val_data[..., None], val_label, files_k_folds[val_k]]
    else:
        return [test_data[..., None], test_label, files_k_folds[k - 1]]


if __name__ == '__main__':
    x, y = DataSet(10, 1)
    # print(y[0].shape)

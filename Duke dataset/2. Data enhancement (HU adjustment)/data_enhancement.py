
import os
import natsort
import SimpleITK as sitk
import pandas as pd


def get_len(array):  # 获取数组中元素数量
    total = 1
    for i in array.shape:
        total *= i
    return total


def data_enhancement(img, drop=0.01):  # img : Multidimensional is OK
    s = pd.value_counts(pd.DataFrame(img.reshape(-1, 1))[0])
    s = s.sort_index()  # 按index排序
    # 找出边界
    total = get_len(img)
    drop_total = total * drop
    drop_num = 0
    s_value = s.tolist()
    cursor = s_value[-1]
    index_i = 0
    index_j = len(s_value) - 1
    while drop_num <= drop_total:
        while drop_num <= drop_total and s_value[index_i] <= cursor:  # 向右
            #         print(s.index[index_i]) # 这是要扔掉的值
            drop_num += s_value[index_i]
            index_i += 1
        cursor = s_value[index_i]
        while drop_num <= drop_total and s_value[index_j] <= cursor:  # 向左
            #         print(s.index[index_j]) # 这是要扔掉的值
            drop_num += s_value[index_j]
            index_j -= 1
    # 处理图片
    img[img > s.index[index_j]] = s.index[index_j]
    img[img < s.index[index_i]] = s.index[index_i]
    return img - s.index[index_i]


def data_enhancements(image_path, save_path, drop):
    file_list = natsort.natsorted(os.listdir(image_path))
    os.makedirs(save_path, exist_ok=True)

    for file_name in file_list:
        # 1.read nii
        image = sitk.ReadImage(image_path + '/' + file_name)
        image = sitk.GetArrayFromImage(image)
        # 2.data enhancement
        image = data_enhancement(image, drop)
        # 3.save
        sitk.WriteImage(sitk.GetImageFromArray(image), save_path + '/' + file_name)


if __name__ == '__main__':
    data_enhancements('post1/auga', 'save1', 0.01)
    data_enhancements('post2/auga', 'save2', 0.01)
    data_enhancements('pre/auga',    'save3', 0.01)

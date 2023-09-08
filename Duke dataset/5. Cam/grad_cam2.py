
import os

from tensorflow.keras.models import load_model
import SimpleITK as sitk
import skimage.transform as trans

import math
import numpy as np
import tensorflow as tf
import cv2

# Display
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from global_guide_attention import global_guide_attention

def norm(img):
	min, max = np.min(img), np.max(img)
	img = (img-min)/(max-min) if (max-min > 0) else img
	return img


def resize(image, s):
    image = image.transpose(1, 2, 0)
    image = cv2.resize(image, s).transpose(2, 0, 1)
    image = cv2.resize(image, s)
    return image


def load_image(path, name, s):
    # read
    image = sitk.ReadImage(path + '/' + name)
    image = sitk.GetArrayFromImage(
        image)  # shape: any shape as long as it's 2D, but length and width must be the same; value:0~any
    
    #------- this is just for 3D
    #resize
    image = resize(image, tuple(s[1:3]))  # shape: 3D, x*y*z
    
    # choose one
    image = image[19]  # todo  shape:2D x*y
    
    sitk.WriteImage(sitk.GetImageFromArray(image), 'see.nii')
    
    # reshape
    image = image.reshape(s)
    print(image.shape)
    
    # norm it
    return norm(image)
    
    
    #todo: #return image.reshape(s) / np.max(image)  # todo!!!
    


def make_gradcam_heatmap_division(img_array, model, layer_name):
    '''分割模型'''
    ## 以输出为[1, 128, 128, 16]的layer为例
    # First, we create a model that maps the input image 
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(layer_name).output, model.output]
    )

    # Then, we compute the gradient of the top predicted class for our input image
    # with respect to the activations of the last conv layer
    # 然后，我们为输入图像计算top预测类关于最后一个conv层的激活的梯度
    with tf.GradientTape() as tape:
        layer_output, probability = grad_model(img_array)
        # layer_output: 目标层的输出, example[1, 128, 128, 16]； probability：最终输出, example[1, 128, 128, 1]

    grads = tape.gradient(probability, layer_output)  # [1, 128, 128, 16], same as layer_output's shape

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))  # [16]， layer_output 的最后一维

    layer_output = layer_output[0]  # [128,128,16]
    # heatmap (128, 128, 1) = (128, 128, 16)  @(16,)相当于(128, 128, 16)乘以(16,1)
    heatmap = layer_output @ pooled_grads[..., tf.newaxis]
    # tf.squeeze 去除1的维度,(128,128)
    heatmap = tf.squeeze(heatmap)

    # For visualization purpose, we will also normalize the heatmap between 0 & 1	
    # tf.maximum(heatmap, 0) 和0比较大小,返回一个>=0的值,相当于relu,然后除以heatmap中最大的 值,进行normalize归一化到0-1
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()


def make_gradcam_heatmap_classification(img_array, model, layer_name, pred_index=None):
    '''分类模型'''
    ## 以输出为[1, 128, 128, 16]的layer为例
    # First, we create a model that maps the input image
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(layer_name).output, model.output]
    )

    # Then, we compute the gradient of the top predicted class for our input image
    # with respect to the activations of the last conv layer
    # 然后，我们为输入图像计算top预测类关于最后一个conv层的激活的梯度
    with tf.GradientTape() as tape:
        layer_output, preds = grad_model(img_array)
        # 如果没有传入pred_index,就计算pred[0]中最大的值对应的下标号index
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    grads = tape.gradient(class_channel, layer_output)

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))  # [16]， layer_output 的最后一维

    layer_output = layer_output[0]  # [128,128,16]
    # heatmap (128, 128, 1) = (128, 128, 16)  @(16,)相当于(128, 128, 16)乘以(16,1)
    heatmap = layer_output @ pooled_grads[..., tf.newaxis]
    # tf.squeeze 去除1的维度,(128,128)
    heatmap = tf.squeeze(heatmap)

    # For visualization purpose, we will also normalize the heatmap between 0 & 1
    # tf.maximum(heatmap, 0) 和0比较大小,返回一个>=0的值,相当于relu,然后除以heatmap中最大的 值,进行normalize归一化到0-1
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()


def get_gradcam(img, heatmap, alpha=0.2):
    img = tf.broadcast_to(img, (max(img.shape), max(img.shape), 3))  # 0~255

    # Rescale heatmap to a range 0-255
    heatmap = np.uint8(255 * heatmap)

    # Use jet colormap to colorize heatmap
    jet = cm.get_cmap("jet")

    # Use RGB values of the colormap
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]  # value: 0~1

    # Create an image with RGB colorized heatmap
    jet_heatmap = trans.resize(jet_heatmap, (max(img.shape), max(img.shape), 3))

    # Superimpose重叠 the heatmap on original image
    superimposed_img = jet_heatmap * alpha + img  # superimposed_img: TensorShape([128, 128, 3])
    #     superimposed_img = superimposed_img.numpy().astype(np.int32)
    #     plt.imshow(superimposed_img)
    return superimposed_img


def save_cam(file_name, cam_imgs, save_path, figsize=(15,15), col=5, show=False):  # figsize=(15, 15)
    '''

    :param cam_imgs: include [layer_name, img_itself]*N
    :param save_path:
    :param figsize: 子图大小
    :param col:
    :param show: 是否显示图片
    :return:
    '''
    if col is None:
        col = int(math.sqrt(len(cam_imgs)))
        print('col:', col)
    
    os.makedirs(save_path, exist_ok=True)
    row = math.ceil(len(cam_imgs) / col)

    fig1=plt.figure(figsize=figsize)
    one_row, one_col = 1/row, 1/col
    title_h = 0.6/max((list(figsize)))
    img_size = min(one_row-title_h, one_col)
    offset_x = (one_col -img_size)/2
    for index, cam_img in enumerate(cam_imgs):
        row_ = index//col
        col_ = index%col
        #cprint(row_,col_)
        ax1 = plt.Axes(fig1,[one_col*col_+offset_x, 1-one_row*row_-one_row, img_size, img_size])
        ax1.imshow(cam_img[1])
        ax1.set_axis_off()
        ax1.set_title(str(index + 1) + '-' + cam_img[0])
        fig1.add_axes(ax1)
     

    plt.xticks([])
    plt.yticks([])
    plt.savefig(save_path + '/' + file_name + '.jpg')
    if show:
        plt.show()


def gradcam_division(path, file_names, model, where_model, save_path, display_col=5, alpha=0.2, figsize=(12,12),  save_origin=True):
    '''
    image type: .nii文件，如果不是nii/dcm这种类型，需改变文件读取方式

    image shape: 2维即可（shape=128*128 or 128*128*1 or 1*128*128 or ***都可以），长宽必须相等

    :param path: image.nii所在位置
    :param file_names: path下的文件名。可以含多个文件名，type: str or list(str)
    :param model: 模型本身
    :param where_model: 模型(.hdf5)所在位置，type:str, include model path and model name
                        model 和 where_model 至少要有一个不为None，若其中一个有值，另一个可为None。
    （函数要么传入model，要么传入model所在路径，但不能传入model的权重）
    :param save_path: 可视化结果(.jpg图片)保存的位置
    :param display_col: 在保存的.jpg图片中，一行显示几个子图
    :param alpha: 热力图的透明度
    :return:
    '''
    if isinstance(file_names, str):  # if not list type, change it
        file_names = [file_names]

    for name in file_names:  # for every image
       
        # get model
        if model is None:
            model = load_model(where_model)
        
        # get shape
        s = list(model.input.shape)
        s[0] = 1
        print(s)
        
        # get image
        img_array = load_image(path, name, s)  # 以128*128为例
        
        # get layer generator
        layer_name_g = (layer.name for layer in model.layers)
        next(layer_name_g)  # first layer will raise error

        cam_imgs = []
        for layer_name in layer_name_g:  # for every layer
            try:
                # Generate heatmap
                heatmap = make_gradcam_heatmap_division(img_array, model, layer_name)
                superimposed_img = get_gradcam(img_array[0], heatmap, alpha=alpha)
                cam_imgs.append([layer_name, superimposed_img])
            except Exception as e:
                print(layer_name, "except:", e)

        if save_origin:
            cam_imgs.append(['image', tf.broadcast_to(img_array[0], (max(img_array.shape), max(img_array.shape), 3))])

        if len(cam_imgs) > 0:
            save_cam(name, cam_imgs, save_path, col=display_col, figsize=figsize)


def gradcam_classification(path, file_names, model, where_model, save_path,
                           display_col=5, alpha=0.2, figsize=(15,15), save_origin=False):
    if isinstance(file_names, str):  # if not list type, change it
        file_names = [file_names]
        

    for name in file_names:  # for every image
        # get image
        # img_array = load_image(path, name)  # 以128*128为例
        # get model
        if model is None:
            model = load_model(where_model)
            
        # get shape
        s = list(model.input.shape)
        s[0] = 1
        print(s)
        
        # get image
        img_array = load_image(path, name, s)  # 以128*128为例
        
        # get layer generator
        layer_name_g = (layer.name for layer in model.layers)
        next(layer_name_g)  # first layer will raise error

        cam_imgs = []
        for layer_name in layer_name_g:  # for every layer
            try:
                #print(layer_name)
                # Generate heatmap
                heatmap = make_gradcam_heatmap_classification(img_array, model, layer_name)
                superimposed_img = get_gradcam(img_array[0], heatmap, alpha=alpha)
                cam_imgs.append([layer_name, superimposed_img])
            except Exception as e:
                print(layer_name, "except:", e)
        
        if save_origin:
            cam_imgs.append(['image', tf.broadcast_to(img_array[0], (max(img_array.shape), max(img_array.shape), 3))])

        if len(cam_imgs) > 0:
            save_cam(name, cam_imgs, save_path, col=display_col, figsize=figsize)


if __name__ == '__main__':
    # # 分割 demo
    # path = 'image'
    # file_names = ['60.nii.gz_s113_aug0.nii', '61.nii.gz_s130_aug0.nii']
    # where_model = 'model_save/UNet_all_HU_r.hdf5'
    # save_path = 'cam_img'
    # gradcam_division(path, file_names, where_model, save_path)

    # 分类 demo
    '''
    path = 'image'
    file_names = ['10.nii']
    where_model = 'model_save/41-0.083885-0.940407.hdf5'
    save_path = 'cam_img'
    gradcam_classification(path, file_names, model, None, save_path, alpha=0.4, figsize=(15, 15))
    '''
    
    path = 'lty_image_processing/grad_cam/image'
    file_names = ['ISPY1_1037post2.nii']
    save_path = 'cam_img'
    
    model = global_guide_attention()
    model.load_weights('model_save/ 32-0.109487-0.947141.hdf5')

    gradcam_classification(path, file_names, model, None, save_path, alpha=0.8, display_col=None, 
                           figsize=(30,30), save_origin=True)

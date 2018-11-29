# -*- coding:utf-8 -*-
"""
@author:TanQingBo
@file:showimage.py
@time:2018/11/1310:25
"""
import os
import numpy as np
import matplotlib.image as mpimg # mpimg 用于读取图片
import matplotlib.pyplot as plt

test_img_path = "E:/liverdata/nii/png/img/img0/"
test_gt_path = "E:/liverdata/nii/png/label/label0/"
test_pre_path="E:/liverdata/nii/png/label/label0/"

valid_img_path = "E:/liverdata/nii/png/img/img0/"
valid_gt_path = "E:/liverdata/nii/png/label/label0/"
valid_pre_path="E:/liverdata/nii/png/label/label0/"

test_img_list = sorted(os.listdir(test_img_path))
test_gt_list = sorted(os.listdir(test_gt_path))
test_pre_list = sorted(os.listdir(test_pre_path))
valid_img_list = sorted(os.listdir(valid_img_path))
valid_gt_list = sorted(os.listdir(valid_gt_path))
valid_pre_list = sorted(os.listdir(valid_pre_path))

test_img_len = len(test_img_list)

for i in range(test_img_len):
    test_img = mpimg.imread(os.path.join(test_img_path, test_img_list[i]))
    test_gt = mpimg.imread(os.path.join(test_gt_path, test_gt_list[i]))
    test_pre = mpimg.imread(os.path.join(test_pre_path, test_pre_list[i]))

    valid_img = mpimg.imread(os.path.join(valid_img_path, valid_img_list[i]))
    valid_gt = mpimg.imread(os.path.join(valid_gt_path, valid_gt_list[i]))
    valid_pre = mpimg.imread(os.path.join(valid_pre_path, valid_pre_list[i]))

    plt.figure(figsize=(10,5)) #设置窗口大小
    plt.suptitle('Multi_Image') # 图片名称
    plt.subplot(2,3,1), plt.title('test_img-'+test_img_list[i])
    plt.imshow(test_img,cmap='gray'), plt.axis('off')
    plt.subplot(2,3,2), plt.title('test_gt-'+test_gt_list[i])
    plt.imshow(test_gt,cmap='gray'), plt.axis('off') #这里显示灰度图要加cmap
    plt.subplot(2,3,3), plt.title('test_pre-'+test_pre_list[i])
    plt.imshow(test_pre,cmap='gray'), plt.axis('off')
    plt.subplot(2,3,4), plt.title('valid_img-'+valid_img_list[i])
    plt.imshow(valid_img,cmap='gray'), plt.axis('off')
    plt.subplot(2,3,5), plt.title('valid_gt-'+valid_gt_list[i])
    plt.imshow(valid_gt,cmap='gray'), plt.axis('off')
    plt.subplot(2,3,6), plt.title('valid_pre-'+valid_pre_list[i])
    plt.imshow(valid_pre,cmap='gray'), plt.axis('off')

    plt.show()

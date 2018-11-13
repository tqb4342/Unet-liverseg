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

# img = Image.open(os.path.join('images', '2007_000648' + '.jpg'))
# gray = img.convert('L')
# r,g,b = img.split()
# img_merged = Image.merge('RGB', (r, g, b))

test_img_path = "E:/liverdata/nii/png/img/img0/liver-orig1-89.png"
test_gt_path = "E:/liverdata/nii/png/label/label0/liver-seg1-89.png"
test_pre_path="E:/liverdata/nii/png/label/label0/liver-seg1-89.png"

valid_img_path = "E:/liverdata/nii/png/img/img0/liver-orig1-89.png"
valid_gt_path = "E:/liverdata/nii/png/label/label0/liver-seg1-89.png"
valid_pre_path="E:/liverdata/nii/png/label/label0/liver-seg1-89.png"

test_img = mpimg.imread(test_img_path)
test_gt = mpimg.imread(test_gt_path)
test_pre = mpimg.imread(test_pre_path)

valid_img = mpimg.imread(valid_img_path)
valid_gt = mpimg.imread(valid_gt_path)
valid_pre = mpimg.imread(valid_pre_path)


plt.figure(figsize=(10,5)) #设置窗口大小
plt.suptitle('Multi_Image') # 图片名称
plt.subplot(2,3,1), plt.title('test_img')
plt.imshow(test_img,cmap='gray'), plt.axis('off')
plt.subplot(2,3,2), plt.title('test_gt')
plt.imshow(test_gt,cmap='gray'), plt.axis('off') #这里显示灰度图要加cmap
plt.subplot(2,3,3), plt.title('test_pre')
plt.imshow(test_pre,cmap='gray'), plt.axis('off')
plt.subplot(2,3,4), plt.title('valid_img')
plt.imshow(valid_img,cmap='gray'), plt.axis('off')
plt.subplot(2,3,5), plt.title('valid_gt')
plt.imshow(valid_gt,cmap='gray'), plt.axis('off')
plt.subplot(2,3,6), plt.title('valid_pre')
plt.imshow(valid_pre,cmap='gray'), plt.axis('off')

plt.show()

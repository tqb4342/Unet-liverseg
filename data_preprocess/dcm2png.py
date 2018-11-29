# -*- coding:utf-8 -*-
"""
@author:TanQingBo
@file:dcm2png.py
@time:2018/10/259:52
"""

import matplotlib.pyplot as plt
from skimage import exposure, img_as_float
import os
import pydicom

def dicom_2png(file):
    _currFile = file
    dcm = pydicom.dcmread(file)
    fileName = os.path.basename(file)
    imageX = dcm.pixel_array
    temp = imageX.copy()
    print("shape ----", imageX.shape)
    picMax = imageX.max()
    vmin = imageX.min()
    vmax = temp[temp < picMax].max()
    # print("vmin : ", vmin)
    # print("vmax : ", vmax)
    imageX[imageX > vmax] = 0
    imageX[imageX < vmin] = 0
    # result = exposure.is_low_contrast(imageX)
    # # print(result)
    image = img_as_float(imageX)
    plt.cla()
    plt.figure('adjust_gamma', figsize=(10.24, 10.24))
    plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)
    plt.imshow(image, 'gray')
    plt.axis('off')
    plt.savefig(fileName + '.png')
    #time.sleep(1)

if __name__ == '__main__':
    dicom_2png('1.dcm')
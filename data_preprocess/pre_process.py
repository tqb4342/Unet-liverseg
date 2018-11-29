# -*- coding:utf-8 -*-
"""
@author:TanQingBo
@file:pre_process.py
@time:2018/11/1514:15
"""
import SimpleITK as sitk
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
import collections

data_url = {
    'sliver07': 'E:/liverdata/nii/nrrd3D/CompleteData/orig/',
    #'3Dircadb': 'E:/Medical_Image_Data/3Dircadb/',os.path.join('E:/liverdata/nii/test/liver-orig', str(i)+'-nor.nrrd')
    'CT_data_batch1': 'E:/liverdata/CHAOS/CT_data_batch1/',
    #'MR_data_batch1': 'E:/Medical_Image_Data/MR_data_batch1/'

}


def norm(x, x_min=None, x_max=None, min_val=0, max_val=255):
    if not x_min and not x_max:
        x_min = np.min(x)
        x_max = np.max(x)
    return min_val + (x - x_min) / (x_max - x_min) * (max_val - min_val)

def writeimg(arr_slice,i):
    filename = os.path.join('E:/liverdata/CHAOS/nrrd/nor-orig/', 'liver-orig-nor'+str(i)+'.nrrd')
    print(filename)
    out = sitk.GetImageFromArray(arr_slice)
    sitk.WriteImage(out, filename)


def show(arr,i):
    arr_1D = np.clip(arr, -1024, 1680)
    M, m = np.max(arr_1D), np.min(arr_1D)
    arr_slice = norm(arr, m, M)
    # writeimg(arr_slice,i)
    cv2.imshow('raw', arr_slice[50] / 255)
    arr_1D = arr_1D.flatten()
    print(M, m)
    # n, bins, patches = plt.hist(arr_1D, bins=M - m)
    n, bins, patches = plt.hist(arr_slice[50].flatten(), bins=255)
    count_max = max(n)
    index = n.tolist().index(count_max)
    print('最小值：', m, ',最频繁HU数目：', count_max, ',最频繁HU：', m + index)
    plt.title(
        'min:{0},num of most frequent:{1},most frequent:{2}'.format(
            m,
            count_max,
            m +
            index))
    plt.show()
    #cv2.waitKey()

def clip_and_normalize(img, val_min=-1024, val_max=1680):
    clipped = np.clip(img, val_min, val_max)
    M, m = np.max(clipped), np.min(clipped)
    normalized = norm(clipped, m, M)
    return normalized


def read_detail(url, name):
    print(name)
    i=22
    if name == 'sliver07':
        for root, dir, file_names in os.walk(url):
            for file_name in file_names:
                if file_name.endswith('nrrd') and 'orig' in file_name:
                    print(file_name)
                    img = sitk.ReadImage(os.path.join(root, file_name))
                    # print(img.GetSpacing())
                    arr = sitk.GetArrayFromImage(img)
                    show(arr,i)
                    i=i+1

    # elif name == '3Dircadb':
    #     for data in os.listdir(url):
    #         raw_url = os.path.join(url, data + '/PATIENT_DICOM')
    #         img = sitk.ReadImage(
    #             sitk.ImageSeriesReader_GetGDCMSeriesFileNames(raw_url))
    #         # print(img.GetSpacing())
    #         arr = sitk.GetArrayFromImage(img)
    #         # arr = np.clip(arr, -1024, 1680)
    #         show(arr)

    # elif name == 'CT_data_batch1':
    #     for data in os.listdir(url):
    #         raw_url = os.path.join(url, data)
    #         img = sitk.ReadImage(
    #             sitk.ImageSeriesReader_GetGDCMSeriesFileNames(
    #                 os.path.join(
    #                     raw_url, 'DICOM_anon')))
    #         # print(img.GetSpacing())
    #         arr = sitk.GetArrayFromImage(img)
    #         # arr = np.clip(arr, -1024, 1680)
    #         show(arr,i)
    #         i=i+1
            #print(arr_slice)

    # elif name == 'MR_data_batch1':
    #     for data in os.listdir(url):
    #         raw_url = os.path.join(url, data)
    #         img = sitk.ReadImage(
    #             sitk.ImageSeriesReader_GetGDCMSeriesFileNames(
    #                 os.path.join(
    #                     raw_url, 'T1DUAL/DICOM_anon')))
    #         # print(img.GetSpacing())
    #         arr = sitk.GetArrayFromImage(img)
    #         # print(np.max(arr), np.min(arr))
    #         img = sitk.ReadImage(
    #             sitk.ImageSeriesReader_GetGDCMSeriesFileNames(
    #                 os.path.join(
    #                     raw_url, 'T2SPIR/DICOM_anon')))
    #         # print(img.GetSpacing())
    #         print('**************')
    #         arr = sitk.GetArrayFromImage(img)
    #         show(arr)


if __name__ == '__main__':
    for name, url in data_url.items():
        read_detail(url, name)
        print('----------------------------')
    pass

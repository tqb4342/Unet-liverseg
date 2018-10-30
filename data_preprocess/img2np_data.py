# This script just loads the images and saves them into NumPy binary format files .npy for faster loading later.

from __future__ import print_function

import os
import numpy as np

from skimage.io import imsave, imread

# img path
train_data_path = './oringinal_data/data'
# test_data_path = './oringinal_data/5-liver-4'

# maks path
train_mask_data_path = './oringinal_data/label'
# test_mask_data_path = './oringinal_data/5liver4'

# path and name of numpy data which stored imgs
npdata_path_for_train = './np_data/train.npy'
# npdata_path_for_test = './np_data/test.npy' 

# path and name of numpy data which stored imgs mask
npdata_path_for_train_mask = './np_data/train_mask.npy'
# npdata_path_for_test_mask = './np_data/test_mask.npy'

# img size
image_rows = 512
image_cols = 512

# save img into .npy
# img_folder_datapath: path of img data, string
# npdata_name: path and name of numpy data, string
def turn_img2npdata(img_folder_datapath, npdata_name):

	print('processing in {}',img_folder_datapath)
	# get the list of img name
	images_name_list = os.listdir(img_folder_datapath)

	# initial the np matix for saving img
	num_images = len(images_name_list)
	imgs = np.ndarray((num_images, image_rows, image_cols), dtype=np.uint8)

	count = 0
	for img_name in images_name_list:
		img_path = img_folder_datapath + '/' + img_name
		img = imread(img_path, as_gray = True)
		img = np.array([img])
		imgs[count] = img
		if count%100 == 0:
			print('Have turn {} imgs to numpydata',count)
		count += 1
	print ('All images have been turned to numpy data\n')

	np.save(npdata_name, imgs)

if __name__ == '__main__':
	turn_img2npdata(train_data_path, npdata_path_for_train)
	# turn_img2npdata(test_data_path, npdata_path_for_test)
	turn_img2npdata(train_mask_data_path, npdata_path_for_train_mask)
	# turn_img2npdata(test_mask_data_path, npdata_path_for_test_mask)
import os, shutil
import random
def partition_data_set(img_path, mask_path, test_img_path, test_mask_path, validation_img_path, validation_mask_path):
	"""generte train, validation and test folder fron original folder
	   # Arguments:
	       img_path: original img folders' path
	       mask_path: original mask folders' path
	       test_img_path: partition test img and save them in this path
	       test_mask_path, validation_img_path, validation_mask_path: like above
	"""
	if not os.path.exists(test_img_path):
		os.makedirs(test_img_path)

	if not os.path.exists(test_mask_path):
		os.makedirs(test_mask_path) 

	if not os.path.exists(validation_img_path):
		os.makedirs(validation_img_path)

	if not os.path.exists(validation_mask_path):
		os.makedirs(validation_mask_path)

	# get img and mask folder list
	imgs_folders_path = sorted(os.listdir(img_path))
	masks_folders_path = sorted(os.listdir(mask_path))
	
	# get num of every folser , put them in a list by list comprehensions
	imgs_num_list = [len(os.listdir(os.path.join(img_path,i))) for i in imgs_folders_path ]
	masks_num_list = [len(os.listdir(os.path.join(mask_path,j))) for j in masks_folders_path ]
	imgs_total_num = sum(imgs_num_list)
	print ('imgs_num_list is {}'.format(imgs_num_list))
	print ('masks_num_list is {}'.format(masks_num_list))
	print ('imgs_total_num is {}'.format(imgs_total_num))

	## Stratified sampling

	# make sure test imgs and validation imgs sample rate
	test_sample_rate = 0.1
	validation_sample_rate = 0.1

	count0 = 0
	for k in range(len(imgs_folders_path)):
		imgs_list_in_this_folder = sorted(os.listdir(os.path.join(img_path, imgs_folders_path[k])))
		masks_list_in_this_folder = sorted(os.listdir(os.path.join(mask_path, masks_folders_path[k])))
		imgs_num_in_this_folder = len(imgs_list_in_this_folder)
		sample_num = int(test_sample_rate*imgs_num_in_this_folder)
		sample_list = random_list(0, imgs_num_in_this_folder-1, 2*sample_num)
		for item in sample_list[:sample_num]:
			
			sampled_validation_img = os.path.join(img_path,imgs_folders_path[k],imgs_list_in_this_folder[item])
			sampled_validation_mask = os.path.join(mask_path,masks_folders_path[k],masks_list_in_this_folder[item])
			shutil.move(sampled_validation_img, validation_img_path)
			shutil.move(sampled_validation_mask, validation_mask_path)
			count0 += 1

		for item in sample_list[sample_num:]:
			sampled_test_img = os.path.join(img_path,imgs_folders_path[k],imgs_list_in_this_folder[item])
			sampled_test_mask = os.path.join(mask_path,masks_folders_path[k],masks_list_in_this_folder[item])
			shutil.move(sampled_test_img, test_img_path)
			shutil.move(sampled_test_mask, test_mask_path)
	print ('count0 is {}'.format(count0))

def random_list(start, stop, length):
	assert length>=0, 'length < 0'
	length = int(length)
	start, stop = (int(start), int(stop)) if start <= stop else (int(stop), int(start))
	res = []
	count = 0
	while len(res)!=length:
		random_number = random.randint(start, stop)
		if random_number not in res:
			res.append(random_number)
		count += 1
		assert count < 10000*length, 'always in circle!'
	return res

if __name__ == '__main__':
	img_path = '../original_data/img'
	mask_path = '../original_data/label'
	test_img_path = '../data/test/img0/img'
	test_mask_path = '../data/test/mask0/mask'
	validation_img_path = '../data/validation/img0/img'
	validation_mask_path = '../data/validation/mask0/mask'
	partition_data_set(img_path, mask_path, test_img_path, test_mask_path, validation_img_path, validation_mask_path)

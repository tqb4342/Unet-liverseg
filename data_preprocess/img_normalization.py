import os
import cv2

def get_filePath_fileName_fileExt(filename):
    (filepath,tempfilename) = os.path.split(filename)
    (shotname,extension) = os.path.splitext(tempfilename)
    return filepath,shotname,extension

def img_normolization(old_path, new_path):
	
	""" imgs normolization by suntract mean for grayscale imgs

		# Arguments:
			old_path: input imgs path
			new_path: output imgs path which have been subtracted mean
	"""
	# get img path list and their number
	imgs_train_list = [os.path.join(root,file) for (root, dirs, files) in os.walk(old_path) for file in files]
	total = len(imgs_train_list)

	# calculate mena val
	img_sum, count = 0, 0
	for i in imgs_train_list:
		img = cv2.imread(i,0)
		img_sum += img.sum()/float(img.shape[0]*img.shape[1])
		count += 1
		if not count%100:
			print ('have calculated {} imgs mean value'.format(count))
	average = img_sum/total
	print '-----------------------------------------------------------'
	print ('average is {}'.format(average))
	# average of nor_train_data is 54.2789521812
	print '-----------------------------------------------------------'

	# subtract mean and save them in a new path
	count = 0
	for j in imgs_train_list:
		img = cv2.imread(j,0)-average
		filepath,shotname,extension = get_filePath_fileName_fileExt(j)
		cv2.imwrite(os.path.join(new_path, shotname+'_nor'+extension), img)
		count += 1
		if not count%100:
			print ('have nomalized {} imgs '.format(count))

if __name__ == '__main__':
	old_path = '../data/img/img0' 
	new_path = '../data/nor_img/img0'
	assert os.path.exists(old_path), 'imput path is not existed'
	if not os.path.exists(new_path):
		os.makedirs(new_path)
	img_normolization(old_path, new_path)

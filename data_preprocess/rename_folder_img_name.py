import os

# folder_path: the path stored imgs to rename
# qianzui: add qianzhui to each img name
def rename_folder(folder_path, qianzhui):
	img_name_list = os.listdir(folder_path)
	for img_name in img_name_list:
		print('path before rename is {}'.format(os.path.join(folder_path,img_name)))
		print ('path after rename is {}'.format(os.path.join(folder_path,qianzhui,img_name)))
		os.rename(os.path.join(folder_path,img_name), os.path.join(folder_path,qianzhui+img_name))

if __name__ == '__main__':
	data_path = './oringinal_data_eight_folder'
	folder_name_list = sorted(os.listdir(data_path))
	print ('folder_name_list {}'.format(folder_name_list))
	count = 0
	qianzhui_list = ['4','5','6','7']
	for folder in folder_name_list:
		rename_folder(os.path.join(data_path, folder), qianzhui_list[count//2])
		count += 1
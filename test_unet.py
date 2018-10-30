from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model
from keras import backend as K
from itertools import izip
import cv2
import os

def my_list_pictures(path):
    """return the img path in a list
    # Arguments
        path: the path which is stored pintures
    # Return
        list 
    """ 
    
    pictures_list = []
    for (root, dirs, files) in os.walk(path):
        for file in files:
            pictures_list.append(os.path.join(root, file))
    return  pictures_list

def get_filePath_fileName_fileExt(filename):
    """ get filepath, shotname and extension from filename
    # Arguments:
        filename: 
    """
    (filepath,tempfilename) = os.path.split(filename)
    (shotname,extension) = os.path.splitext(tempfilename)
    return filepath,shotname,extension

#  metric function and loss function
def dice_coef(y_true, y_pred):
	y_true_f = K.flatten(y_true)
	y_pred_f = K.flatten(y_pred)
	intersection = K.sum(y_true_f * y_pred_f)
	return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_coef_loss(y_true, y_pred):
	return -dice_coef(y_true, y_pred)

# parameter for loss function
smooth = 1.

# get test data and labels
test_data_path = './data/test/img0'
test_mask_data_path = './data/test/mask0'

# folder to store predicted masks
predict_labels_path = './predict_labels'
if not os.path.exists(predict_labels_path):
    os.makedirs(predict_labels_path)

# initial test image data generater
test_gen_args = dict(rescale=1./255)

test_image_datagen = ImageDataGenerator(**test_gen_args)
test_mask_datagen = ImageDataGenerator(**test_gen_args)

seed2 = 2
test_img_generator = test_image_datagen.flow_from_directory(
    test_data_path,
    target_size=(512,512),
    color_mode='grayscale',
    class_mode=None,
    seed=seed2,
    batch_size=1,
    shuffle=False)
test_mask_generator = test_mask_datagen.flow_from_directory(
    test_mask_data_path,
    target_size=(512,512),
    color_mode='grayscale',
    class_mode=None,
    seed=seed2,
    batch_size=1,
    shuffle=False)

test_generator = izip(test_img_generator, test_mask_generator)

# load model 
weight_path = './D9805weights.h5'
model = load_model(weight_path,custom_objects={'dice_coef_loss': dice_coef_loss,'dice_coef':dice_coef})

test_mask_name_list = sorted(my_list_pictures(test_mask_data_path))
test_mask_num = len(test_mask_name_list)

# evaluate and predict it
tset_dice = model.evaluate_generator(test_generator, steps=test_mask_num, max_queue_size=10, workers=1, use_multiprocessing=False)
predict_masks = model.predict_generator(test_img_generator, steps=test_mask_num, max_queue_size=10, workers=1, use_multiprocessing=False, verbose=0)

print ('tset_dice is {}'.format(tset_dice))
print ('size of predict_masks is {}'.format(predict_masks.shape))

# save maks from np matrix to img


for i in range(test_mask_num):
    (filepath,shotname,extension) = get_filePath_fileName_fileExt(test_mask_name_list[i])
    predict_mask_path = os.path.join(predict_labels_path, 'predict_'+shotname+extension)
    predict_mask = predict_masks[i,...]*255
    # print model.predict_generator.filenames
    cv2.imwrite(predict_mask_path, predict_mask)

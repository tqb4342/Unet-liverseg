from __future__ import print_function

import os
import cv2
from skimage.transform import resize
from skimage.io import imsave
import numpy as np
from keras.models import Model
from keras.layers import Input, concatenate, Conv2D, MaxPooling2D, Conv2DTranspose
from keras.optimizers import Adam, SGD
from keras.callbacks import ModelCheckpoint
from keras import backend as K
from keras import initializers
from keras.layers.core import Dropout
from keras.layers.normalization import BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import TensorBoard
from itertools import izip
# define the input size
img_rows = 512
img_cols = 512
channels = 1

# parameter for loss function
smooth = 1.

# define u-net architecture
def UNET():
	inputs = Input((img_rows, img_cols, channels))
	conv1 = Conv2D(32, (3, 3), activation='relu', padding='same',kernel_initializer='he_normal')(inputs)
	conv1 = Conv2D(32, (3, 3), activation='relu', padding='same',kernel_initializer='he_normal')(conv1)
	pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

	conv2 = Conv2D(64, (3, 3), activation='relu', padding='same',kernel_initializer='he_normal')(pool1)
	conv2 = Conv2D(64, (3, 3), activation='relu', padding='same',kernel_initializer='he_normal')(conv2)
	pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

	conv3 = Conv2D(128, (3, 3), activation='relu', padding='same',kernel_initializer='he_normal')(pool2)
	conv3 = Conv2D(128, (3, 3), activation='relu', padding='same',kernel_initializer='he_normal')(conv3)
	pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

	conv4 = Conv2D(256, (3, 3), activation='relu', padding='same',kernel_initializer='he_normal')(pool3)
	conv4 = Conv2D(256, (3, 3), activation='relu', padding='same',kernel_initializer='he_normal')(conv4)
	conv4 = Dropout(0.5)(conv4)
	pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

	conv5 = Conv2D(512, (3, 3), activation='relu', padding='same',kernel_initializer='he_normal')(pool4)
	conv5 = Conv2D(512, (3, 3), activation='relu', padding='same',kernel_initializer='he_normal')(conv5)
	conv5 = Dropout(0.5)(conv5)

	up6 = concatenate([Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same',kernel_initializer='he_normal')(conv5), conv4], axis=3)
	conv6 = Conv2D(256, (3, 3), activation='relu', padding='same',kernel_initializer='he_normal')(up6)
	conv6 = Conv2D(256, (3, 3), activation='relu', padding='same',kernel_initializer='he_normal')(conv6)

	up7 = concatenate([Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same',kernel_initializer='he_normal')(conv6), conv3], axis=3)
	conv7 = Conv2D(128, (3, 3), activation='relu', padding='same',kernel_initializer='he_normal')(up7)
	conv7 = Conv2D(128, (3, 3), activation='relu', padding='same',kernel_initializer='he_normal')(conv7)

	up8 = concatenate([Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same',kernel_initializer='he_normal')(conv7), conv2], axis=3)
	conv8 = Conv2D(64, (3, 3), activation='relu', padding='same',kernel_initializer='he_normal')(up8)
	conv8 = Conv2D(64, (3, 3), activation='relu', padding='same',kernel_initializer='he_normal')(conv8)

	up9 = concatenate([Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same',kernel_initializer='he_normal')(conv8), conv1], axis=3)
	conv9 = Conv2D(32, (3, 3), activation='relu', padding='same',kernel_initializer='he_normal')(up9)
	conv9 = Conv2D(32, (3, 3), activation='relu', padding='same',kernel_initializer='he_normal')(conv9)

	conv10 = Conv2D(1, (1, 1), activation='sigmoid')(conv9)

	model = Model(inputs=[inputs], outputs=[conv10])

	# model.compile(optimizer=Adam(lr=1e-5), loss='binary_crossentropy', metrics=[dice_coef])
	model.compile(optimizer=Adam(lr=1e-5), loss=dice_coef_loss, metrics=[dice_coef])
	# sgd = SGD(lr=0.1, momentum=0.99, decay=1, nesterov=False)
	# model.compile(optimizer=sgd, loss=dice_coef_loss, metrics=[dice_coef])
	return model

# preprocess(imgs) will not be used in u-net V2
def preprocess(imgs):
	imgs_p = np.ndarray((imgs.shape[0], img_rows, img_cols), dtype=np.uint8)
	print("shape of imgs_p is {}".format(imgs_p.shape))
	
	for i in range(imgs.shape[0]):
		imgs_p[i] = imgs[i]
	
	imgs_p = imgs_p[..., np.newaxis]
	print("shape of imgs_p is {}".format(imgs_p.shape))
	return imgs_p

# loss function
def dice_coef(y_true, y_pred):
	y_true_f = K.flatten(y_true)
	y_pred_f = K.flatten(y_pred)
	intersection = K.sum(y_true_f * y_pred_f)
	return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_coef_loss(y_true, y_pred):
	return -dice_coef(y_true, y_pred)

# trian u-net and validate it
def train_and_predict(train_data_path, validation_data_path, train_mask_data_path, validation_mask_data_path):

	# initial img and mask generator
	image_datagen = ImageDataGenerator(featurewise_center=True,rescale=1./255)
	mask_datagen = ImageDataGenerator(rescale=1./255)

	## compute quantities required for featurewise normalization
	## (std, mean, and principal components if ZCA whitening is applied)
	
	# load imgs in a numpy matrix
	imgs_train_list = [os.path.join(root,file) for (root, dirs, files) in os.walk(train_data_path) for file in files]
	total = len(imgs_train_list)
	imgs_train = np.ndarray((total, img_rows, img_cols, channels), dtype=np.uint8)
	for index in range(total):
		img_train = cv2.imread(imgs_train_list[index], 0)
		imgs_train[index] = img_train[..., np.newaxis]
		print ('{} imgs has been loaded in imgs_train matrix'.format(index))

	# compute quantities required for featurewise normalization
	image_datagen.fit(imgs_train)

	# Provide the same seed and keyword arguments to the fit and flow methods
	seed = 1
	image_generator = image_datagen.flow_from_directory(
		train_data_path,
		target_size=(512,512),
		color_mode='grayscale',
		class_mode=None,
		seed=seed,
		batch_size=16)

	mask_generator = mask_datagen.flow_from_directory(
		train_mask_data_path,
		target_size=(512,512),
		color_mode='grayscale',
		class_mode=None,
		seed=seed,
		batch_size=16)

	# get train generator
	train_generator = izip(image_generator, mask_generator)

	# initial validation image data generater

	validation_image_datagen = ImageDataGenerator(featurewise_center=True,rescale=1./255)
	validation_mask_datagen = ImageDataGenerator(rescale=1./255)

	validation_image_datagen.fit(imgs_train)

	seed2 = 2
	validation_img_generator = validation_image_datagen.flow_from_directory(
		validation_data_path,
		target_size=(512,512),
		color_mode='grayscale',
		class_mode=None,
		seed=seed2,
		batch_size=16)
	validation_mask_generator = validation_mask_datagen.flow_from_directory(
		validation_mask_data_path,
		target_size=(512,512),
		color_mode='grayscale',
		class_mode=None,
		seed=seed2,
		batch_size=16)

	validation_generator = izip(validation_img_generator, validation_mask_generator)

	# instantiate a U-net 
	model = UNET()

	# make weight and tensorboard path
	weight_path = './weight'
	if not os.path.exits(weight_path):
		os.makedirs(weight_path)

	tensorboard_path = os.path.join('./mytensorboard','V3')
	if not os.path.exits(tensorboard_path):
		os.makedirs(tensorboard_path)

	model_checkpoint = ModelCheckpoint(os.path.join(weight_path,'2500weights.h5'), monitor='val_loss', save_best_only=True)
	model.fit_generator(train_generator, steps_per_epoch=1683, epochs=3000, verbose=1,
		validation_data=validation_generator,
		validation_steps=14,
		callbacks=[model_checkpoint,TensorBoard(log_dir=tensorboard_path)])

if __name__ == '__main__':
	# input data
	train_data_path = "data/train/img0/"
	validation_data_path = 'data/validation/img0'
	train_mask_data_path = 'data/train/mask0'
	validation_mask_data_path = 'data/validation/mask0'
	train_and_predict(train_data_path, validation_data_path, train_mask_data_path, validation_mask_data_path)

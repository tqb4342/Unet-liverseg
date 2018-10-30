from __future__ import print_function

import os
from skimage.transform import resize
from skimage.io import imsave
import numpy as np
from keras.models import Model
from keras.layers import Input, concatenate, Conv2D, MaxPooling2D, Conv2DTranspose
from keras.optimizers import Adam, SGD
from keras.callbacks import ModelCheckpoint
from keras import backend as K
from keras import initializers

# define the input size
img_rows = 512
img_cols = 512
channels = 1

# parameter for loss function
smooth = 1.

# input data
train_data_path = './np_data/train.npy'
# test_data_path = './np_data/test.npy'
train_mask_data_path = './np_data/train_mask.npy'
# test_mask_data_path = './np_data/test_mask.npy'

# define u-net architecture
def UNET():
	inputs = Input((img_rows, img_cols, channels))
	conv1 = Conv2D(32, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
	conv1 = Conv2D(32, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(conv1)
	pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

	conv2 = Conv2D(64, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
	conv2 = Conv2D(64, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(conv2)
	pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

	conv3 = Conv2D(128, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
	conv3 = Conv2D(128, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(conv3)
	pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

	conv4 = Conv2D(256, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(pool3)
	conv4 = Conv2D(256, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(conv4)
	pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

	conv5 = Conv2D(512, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(pool4)
	conv5 = Conv2D(512, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(conv5)

	up6 = concatenate([Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(conv5), conv4], axis=3)
	conv6 = Conv2D(256, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(up6)
	conv6 = Conv2D(256, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(conv6)

	up7 = concatenate([Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conv6), conv3], axis=3)
	conv7 = Conv2D(128, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(up7)
	conv7 = Conv2D(128, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(conv7)

	up8 = concatenate([Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv7), conv2], axis=3)
	conv8 = Conv2D(64, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(up8)
	conv8 = Conv2D(64, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(conv8)

	up9 = concatenate([Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(conv8), conv1], axis=3)
	conv9 = Conv2D(32, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(up9)
	conv9 = Conv2D(32, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(conv9)

	conv10 = Conv2D(1, (1, 1), activation='sigmoid')(conv9)

	model = Model(inputs=[inputs], outputs=[conv10])

	model.compile(optimizer=Adam(), loss=dice_coef_loss, metrics=[dice_coef])
	# sgd = SGD(lr=1, momentum=0.99, decay=1, nesterov=False)
	# model.compile(optimizer=sgd, loss=dice_coef_loss, metrics=[dice_coef])

	return model

def preprocess(imgs):
    imgs_p = np.ndarray((imgs.shape[0], img_rows, img_cols), dtype=np.uint8)
    imgs_p = imgs_p[..., np.newaxis]
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
def train_and_predict():

	# load train data and validate data
	# turn them into img matrix again
	imgs_train = preprocess(np.load(train_data_path))
	# imgs_test = preprocess(np.load(test_data_path))
	mask_train = preprocess(np.load(train_mask_data_path))
	# mask_test = preprocess(np.load(test_mask_data_path))

	# preprocess
	imgs_train = imgs_train.astype('float32')
	# imgs_test = imgs_test.astype('float32')

	mask_train = mask_train.astype('float32')
	# mask_test = mask_test.astype('float32')

	mask_train /= 255.
	# mask_test /= 255.

	# instantiate a U-net 
	model = UNET()

	model_checkpoint = ModelCheckpoint('weights.h5', monitor='val_loss', save_best_only=True)
	model.fit(imgs_train, mask_train, batch_size=16, nb_epoch=2000, verbose=1, shuffle=True,
              validation_split=0.2,
              callbacks=[model_checkpoint])

if __name__ == '__main__':
    train_and_predict()
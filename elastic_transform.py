# Import stuff
import os
import numpy as np
import pandas as pd
import cv2
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter
import matplotlib.pyplot as plt

# Function to distort image
def elastic_transform(image, alpha, sigma, alpha_affine, random_state=None):
    """Elastic deformation of images as described in [Simard2003]_ (with modifications).
    .. [Simard2003] Simard, Steinkraus and Platt, "Best Practices for
         Convolutional Neural Networks applied to Visual Document Analysis", in
         Proc. of the International Conference on Document Analysis and
         Recognition, 2003.

     Based on https://gist.github.com/erniejunior/601cdf56d2b424757de5
    """
    if random_state is None:
        random_state = np.random.RandomState(None)

    shape = image.shape
    shape_size = shape[:2]

    # Random affine
    center_square = np.float32(shape_size) // 2
    square_size = min(shape_size) // 3
    pts1 = np.float32([center_square + square_size, [center_square[0]+square_size, center_square[1]-square_size], center_square - square_size])
    pts2 = pts1 + random_state.uniform(-alpha_affine, alpha_affine, size=pts1.shape).astype(np.float32)
    M = cv2.getAffineTransform(pts1, pts2)
    image = cv2.warpAffine(image, M, shape_size[::-1], borderMode=cv2.BORDER_REFLECT_101)

    dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha
    dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha
    dz = np.zeros_like(dx)

    x, y, z = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]), np.arange(shape[2]))
    indices = np.reshape(y+dy, (-1, 1)), np.reshape(x+dx, (-1, 1)), np.reshape(z, (-1, 1))
    # indices = np.reshape(y+dy, (-1, 1)), np.reshape(x+dx, (-1, 1)), np.reshape(z, (-1, 1))

    return map_coordinates(image, indices, order=1, mode='reflect').reshape(shape)

# Define function to draw a grid
def draw_grid(im, grid_size):
    # Draw grid lines
    for i in range(0, im.shape[1], grid_size):
        cv2.line(im, (i, 0), (i, im.shape[0]), color=(255,))
    for j in range(0, im.shape[0], grid_size):
        cv2.line(im, (0, j), (im.shape[1], j), color=(255,))


if __name__ == '__main__':
    img_path =  '../data/train/img'
    mask_path = '../data/train/mask'
    # img_path =  '/home/changzhang/ liubo_workspace/tmp_for_test/img'
    # mask_path = '/home/changzhang/liubo_workspace/tmp_for_test/mask'

    img_list = sorted(os.listdir(img_path))
    mask_list = sorted(os.listdir(mask_path))

    img_num = len(img_list)
    mask_num = len(mask_list)

    assert img_num == mask_num, 'img nuimber is not equal to mask num.'

    count_total = 0 
    for i in range(img_num):
        im = cv2.imread(os.path.join(img_path, img_list[i]), -1)
        im_mask = cv2.imread(os.path.join(mask_path, mask_list[i]), -1)

        # # Draw grid lines
        # draw_grid(im, 50)
        # draw_grid(im_mask, 50)

        # Merge images into separete channels (shape will be (cols, rols, 2))
        im_merge = np.concatenate((im[...,None], im_mask[...,None]), axis=2)

        # get img and mask shortname
        (img_shotname,img_extension) = os.path.splitext(img_list[i])
        (mask_shotname,mask_extension) = os.path.splitext(mask_list[i])

        # Elastic deformation 10 times
        count = 0
              
        while count < 10:

            # Apply transformation on image
            im_merge_t = elastic_transform(im_merge, im_merge.shape[1] * 2, im_merge.shape[1] * 0.08, im_merge.shape[1] * 0.08)

            # Split image and mask
            im_t = im_merge_t[...,0]
            im_mask_t = im_merge_t[...,1]

            # save the new imgs and masks
            cv2.imwrite(os.path.join(img_path, img_shotname+'-'+str(count)+img_extension), im_t)
            cv2.imwrite(os.path.join(mask_path, mask_shotname+'-'+str(count)+mask_extension),im_mask_t )

            count += 1
            count_total += 1
        if count_total%100 == 0:
            print('Elastic deformation generated {} imgs',format(count_total))
            # # Display result
            # print 'Display result'
            # plt.figure(figsize = (16,14))
            # plt.imshow(np.c_[np.r_[im, im_mask], np.r_[im_t, im_mask_t]], cmap='gray')
            # plt.show()
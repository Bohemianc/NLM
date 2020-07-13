#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""code_info
@Time    : 2020 2020/7/13 9:29
@Author  : pink
@File    : nlm.py
"""
import cv2
import numpy as np
from numpy.matlib import randn
from matplotlib import pyplot as plt
# from skimage.util import random_noise

IMG_SIZE_X = 100
IMG_SIZE = IMG_SIZE_X ** 2


img = cv2.imread("lena.jpg", cv2.IMREAD_GRAYSCALE)
img = cv2.resize(img, (IMG_SIZE_X, IMG_SIZE_X))
img = img / 255.0

# add a gaussian white noise
# sigma_noise is the standard deviation
sigma_noise = 0.03
noise = sigma_noise * randn(img.shape)
noise_img = img + noise
# noise_img=random_noise(img, mode='gaussian', seed=None, clip=True, var=sigma_noise**2)

h = sigma_noise*10
R = 10
r = 3

# the denoised image
nl = np.zeros(IMG_SIZE)

# extend the image to prevent the index out of range
img_pad = np.pad(noise_img, R + r, "constant")

# calculate each pixel of denoised image
for i in range(IMG_SIZE):
    if i % 20 == 0 and i != 0:
        print("======%f done=====" % (i / IMG_SIZE))

    weight = np.zeros((2*R+1,2*R+1))
    cx = i // IMG_SIZE_X + R + r
    cy = i % IMG_SIZE_X + R + r
    c_img = img_pad[cx - r:cx + r + 1, cy - r:cy + r + 1]
    for dx in range(-R, R + 1):
        for dy in range(-R, R + 1):
            nx, ny = cx + dx, cy + dy
            weight[dx+R][dy+R] = np.square(img_pad[nx - r:nx + r + 1, ny - r:ny + r + 1] - c_img).sum()
    weight=np.exp(-weight/(h*h))
    sum_z = weight.sum()
    weight = weight / sum_z
    img_pad_part=img_pad[cx-R:cx+R+1,cy-R:cy+R+1]
    nl[i] = (weight * img_pad_part).sum()


nl = np.reshape(nl, (IMG_SIZE_X, IMG_SIZE_X))
plt.figure()
plt.subplot(221)
plt.imshow(img, 'gray')
plt.subplot(222)
plt.imshow(noise_img, 'gray')
plt.subplot(223)
plt.imshow(nl, 'gray')
plt.show()

# img = np.uint8(img * 255)
# noise_img = np.uint8(noise_img * 255)
# nl = np.uint8(nl * 255)
# cv2.imwrite("img.jpg", img)
# cv2.imwrite("noise_img.jpg", noise_img)
# cv2.imwrite("nl.jpg", nl)



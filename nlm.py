#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""code_info
@Time    : 2020 2020/7/13 9:29
@Author  : pink
@File    : nlm.py
"""
import numpy as np
import cv2

# from skimage.util import random_noise


def nlm(img, h, R=10, r=3, f=False):
    # f: whether add a Gaussian kernel to the Euclidean distance or not
    size_x = img.shape[0]
    size = size_x ** 2


    # the denoised image
    nl = np.zeros(size)

    # extend the image to prevent the index out of range
    img_pad = np.pad(img, R + r, "constant")

    # calculate each pixel of denoised image
    for i in range(size):
        # if i % 500 == 0 and i != 0:
        #     print("======%f done=====" % (i / size))

        weight = np.zeros((2 * R + 1, 2 * R + 1))
        cx = i // size_x + R + r
        cy = i % size_x + R + r
        c_img = img_pad[cx - r:cx + r + 1, cy - r:cy + r + 1]
        for dx in range(-R, R + 1):
            for dy in range(-R, R + 1):
                nx, ny = cx + dx, cy + dy
                weight[dx + R][dy + R] = np.square(img_pad[nx - r:nx + r + 1, ny - r:ny + r + 1] - c_img).sum()
        if f:
            weight = cv2.GaussianBlur(weight, (3, 3), 0.6)
        weight = np.exp(-weight / (h * h))
        sum_z = weight.sum()
        weight = weight / sum_z
        img_pad_part = img_pad[cx - R:cx + R + 1, cy - R:cy + R + 1]
        nl[i] = (weight * img_pad_part).sum()

    return nl.reshape((size_x, size_x))

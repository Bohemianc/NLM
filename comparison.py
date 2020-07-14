#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""code_info
@Time    : 2020 2020/7/14 12:34
@Author  : pink
@File    : comparison.py
"""
import cv2
import numpy as np
from matplotlib import pyplot as plt
# from skimage.util import random_noise
import nlm
import gauss_filter
import aniso_filter
import neighbor_filter

IMG_SIZE_X = 100
IMG_SIZE = IMG_SIZE_X ** 2

img = cv2.imread("lena.jpg", cv2.IMREAD_GRAYSCALE)
img = cv2.resize(img, (IMG_SIZE_X, IMG_SIZE_X))
img = img / 255.0

# add a gaussian white noise
# sigma_noise is the standard deviation
sigma_noise = 0.03
noise = sigma_noise * np.random.randn(img.shape[0], img.shape[1])
noise_img = img + noise
# noise_img=random_noise(img, mode='gaussian', seed=None, clip=True, var=sigma_noise**2)

nl = nlm.nlm(noise_img, 10 * sigma_noise)
gauss = gauss_filter.gf(noise_img, (3, 3), 1)
anisoid = aniso_filter.af(img)
neighbor = neighbor_filter.nf(img, (3, 3))

# plot the restored images
plt.figure()
res = []
res.extend((img, noise_img, nl, gauss, anisoid, neighbor))
pos = [x for x in range(231, 231 + len(res))]
for i in range(len(res)):
    plt.subplot(pos[i])
    plt.imshow(res[i], 'gray')
    plt.axis('off')
plt.show()

# plot the method noise
plt.figure()
res1 = []
res1.extend((img, noise_img, img - nl, img - gauss, img - anisoid, img - neighbor))
for i in range(len(res)):
    plt.subplot(pos[i])
    plt.imshow(res1[i], 'gray')
    plt.axis('off')
plt.show()

# calculate MSE for every approach


def autolabel(rects, ax, xpos='center'):
    """
    Attach a text label above each bar in *rects*, displaying its height.
    *xpos* indicates which side to place the text w.r.t. the center of
    the bar. It can be one of the following {'center', 'right', 'left'}.
    """

    xpos = xpos.lower()  # normalize the case of the parameter
    ha = {'center': 'center', 'right': 'left', 'left': 'right'}
    offset = {'center': 0.5, 'right': 0.55, 'left': 0.45}  # x_txt = x + w*off

    for rect in rects:
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width() * offset[xpos], 1.01 * height,
                '{:.2f}'.format(height), ha=ha[xpos], va='bottom', fontdict={"fontsize": 16})


fig, ax = plt.subplots()
x = ["NLM", "GF", "AF", "NF"]
y = list(map(lambda t: np.sum(np.square(img - t)), [nl, gauss, anisoid, neighbor]))
for i in range(4):
    print(f"{x[i]}: {y[i]}")
rects = ax.bar(range(4), y, width=0.4)
autolabel(rects, ax, "center")
plt.xticks(range(4), x)
plt.yticks(range(0, 60, 10))
plt.show()
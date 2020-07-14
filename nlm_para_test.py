#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""code_info
@Time    : 2020 2020/7/14 12:50
@Author  : pink
@File    : nlm_para_test.py
"""
import nlm
import numpy as np
import cv2
from matplotlib import pyplot as plt

IMG_SIZE_X = 100
IMG_SIZE = IMG_SIZE_X ** 2

img = cv2.imread("lena.jpg", cv2.IMREAD_GRAYSCALE)
img = cv2.resize(img, (IMG_SIZE_X, IMG_SIZE_X))
img = img / 255.0

# add a gaussian white noise
# sigma_noise is the standard deviation
sigma_noise = 0.025
noise = sigma_noise * np.random.randn(img.shape[0], img.shape[1])
noise_img = img + noise

# test for parameter h
print("======h:======")
hs=[5,10,15]
nlhs=[]
for h in hs:
    img_t=nlm.nlm(noise_img,sigma_noise*h)
    nlhs.append(img_t)
    # cv2.imwrite(f'results/h_{h//5-1}.jpg', np.uint8(img_t * 255))
    print(np.sum(np.square(img-img_t))/IMG_SIZE)

pos=[x for x in range(131,131+len(hs))]
plt.figure()
for i in range(len(hs)):
    plt.subplot(pos[i])
    plt.imshow(nlhs[i], 'gray')
    plt.axis('off')
plt.show()


# test for parameter R, the radius of the search window
print("======R:======")
Rs=[1,5,10]
nlRs=[]
for R in Rs:
    img_t=nlm.nlm(noise_img,sigma_noise*10,R)
    nlRs.append(img_t)
    print(np.sum(np.square(img-img_t))/IMG_SIZE)
    # cv2.imwrite(f'results/R_{i}.jpg', np.uint8(img_t * 255))

pos=[x for x in range(131,131+len(Rs))]
plt.figure()
for i in range(len(Rs)):
    plt.subplot(pos[i])
    plt.imshow(nlRs[i], 'gray')
    plt.axis('off')
plt.show()

# test for parameter r, the radius of the similarity window
print("======r:======")
rs = [0, 3, 6]
nlrs = []
for r in rs:
    img_t=nlm.nlm(noise_img, sigma_noise * 10, 10, r)
    nlrs.append(img_t)
    print(np.sum(np.square(img - img_t))/IMG_SIZE)
    # cv2.imwrite(f'results/r1_{r//3}.jpg', np.uint8(img_t * 255))

pos = [x for x in range(131, 131 + len(rs))]
plt.figure()
for i in range(len(rs)):
    plt.subplot(pos[i])
    plt.imshow(nlrs[i], 'gray')
    plt.axis('off')
plt.show()

# compare the results
# whether add a Gaussian kernel to the Euclidean distance or not
print("======G:======")
nlgs = []
nlgs.append(noise_img)
nlgs.append(nlm.nlm(noise_img, sigma_noise * 10, 10, 3, False))
nlgs.append(nlm.nlm(noise_img, sigma_noise * 10, 10, 3, True))

for i in range(3):
    print(np.sum(np.square(img-nlgs[i] ))/IMG_SIZE)
    # cv2.imwrite(f'results/g_{i}.jpg', np.uint8(nlgs[i] * 255))

pos = [x for x in range(131, 131 + len(nlgs))]
plt.figure()
for i in range(len(nlgs)):
    plt.subplot(pos[i])
    plt.imshow(nlgs[i], 'gray')
    plt.axis('off')
plt.show()

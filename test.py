#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""code_info
@Time    : 2020 2020/7/13 0:09
@Author  : pink
@File    : test.py
"""
import cv2
import numpy as np

img = cv2.imread("1.jpg", cv2.IMREAD_GRAYSCALE)
cv2.imshow('image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()


def fun():
    print("pink")

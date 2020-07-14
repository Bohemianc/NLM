#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""code_info
@Time    : 2020 2020/7/14 12:21
@Author  : pink
@File    : gauss_filter.py
"""
import cv2


def gf(img, shape=(3, 3), sigma=1):
    return cv2.GaussianBlur(img, shape, sigma)

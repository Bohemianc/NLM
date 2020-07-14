#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""code_info
@Time    : 2020 2020/7/14 12:26
@Author  : pink
@File    : neighbor_filter.py
"""
import cv2


def nf(img,shape=(3,3)):
    return cv2.blur(img,shape)
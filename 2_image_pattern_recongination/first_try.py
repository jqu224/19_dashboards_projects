#!/usr/bin/env python
import time
from matplotlib.pyplot import imshow
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys
import random
import glob
%matplotlib inline
path = r'R:\Personal Folders\Jiacheng Qu\4_white_spots\50um'
files = glob.glob(path + "/**/*.jpg", recursive=True)
kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
temp_error_count = []

for name in files:
    plt.figure(figsize=(20, 10))
    img = cv2.imread(name, 0)
    quartiles = np.percentile(img, range(0, 100, 3), interpolation='midpoint')
    print(quartiles)

    clahe = cv2.createCLAHE(clipLimit=0.05, tileGridSize=(8, 8))
    cl1 = clahe.apply(img)

    quartiles = np.percentile(cl1, range(0, 100, 3), interpolation='midpoint')
    print(quartiles, name)
    plt.subplot(1, 6, 1), plt.imshow(cl1, cmap='gray'), plt.title("c l 1")
    cl_copy = cl1
    ret, cl1 = cv2.threshold(cl1, quartiles[-12], 255, cv2.THRESH_BINARY)
    dilation = cv2.dilate(cl1, kernel, iterations=3)
    erosion = cv2.erode(dilation, kernel, iterations=11)
    cl1 = erosion

    quartiles = np.percentile(cl1, range(0, 100, 3), interpolation='midpoint')
    print(quartiles)
    plt.subplot(1, 6, 2), plt.imshow(
        cl1, cmap='gray'), plt.title("c l 1 with threshold")

    img_bw = 255 * (cl1 > 5).astype('uint8')
    se1 = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    se2 = cv2.getStructuringElement(cv2.MORPH_RECT, (33, 33))
    mask = cv2.morphologyEx(img_bw, cv2.MORPH_CLOSE, se1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, se2)
    mask = mask / 255
    cl1 = (cl1 * mask).astype('uint8')

    plt.subplot(1, 6, 3), plt.imshow(cl1, cmap='gray'), plt.title("edged")

    contours, hierarchy = cv2.findContours(
        cl1, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_TC89_KCOS)
    cnt = contours[0]
    (x, y), radius = cv2.minEnclosingCircle(cnt)
    center = (int(x), int(y))
    radius = int(radius)
    edges_4 = (center[0] - radius, center[0] + radius, center[1] - radius,
               center[1] + radius)
    print("edge4: ", edges_4)
    if edges_4[0] <= 0 or edges_4[2] <= 0 or edges_4[1] >= 640 or edges_4[
            3] >= 480:
        radius = -1
        plt.subplot(1, 6, 4), plt.imshow(img), plt.title("no contours")
    else:
        con = cv2.circle(img, center, radius, 255, 22)
        print(center, radius, "ordinary")
        plt.subplot(1, 6, 4), plt.imshow(con), plt.title("contours")

    cimg = cv2.cvtColor(cl1, cv2.COLOR_GRAY2BGR)
    circles = cv2.HoughCircles(
        cl1, cv2.HOUGH_GRADIENT, 8, minDist=444, minRadius=144, maxRadius=222)

    circles = np.uint16(np.around(circles))
    for i in circles[0, :]:
        normalizedImg = np.zeros((480, 640))
        # draw the outer circle
        cv2.circle(con, (i[0], i[1]), i[2], (219, 112, 147), 5)
        print("location of 2nd option: ", i[0], i[1], i[2])

        edges_4 = (i[0] - radius, i[0] + radius, i[1] - radius, i[1] + radius)
        if edges_4[0] < 0 or edges_4[2] < 0 \
           or edges_4[1] > 640 or edges_4[3] > 480:
            pass
        elif i[2] > radius:
            radius = i[2]
            center = (i[0], i[1])

    plt.subplot(1, 6, 5), plt.imshow(con), plt.title("222")

    dim = (radius * 2, radius * 2)
    dst = 0
    b = cl_copy[center[1] - radius:center[1] + radius, center[0] -
                radius:center[0] + radius]
    for i in range(1, 11):
        ref_img_1 = cv2.imread(
            r"R:\Personal Folders\Jiacheng Qu\4_white_spots\die" + str(i) +
            r".png", cv2.IMREAD_UNCHANGED)
        ref_img_1 = ref_img_1[:, :, 3]
        ref_img_1 = cv2.resize(ref_img_1, dim)
        #         print("size of ref ", ref_img_1.shape)
        #         print("size of cl1 ", cl_copy.shape)
        b = cv2.addWeighted(b, 1, ref_img_1, 0.1, 0)
    plt.subplot(1, 6, 6), plt.imshow(b, cmap='gray'), plt.title("444")
    plt.show()

    plt.figure(figsize=(20, 10))

    ref_img_1 = cv2.imread(
        r"R:\Personal Folders\Jiacheng Qu\4_white_spots\die0.png",
        cv2.IMREAD_UNCHANGED)
    ref_img_1 = ref_img_1[:, :, 3]

    dim = (radius * 2, radius * 2)
    ref_img_1 = cv2.resize(ref_img_1, dim)

    erosion = cv2.erode(cl_copy, kernel, iterations=3)
    dilation = cv2.dilate(cl_copy, kernel, iterations=3)

    cl_copy_0 = np.zeros((480, 640))
    cl_copy_0 = cv2.normalize(dilation - erosion, normalizedImg, 0, 1,
                              cv2.NORM_MINMAX)
    cl_copy_0 = np.multiply(cl_copy_0, 255)

    b = cl_copy_0[center[1] - radius:center[1] + radius, center[0] -
                  radius:center[0] + radius]

    dst = cv2.addWeighted(b, 0.9, ref_img_1, 0.1, 0)

    normalizedImg = np.zeros(dim)
    normalizedImg = cv2.normalize(ref_img_1, normalizedImg, 0, 1,
                                  cv2.NORM_MINMAX)
    c_0 = np.multiply(b, normalizedImg)

    plt.figure(figsize=(20, 10))
    a = [0] * 10
    for it in range(1, 11):
        ref_img_1 = cv2.imread(
            r"R:\Personal Folders\Jiacheng Qu\4_white_spots\die" + str(it) +
            r".png", cv2.IMREAD_UNCHANGED)
        ref_img_1 = ref_img_1[:, :, 3]

        dim = (radius * 2, radius * 2)
        ref_img_1 = cv2.resize(ref_img_1, dim)

        erosion = cv2.erode(cl_copy, kernel, iterations=3)
        dilation = cv2.dilate(cl_copy, kernel, iterations=3)
        cl_copy_0 = np.zeros((480, 640))
        cl_copy_0 = cv2.normalize(dilation - erosion, normalizedImg, 0, 1,
                                  cv2.NORM_MINMAX)
        cl_copy_0 = np.multiply(cl_copy_0, 255)

        b = cl_copy_0[center[1] - radius:center[1] + radius, center[0] -
                      radius:center[0] + radius]

        dst = cv2.addWeighted(b, 0.9, ref_img_1, 0.1, 0)

        normalizedImg = np.zeros(dim)
        normalizedImg = cv2.normalize(ref_img_1, normalizedImg, 0, 1,
                                      cv2.NORM_MINMAX)
        c = np.multiply(b, normalizedImg) - c_0
        a[it - 1] = (sum(sum(c[:])))
        plt.subplot(2, 5, it), plt.imshow(c, cmap='gray'), plt.title(it)
    plt.show()
    temp_error_count.append(a)

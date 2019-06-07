#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  1 08:49:09 2019

@author: Waysure
"""
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
from PIL import Image
import json

"Lines of code for simple gray scale thresholding"


image = cv.imread('SkyImage1.jpg')
imop = Image.open('SkyImage1.jpg')
pix = imop.load()
pix_sky = pix[320,10]

cv.waitKey(0)
cv.destroyAllWindows()
#Target obtained from the image itself
#target = [115,154,211]
target = [pix_sky[2],pix_sky[1],pix_sky[0]]

#Range manually tuned
ranged = 20

R_min = target[0]-ranged
G_min = target[1]-ranged
B_min = target[2]-ranged

R_max = target[0]+ranged
G_max = target[1]+ranged
B_max = target[2]+ranged

if R_min < 0:

    R_min = 0

if G_min < 0:

    G_min = 0

if B_min < 0:

    B_min = 0


if R_max > 255:

    R_max = 255

if G_max > 255:

    G_max = 255

if B_max > 255:

    B_max = 255


lower = [R_min,G_min,B_min]
upper = [R_max,G_max,B_max]

#create NumPy arrays from the boundaries
lower = np.array(lower, dtype = "uint8")
upper = np.array(upper, dtype = "uint8")

#Find colors inside specified boundaries and apply masking:
mask = cv.inRange(image,lower,upper)
output = cv.bitwise_and(image,image,mask = mask)
imagegray = cv.cvtColor(output,cv.COLOR_RGB2GRAY)
TotalPixels = imagegray.size
SkyPixels = cv.countNonZero(imagegray)
print("Sky fraction: "+str(SkyPixels/TotalPixels))

#plt.imshow(output)
cv.imshow("Images",np.hstack([image,output]))
cv.waitKey(0)
cv.destroyAllWindows()

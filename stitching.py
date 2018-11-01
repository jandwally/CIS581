'''
  File name: stitching.py
'''

# libs
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy import signal
from PIL import Image
from skimage.feature import corner_peaks, peak_local_max

# files
from corner_detector import *
from anms import *
from feat_desc import *
from feat_match import *

'''
  Convert RGB image to gray one manually
  - Input I_rgb: 3-dimensional rgb image
  - Output I_gray: 2-dimensional grayscale image
'''
def rgb2gray(I_rgb):
  r, g, b = I_rgb[:, :, 0], I_rgb[:, :, 1], I_rgb[:, :, 2]
  I_gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
  return I_gray

# Image blending script:
def stitching():
    img = np.array([[1,1,0],[1,0,0],[0,0,0]])
    print(corner_harris(img))

    # load images
    print("Opening images...")
    image1 = np.array(Image.open("images/philly-3.jpg").convert('RGB'))
    image2 = np.array(Image.open("images/philly-4.jpg").convert('RGB'))

    # corner_detector
    print("Running corner detection...")
    corner_matrix1 = corner_detector(rgb2gray(image1))
    corner_matrix2 = corner_detector(rgb2gray(image2))
    #print(corner_matrix1)

    # adaptive non-maximal suppression
    print("Non-maximal suppression...")
    max_pts = 20
    corners1 = anms(corner_matrix1, max_pts)
    corners2 = anms(corner_matrix2, max_pts)
    # print("corners1", corners1)
    # print("corners2", corners2)

    ''' TEST '''
    # Display image
    plt.subplot(121)
    plt.imshow(image1)
    plt.plot(corners1[0], corners1[1], 'ro')

    plt.subplot(122)
    plt.imshow(image2)
    plt.plot(corners2[0], corners2[1], 'ro')

    plt.show()

    # get descriptors
    print("Get descriptor vectors...")
    descriptors1 = feat_desc(rgb2gray(image1), corners1[1], corners1[0])
    descriptors2 = feat_desc(rgb2gray(image2), corners2[1], corners2[0])
    print(descriptors1.shape)
    print(descriptors2.shape)

    # find the matches
    print("Finding matches...")
    matches_idx = feat_match(descriptors1, descriptors2)
    #matches_idx = np.arange(0, corners1[0].shape[0])

    # preparing matched points
    num_match = matches_idx.shape[0]
    x1, y1 = corners1[1], corners1[0]
    x2, y2 = np.zeros(num_match).astype(int), np.zeros(num_match).astype(int)

    idx = np.arange(0, num_match)
    # idx = np.arange(0, num_match)
    x2[matches_idx] = x1[idx]
    y2[matches_idx] = y1[idx]

    print("matches", matches_idx)
    print("idx:", idx)
    print("x1:", x1)
    print("y1:", y1)
    print("x2:", x2)
    print("y2:", y2)

    # RANSAC
    print("Doing RANSAC...")


if __name__ == "__main__":
    stitching()



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

    # load images
    print("Opening images...")
    image1 = np.array(Image.open("images/philly-3.jpg").convert('RGB'))
    image2 = np.array(Image.open("images/philly-4.jpg").convert('RGB'))

    # corner_detector
    print("Running corner detection...")
    corner_matrix1 = corner_detector(rgb2gray(image1))
    corner_matrix2 = corner_detector(rgb2gray(image2))
    print(corner_matrix1)

    # for now just do this
    print("Non-maximal suppression...")
    corners1 = corner_peaks(corner_matrix1, min_distance=3).transpose()
    corners2 = corner_peaks(corner_matrix2, min_distance=3).transpose()
    print("corners1", corners1)
    print("corners2", corners2)

    ''' TEST '''
    # Display image
    plt.subplot(121)
    plt.imshow(image1)
    plt.plot(corners1[1], corners1[0], 'ro')

    plt.subplot(122)
    plt.imshow(image2)
    plt.plot(corners2[1], corners2[0], 'ro')

    plt.show()

    # get descriptors


if __name__ == "__main__":
    stitching()


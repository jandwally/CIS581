'''
  File name: stitching.py
'''

# libs
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy import signal
from PIL import Image
from skimage.feature import peak_local_max
from scipy.ndimage.interpolation import geometric_transform, map_coordinates
import cv2

# files
from corner_detector import corner_detector
from anms import anms
from feat_desc import feat_desc
from feat_match import feat_match
from ransac_est_homography import ransac_est_homography
from mymosaic import mymosaic

# constants
ERROR_THRESH = 0.5

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

    sample1 = np.array(Image.open("test_img/1L.jpg").convert('RGB'))
    sample2 = np.array(Image.open("test_img/1M.jpg").convert('RGB'))
    sample3 = np.array(Image.open("test_img/1R.jpg").convert('RGB'))

    philly1 = np.array(Image.open("images/philly-3.jpg").convert('RGB'))
    philly2 = np.array(Image.open("images/philly-4.jpg").convert('RGB'))
    philly3 = np.array(Image.open("images/philly-5.jpg").convert('RGB'))


    # First mosaic: sample pics
    image_inputs = np.array([ sample1, sample2, sample3 ]).astype(np.ndarray)
    mosaic1 = mymosaic(image_inputs)

    # Second mosaic: custom pics
    image_inputs = np.array([ philly1, philly2, philly3 ]).astype(np.ndarray)
    mosaic2 = mymosaic(image_inputs)


if __name__ == "__main__":
    stitching()


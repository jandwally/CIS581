'''
  File name: mymosaic.py
  Author:
  Date created:
'''

'''
  File clarification:
    Produce a mosaic by overlaying the pairwise aligned images to create the final mosaic image. If you want to implement
    imwarp (or similar function) by yourself, you should apply bilinear interpolation when you copy pixel values. 
    As a bonus, you can implement smooth image blending of the final mosaic.
    - Input img_input: M elements numpy array or list, each element is a input image.
    - Outpuy img_mosaic: H × W × 3 matrix representing the final mosaic image.
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

# constants
ERROR_THRESH = 0.5
MAX_PTS = 200

'''
  Convert RGB image to gray one manually
  - Input I_rgb: 3-dimensional rgb image
  - Output I_gray: 2-dimensional grayscale image
'''
def rgb2gray(I_rgb):
  r, g, b = I_rgb[:, :, 0], I_rgb[:, :, 1], I_rgb[:, :, 2]
  I_gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
  return I_gray


# image1: centered image
# image2: image to warp
def mosaic(image1, image2):

  ''' Get the corners of the images '''
  print("Running corner detection...")
  corner_matrix1 = corner_detector(rgb2gray(image1))
  corner_matrix2 = corner_detector(rgb2gray(image2))


  ''' Adaptive non-maximal suppression '''
  print("Non-maximal suppression...")
  max_pts = MAX_PTS
  corners1 = anms(corner_matrix1, max_pts)[0:2]
  corners2 = anms(corner_matrix2, max_pts)[0:2]


  # FINAL: Display corners after ANMS
  def display_after_anms():
    # Display image
    plt.imshow(image1)
    plt.plot(corners1[0], corners1[1], 'ro')
    plt.show()

    plt.imshow(image2)
    plt.plot(corners2[0], corners2[1], 'ro')
    plt.show()
  display_after_anms()


  ''' Get descriptors '''
  print("Get descriptor vectors...")
  descriptors1 = feat_desc(rgb2gray(image1), corners1[0], corners1[1])
  descriptors2 = feat_desc(rgb2gray(image2), corners2[0], corners2[1])


  ''' Finding feature matches '''
  print("Finding matches...")
  matches_idx = feat_match(descriptors1, descriptors2)

  # Boolean vector corresponding to whether or not there was a match found
  num_match = matches_idx.shape[0]
  has_match = np.where(matches_idx != -1)
  matches_idx[np.where(matches_idx == -1)] = 0

  # Pair up the matches
  x1, y1 = corners1[0], corners1[1]
  x2, y2 = np.zeros(num_match).astype(int), np.zeros(num_match).astype(int)
  idx = np.arange(0, num_match)
  x2[idx] = corners2[0][matches_idx]
  y2[idx] = corners2[1][matches_idx]

  # Get rid of non-matches
  x1 = x1[has_match]
  y1 = y1[has_match]
  x2 = x2[has_match]
  y2 = y2[has_match]


  ''' RANSAC '''
  print("Doing RANSAC...")
  homography, inlier_idx = ransac_est_homography(x1, y1, x2, y2, ERROR_THRESH)

  # FINAL: Visualize matches after RANSAC
  def show_ransac_12():
    big_im = np.concatenate((image1, image2), axis=1)
    plt.imshow(big_im)

    x2_shift = x2 + image1.shape[1]
    for i in range(x1.shape[0]):
        if inlier_idx[i] == 1:
            plt.plot([x1[i], x2_shift[i]], [y1[i], y2[i]], marker = "o")
    plt.show()
  #show_ransac_12()

  ''' Mosaicing '''

  # Apply homography to corner points, to get size of canvas
  h, w = image2.shape[0:2]
  x_corners = np.array([
      [ 0,   0, w-1, w-1 ],
      [ 0, h-1,   0, h-1 ],
      [ 1,   1,   1,   1 ]
  ])
  homography_inv = np.linalg.inv(homography)
  x_corners_trans = np.dot(homography_inv, x_corners)
  x_corners_trans = x_corners_trans / x_corners_trans[2]

  ''' Create canvas '''
  min_x = np.floor(np.min(x_corners_trans[0])).astype(int)
  min_y = np.floor(np.min(x_corners_trans[1])).astype(int)
  max_x = np.ceil(np.max(x_corners_trans[0])).astype(int)
  max_y = np.ceil(np.max(x_corners_trans[1])).astype(int)

  # Find required padding
  neg_padding = -1 * min(np.array([0, min_y, min_x])).astype(int)
  pos_padding = max(np.array([max_y-h, max_x-w])).astype(int)

  # Compute required height+width
  num_colors = image1.shape[2]
  h1, w1 = image1.shape[0:2]
  mosaic = np.zeros((neg_padding + h1 + pos_padding, neg_padding + w1 + pos_padding, num_colors)).astype(np.uint8)
  mosaic[neg_padding:h1+neg_padding, neg_padding:w1+neg_padding, :] = image1

  # Define function to warp with
  def apply_homography(output_coords):

      # Do homography
      x_output = np.array([ output_coords[1], output_coords[0], 1 ]).reshape((3,1)).astype(int)
      x_transform = np.dot(homography, x_output)
      x_transform = x_transform / x_transform[2]

      # Add constant shift to deal with padding
      x_input = x_transform[1] - neg_padding, x_transform[0] - neg_padding
      return x_input

  # Add a translation to the homography, based on negative padding
  translation = np.array([
      [1, 0, neg_padding],
      [0, 1, neg_padding],
      [0, 0, 1],
  ])
  homography_trans = np.dot(translation, homography_inv)

  # Warp image
  print("Warping image...")
  warped_image = np.zeros(mosaic.shape).astype(np.uint8)
  warped_image = cv2.warpPerspective(image2, homography_trans, (warped_image.shape[1], warped_image.shape[0]))


  ''' Combining images '''
  print("Combining images...")

  # Copy the warped image into the mosaic
  where_nonzero_warped = (np.sum(warped_image, axis=2) > 0)
  where_nonzero_original = (np.sum(mosaic, axis=2) > 0)

  for r in range(mosaic.shape[0]):
      for c in range(mosaic.shape[1]):

          # Both: average the two
          if where_nonzero_original[r,c] and where_nonzero_warped[r,c]:
              mosaic[r,c,:] = 0.5 * mosaic[r,c,:] + 0.5 * warped_image[r,c,:]

          # Warped image only
          elif where_nonzero_warped[r,c]:
              mosaic[r,c,:] = warped_image[r,c,:]

  return mosaic





def mymosaic(image_input):

  # Take apart the input into three input images
  image1 = image_input[0].astype(np.uint8)     # left image
  image2 = image_input[1].astype(np.uint8)     # middle image
  image3 = image_input[2].astype(np.uint8)     # right image

  # Call on the first two pictures
  print("Mosaicing images 1 and 2...")
  mosaic12 = mosaic(image2, image1)
  plt.imshow(mosaic12)
  plt.show()

  # Call on next two
  print("Mosaicing images 2 and 3...")
  mosaic123 = mosaic(mosaic12, image3)

  # Final output
  def show_mosaic():
    plt.imshow(mosaic123)
    plt.show()
  show_mosaic()

  return mosaic123
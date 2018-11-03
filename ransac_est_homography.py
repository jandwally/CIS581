'''
  File name: ransac_est_homography.py
  Author:
  Date created:
'''

'''
  File clarification:
    Use a robust method (RANSAC) to compute a homography. Use 4-point RANSAC as 
    described in class to compute a robust homography estimate:
    - Input x1, y1, x2, y2: N × 1 vectors representing the correspondences feature coordinates in the first and second image. 
                            It means the point (x1_i , y1_i) in the first image are matched to (x2_i , y2_i) in the second image.
    - Input thresh: the threshold on distance used to determine if transformed points agree.
    - Outpuy H: 3 × 3 matrix representing the homograph matrix computed in final step of RANSAC.
    - Output inlier_ind: N × 1 vector representing if the correspondence is inlier or not. 1 means inlier, 0 means outlier.
'''

import numpy as np
from scipy.optimize import least_squares

from est_homography import est_homography

NUM_RANSAC = 1000
MIN_CONSENSUS = 0

def ransac_est_homography(x1, y1, x2, y2, thresh):
  n = x1.shape[0]
  ERROR_THRESH = thresh

  ''' One call to RANSAC '''
  def ransac():

    ''' 1. Randomly select 4 points '''
    rands = np.random.permutation(n)[0:4]
    rand_points1, rand_points2 = np.zeros(4).astype(int), np.zeros(4).astype(int)
    rand_points1 = x1[rands], y1[rands]
    rand_points2 = x2[rands], y2[rands]

    ''' 2. Compute homography relating these four matches '''
    homography = est_homography(rand_points1[0], rand_points1[1], rand_points2[0], rand_points2[1])

    ''' 3. Find number of inliers for this homography '''

    # Apply homography
    x_source = np.concatenate((x1.reshape((n,1)), y1.reshape((n,1)), np.ones((n,1))), axis=1).transpose((1,0)).astype(int)
    x_target = np.concatenate((x2.reshape((n,1)), y2.reshape((n,1)), np.ones((n,1))), axis=1).transpose((1,0)).astype(int)
    x_transform = np.dot(homography, x_source)

    # Normalize z coord back to 1
    x_transform = x_transform / x_transform[2]

    # Compute SSD between transformed source and target
    differences = np.sum((np.power(x_transform - x_target, 2)), axis=0)

    # Find whether each point is an inlier (thresholding); set them to 1 if they pass
    inliner_idx = np.zeros(n).astype(int)
    inliner_idx[np.where(differences < ERROR_THRESH)] = 1
    num_inliners = np.count_nonzero(inliner_idx)

    ''' Return '''
    return homography, num_inliners, inliner_idx

  ''' 4. Repeat this process NUM_RANSAC times, and keep the one with the largest number of inliners '''
  best_homography = None
  most_inliers = MIN_CONSENSUS
  #most_inliers = 0
  inliner_idx = None

  # Do this many times, find the best
  for i in np.arange(0, NUM_RANSAC):
    homography, num_inliners, this_inliner_idx = ransac()
    #print("i =", i, ":", num_inliners, this_inliner_idx)

    # Save if this homography was better than the previous best
    if num_inliners > most_inliers:
      best_homography = homography
      most_inliers = num_inliners
      inliner_idx = this_inliner_idx

  print("BEST RANSAC:")
  print("best_homography:", best_homography)
  print("most_inliers:", most_inliers)

  ''' 5. Compute the least squares estimate for homography using all inlier matches '''

  # Get all inliers
  final_x1 = x1[inliner_idx == 1]
  final_y1 = y1[inliner_idx == 1]
  final_x2 = x2[inliner_idx == 1]
  final_y2 = y2[inliner_idx == 1]

  # Recompute final homography with all inliers, and return
  final_homography = est_homography(final_x1, final_y1, final_x2, final_y2)
  return final_homography, inliner_idx
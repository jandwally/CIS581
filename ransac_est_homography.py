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

from est_homography import est_homography

def ransac_est_homography(x1, y1, x2, y2, thresh):
  n = x1.shape[0]

  def ransac():

    ''' 1. Randomly select 4 points '''
    rands = np.random.permutation(n)[0:4]
    rand_points1 = x1[rands], y1[rands]
    rand_points2 = x2[rands], y2[rands]

    est_homo = est_homography(x, y, X, Y)

  '''
  We use RANSAC to pull out a minimal set of feature matches, estimate the homography and then
  count the number of inliers that agree with the current homography estimate. After repeated trials, the
  homography estimate with the largest number of inliers is used to compute a least squares estimate
  for the homography, which is then returned in the homography estimate H.
  H is a 3×3 matrix with 8 degrees of freedom. You need to solve a system of at least 8 linear equations
  to solve for the 8 unknowns of H. These 8 linear equations are based on the 4 pairs of corresponding
  points.
  Recall RANSAC:
  1. Randomly select four feature pairs
  2. Compute the homography relating the four selected matches with the function est_homography.m
  3. Compute the number of inliers to count (SSD distance after applying the estimated homography
  below the threshold thresh) how many matches agree with this estimate. Don’t forget to create
  inlier_ind
  4. Repeat the above random selection nRANSAC times and keep the estimate with the largest number
  of inliers
  5. Computes the least squares estimate for the homography using all of the matches previously esti-
  mated as inliers.
  '''

  return H, inlier_ind
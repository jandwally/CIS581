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

# files
from corner_detector import corner_detector
from anms import anms
from feat_desc import feat_desc
from feat_match import feat_match
from ransac_est_homography import ransac_est_homography

# constants
ERROR_THRESH = 1.5

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
    #img = np.array([[1,1,0],[1,0,0],[0,0,0]])
    #print(corner_harris(img))

    # load images
    print("Opening images...")
    #image1 = np.array(Image.open("images/philly-3.jpg").resize((500,500), Image.ANTIALIAS).convert('RGB'))
    #image2 = np.array(Image.open("images/philly-4.jpg").resize((500,500), Image.ANTIALIAS).convert('RGB'))
    image1 = np.array(Image.open("test_img/1M.jpg").convert('RGB'))
    image2 = np.array(Image.open("test_img/1R.jpg").convert('RGB'))

    # corner_detector
    print("Running corner detection...")
    corner_matrix1 = corner_detector(rgb2gray(image1))
    corner_matrix2 = corner_detector(rgb2gray(image2))
    #print(corner_matrix1)

    # adaptive non-maximal suppression
    print("Non-maximal suppression...")
    max_pts = 400
    corners1 = anms(corner_matrix1, max_pts)[0:2]
    corners2 = anms(corner_matrix2, max_pts)[0:2]
    # print("corners1", corners1)
    # print("corners2", corners2)

    ''' TEST '''
    # Display image
    '''plt.subplot(121)
                plt.imshow(image1)
                plt.plot(corners1[0], corners1[1], 'ro')
            
                plt.subplot(122)
                plt.imshow(image2)
                plt.plot(corners2[0], corners2[1], 'ro')
            
                plt.show()'''

    # get descriptors
    print("Get descriptor vectors...")
    descriptors1 = feat_desc(rgb2gray(image1), corners1[0], corners1[1])
    descriptors2 = feat_desc(rgb2gray(image2), corners2[0], corners2[1])
    print(descriptors1.shape)
    print(descriptors2.shape)

    # find the matches
    print("Finding matches...")
    matches_idx = feat_match(descriptors1, descriptors2)

    #'''
    ### for now use these to test
    #matches_idx = np.array([0, -1, 2, 3, 4, -1])
    #corners1 = np.array([
    #    [750, 985, 839, 1306, 1693, 1687],
    #    [488, 1142, 799, 537, 494, 1071]
    #])
    #corners2 = np.array([
    #    [171, 431, 274, 788, 1141, 1131],
    #    [464, 1174, 795, 549, 519, 1065]
    #])
    #print("matches", matches_idx)
    #'''

    ''' preparing matched points (have to deal with -1) '''

    # Boolean vector corresponding to whether or not there was a match found
    # For now, set all -1s to 0 so it works
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

    ''' Visualize matches '''

    # Display image
    big_im = np.concatenate((image1, image2), axis=1)
    plt.imshow(big_im)
    plt.plot(x1, y1, 'ro')
    x2_shift = x2 + image1.shape[1]
    plt.plot(x2_shift, y2, 'ro')
    for i in range(x1.shape[0]):
        plt.plot([x1[i], x2_shift[i]], [y1[i], y2[i]], marker = "o")
    plt.show()

    # RANSAC
    print("Doing RANSAC...")
    homography, inliner_idx = ransac_est_homography(x1, y1, x2, y2, ERROR_THRESH)

    ''' Mosaicing '''

    # Apply homography to corner points, to get size of canvas
    h, w = image2.shape[0:2]
    x_corners = np.array([
        [ 0, 0, w, w ],
        [ 0, h, 0, h ],
        [ 1, 1, 1, 1 ]
    ])
    x_corners_trans = np.dot(np.linalg.inv(homography), x_corners)
    #x_corners_trans = np.dot(homography, x_corners)
    x_corners_trans = x_corners_trans / x_corners_trans[2]
    print("x_corners_trans:\n", np.round(x_corners_trans).astype(int))

    # Create canvas
    min_x = np.floor(np.min(x_corners_trans[0])).astype(int)
    min_y = np.floor(np.min(x_corners_trans[1])).astype(int)
    max_x = np.ceil(np.max(x_corners_trans[0])).astype(int)
    max_y = np.ceil(np.max(x_corners_trans[1])).astype(int)

    neg_padding = -1 * min(np.array([0, min_y, min_x])).astype(int)
    pos_padding = max(np.array([max_y-h, max_x-w])).astype(int)
    print("required_padding", neg_padding, pos_padding)

    # Compute required height+width
    if (neg_padding == 0):
        h1, w1 = image1.shape[0:2]
        mosaic = np.zeros((h1 + pos_padding, w1 + pos_padding, 3)).astype(np.uint8)
        mosaic[0:h1, 0:w1, :] = image1
    if (neg_padding > 0):
        h1, w1 = image1.shape[0:2]
        mosaic = np.zeros((neg_padding + h1 + pos_padding, neg_padding + w1 + pos_padding, 3)).astype(np.uint8)
        mosaic[neg_padding:h1+neg_padding, neg_padding:w1+neg_padding, :] = image1

    # Plot the corners of image B oon image A
    plt.imshow(mosaic.astype(np.uint8))
    for i in range(4):
        plt.plot(x_corners_trans[0,i], x_corners_trans[1,i], 'ro')
    plt.show()

    # Define function to warp with
    def apply_homography(output_coords):

        # Do homography
        x_output = np.array([ output_coords[1], output_coords[0], 1 ]).astype(int)
        x_transform = np.dot(homography, x_output)
        x_transform = x_transform / x_transform[2]

        # Add constant shift to deal with padding
        x_input = x_transform[1] - neg_padding, x_transform[0] - neg_padding
        return x_input

    # Warp image
    warped_image = np.zeros(mosaic.shape).astype(np.uint8)
    for i in range(3):
        print("transform:", i)
        warped_image[:,:,i] = geometric_transform(image2[:,:,i], apply_homography, output_shape=warped_image.shape[0:2])
        plt.imshow(warped_image.astype(np.uint8))
        for i in range(4):
            plt.plot(x_corners_trans[0,i], x_corners_trans[1,i], 'ro')
        plt.show()

    # Copy the warped image into the mosaic
    where_nonzero_warped = (np.sum(warped_image, axis=2) > 0)
    where_nonzero_original = (np.sum(mosaic, axis=2) > 0)

    plt.imshow(255*(np.sum(mosaic, axis=2) > 0))
    plt.show()
    plt.imshow(255*(np.sum(warped_image, axis=2) > 0))
    plt.show()

    print("Combining images...")
    for r in range(mosaic.shape[0]):
        for c in range(mosaic.shape[1]):

            # Both: average the two
            if where_nonzero_original[r,c] and where_nonzero_warped[r,c]:
                mosaic[r,c,:] = 0.5 * mosaic[r,c,:] + 0.5 * warped_image[r,c,:]

            # Warped image only
            elif where_nonzero_warped[r,c]:
                mosaic[r,c,:] = warped_image[r,c,:]

    # Plot the corners of image B oon image A
    plt.imshow(mosaic.astype(np.uint8))
    '''
    for i in range(4):
        plt.plot(x_corners_trans[0,i], x_corners_trans[1,i], 'ro')
    '''
    plt.show()

if __name__ == "__main__":
    stitching()


'''
  File name: feat_match.py
  Author:
  Date created:
'''

'''
  File clarification:
    Matching feature descriptors between two images. You can use k-d tree to find the k nearest neighbour. 
    Remember to filter the correspondences using the ratio of the best and second-best match SSD. You can set the threshold to 0.6.
    - Input descs1: 64 × N1 matrix representing the corner descriptors of first image.
    - Input descs2: 64 × N2 matrix representing the corner descriptors of second image.
    - Outpuy match: N1 × 1 vector where match i points to the index of the descriptor in descs2 that matches with the
                    feature i in descriptor descs1. If no match is found, you should put match i = −1.
'''

import numpy as np

def feat_match(descs1, descs2):
  h,w= descs1.shape[0:2]
  b = np.zeros([h*w,h,w])
  for i in range(h):
    for j in range(w):
      curr = np.empty([h,w])
      curr.fill(descs2[i,j])
      b[(i*h+j),:,:] = curr
  matched = descs1[np.newaxis,:]-b[:,np.newaxis]
  return match

a= np.array([[5,4,3],[10,9,1]])
c = np.array([[1,2,3],[4,5,6]])
print(feat_match(a,c))


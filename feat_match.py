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
  pairs = []
  b = np.zeros([h*w,h,w])

  for i in range(descs2.shape[1]):
    d1i = descs2[:,i]
    #maybe np.subtract
    dist = np.sum(np.square(descs2-d1i), axis = 0) 
    dist.sort()
    best = dist[0]
    index_of_best = np.where(dist == best)
    sbest = dist[1]
    index_of_second_best = np.where(dist == sbest)
    d2i = desc2[index_of_best]
    sdist = np.sum(np.square(dist1-d2i), axis = 0)
    sdist.sort()
    pbest = sdist[0]
    psbest = sdist[1]
    ratio = best/pbest
    ratios = sbest/psbest

    if ratio < 0.7 and ratios < 0.7:
        #take index_of_best, index_of_second_best,
      pairs.append(index_of_best, index_of_second_best)
    return match

a= np.array([[5,4,3],[10,9,1]])
c = np.array([[1,2,3],[4,5,6]])
print(feat_match(a,c))


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
  '''For every feature in descs1, i.e. every colomn, lets find the squared 
  difference and put it in diff[:,j]''' 
  for i in range(descs1.shape[1]):
    d1i = descs2[:,i]
    diff = np.zeros((descs2.shape[0],descs2.shape[1]))
    for j in range(descs2.shape[1]):
      diff[:,j] = np.subtract(d1i,descs2[:,j])
    dist = np.sum(np.square(diff), axis = 0) 

    '''Lets sort the differences and get the "best" and the "sbest" i.e. second best
    as well as their indices'''
    dist.sort()
    best = dist[0]
    index_of_best = np.where(dist == best)
    sbest = dist[1]
    index_of_second_best = np.where(dist == sbest)

    '''Repeat the same process except inverse: we are now iterating through
    the columns of descs2 and finding the difference with the features (cols) of descs1'''

    d2i = descs2[:,index_of_best[0]].reshape(2,)
    '''sdiff is the equivalent of diff in the first part'''
    sdiff = np.zeros((descs1.shape[0],descs1.shape[1]))   
    for j in range(descs1.shape[1]):
      sdiff[:,j] = np.subtract(d2i, descs1[:,j])
    '''sdist is the equivalent of dist in the first part'''
    sdist = np.sum(np.square(sdiff), axis = 0)
    sdist.sort()
    
    '''Same process getting the best and second best'''
    pbest = sdist[0]
    psbest = sdist[1]

    '''Getting the ratios of both parts'''
    ratio = best/pbest
    ratios = sbest/psbest

    '''Check that both are greater than a threshold and if so adding that pair of featuers'''
    if ratio < 0.7 and ratios < 0.7:
        #take index_of_best, index_of_second_best,
      pairs.append((index_of_best[0], index_of_second_best[0]))
    return pairs

a= np.array([[5,4,3],[10,9,1]])
c = np.array([[1,2,3],[4,5,6]])
print(feat_match(a,c))

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
    - Outpuy match: N1 × 1 vector where match i points to the index of 
    the descriptor in descs2 that matches with the
    feature i in descriptor descs1. If no match is found, you should put match i = −1.
'''
'''[m,n] = scipy.spatial.distance.cdist instead of the for loop'''
import numpy as np
from scipy import spatial

def feat_match(descs1, descs2):
  h,w= descs1.shape[0:2]
  pairs = []
  print(descs2)
  b = np.zeros([h*w,h,w])
  '''For every feature in descs1, i.e. every colomn, lets find the squared 
  difference and put it in diff[:,j]''' 
 
  dist = spatial.distance.cdist(descs1,descs2, 'cityblock')
  '''Repeat the same process except inverse: we are now iterating through
      the columns of descs2 and finding the difference with the features (cols) of descs1'''
  print(dist)
  dist2 = spatial.distance.cdist(descs2, descs1, 'cityblock')

  for row in range(dist.shape[0]):
    current = dist[row,:]
    current.sort()
    '''Lets sort the differences and get the "best" and the "second_best" 
    as well as their indices'''
    best = current[0]
    second_best = current[1]
    index_of_best = np.where(dist[row,:] == best)
    index_of_second_best = np.where(dist[row,:] == second_best)
    best2 = dist2[index_of_best,row]
    second_best2 = dist2[index_of_second_best, row]
    ratio = best/second_best
    ratio2 = best2/second_best2
    ratio2 = ratio2[0,0]
    print(ratio)
    print(ratio2)
    '''Check that both are greater than a threshold and if so adding that pair of featuers'''
    if ratio < 0.7 and ratio2 < 0.7:
          #take index_of_best, index_of_second_best,
      pairs.append(index_of_best)
    else:
      pairs.append(-1)
  return pairs

a= np.array([[5,4,3],[10,9,1]])
c = np.array([[1,2,3],[5,4,2]])
print(feat_match(a,c))

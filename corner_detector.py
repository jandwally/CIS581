'''
  File name: corner_detector.py
  Author:
  Date created:
'''

'''
  File clarification:
    Detects corner features in an image. You can probably find free “harris” corner detector on-line, 
    and you are allowed to use them.
    - Input img: H × W matrix representing the gray scale input image.
    - Output cimg: H × W matrix representing the corner metric matrix.
'''
from skimage.feature import corner_harris

def corner_detector(img):
  cimg = corner_harris(img, method='k', k=0.01, eps=1e-06, sigma=3)
  return cimg

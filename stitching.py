
# stitching.py

'''
  File name: stitching.py
'''

# libs
import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import corner_harris
# files


# Image blending script:
def stitching():
    img = np.array([[1,1,0],[1,0,0],[0,0,0]])
    print(corner_harris(img))
    # corner_detector
    

if __name__ == "__main__":
    stitching()


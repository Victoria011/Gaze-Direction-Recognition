import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# generate ground truth Gaussian:
def gaussian(img, c1_x, c1_y, c2_x, c2_y, sigma = 1):
    """ Draw a gaussian map
    Args:
            width     : input img width
            height    : input img height
            c1_x       : center point 1 x 
            c1_y       : center point 1 y
            c2_x       : center point 2 x 
            c2_y       : center point 2 y
            sigma     : 
    """
    height = img.shape[0]
    width = img.shape[1]
    gaussian_map = np.zeros((height, width))
    
    # Generate gaussian
    for x in range(width):
        for y in range(height):
            if(x<width/2):
                g = np.exp(- ((x - c1_x) ** 2 + (y - c1_y) ** 2) / (2 * sigma ** 2))
            else:
                g = np.exp(- ((x - c2_x) ** 2 + (y - c2_y) ** 2) / (2 * sigma ** 2))
            gaussian_map[y, x] = g
         
    return gaussian_map

img = mpimg.imread('groundtruth/x/8.png')
print (img.shape)
# im = gaussian(img, 0, 0, 0, 0, 1) #0
# im = gaussian(img, 90, 110, 165, 110, 1) #1
# im = gaussian(img, 95, 110, 170, 110, 1) #2
# im = gaussian(img, 95, 110, 190, 110, 1) #3
# im = gaussian(img, 90, 120, 165, 120, 1) #4
# im = gaussian(img, 95, 120, 170, 120, 1) #5
# im = gaussian(img, 95, 120, 190, 120, 1) #6
# im = gaussian(img, 90, 130, 165, 130, 1) #7
# im = gaussian(img, 95, 130, 170, 130, 1) #8
# im = gaussian(img, 95, 130, 190, 130, 1) #9

# test 2 point in large 9 
# im = gaussian(img, 0, 0, 0, 0, 1) #0
# im = gaussian(img, 21, 64, 149, 64, 1) #1
# im = gaussian(img, 64, 64, 192, 64, 1) #2
# im = gaussian(img, 106, 64, 234, 64, 1) #3
# im = gaussian(img, 21, 128, 149, 128, 1) #4
# im = gaussian(img, 64, 128, 192, 128, 1) #5
# im = gaussian(img, 106, 128, 234, 128, 1) #6
# im = gaussian(img, 21, 192, 149, 192, 1) #7
im = gaussian(img, 64, 192, 192, 192, 1) #8
# im = gaussian(img, 95, 130, 190, 130, 1) #9

plt.figure()
plt.subplot(1,3,1)
plt.imshow(img)
plt.subplot(1,3,3)
plt.imshow(im)
plt.show()  # display it

plt.imsave('groundtruth/y2/frame8.png', im)
import numpy as np
from matplotlib import pyplot as plt
import skimage
from skimage import filters, data, io
from skimage.transform import resize
import random

img = io.imread('C:/path/to/your/file.jpg',as_gray=True)
#img = skimage.data.camera()

output_width = 40.0
quantificationlevels = 4

scale_factor = round(img.shape[0]/output_width)

img = resize(img,(round(img.shape[0]/scale_factor), round(img.shape[1]/scale_factor)),anti_aliasing=True)
img_orig = img

img_dy = skimage.filters.sobel_h(img, mask=None)
img_dx = skimage.filters.sobel_v(img, mask=None)

img = 1-img
img = np.round(np.multiply(img,quantificationlevels - 1)) + 1

randomness_vertex = 0.1 # 0 is no randomness, 0.01 is suitable.
randomness_position = 0.5
# min_pixel_size = 0.1
# max_pixel_size = 0.9

randomness_length = 0.0


def rotate_origin(xy, radians):
    x, y = xy
    xx = x * np.cos(radians) + y * np.sin(radians)
    yy = -x * np.sin(radians) + y * np.cos(radians)
    return xx, yy

def pixplot():
    # pixel_sizes = np.multiply(np.linspace(min_pixel_size,max_pixel_size,quantificationlevels),np.random.uniform(1-randomness_position,1+randomness_position,quantificationlevels))
    # pixel_sizes = pixel_sizes[0:val]
    # for n in pixel_sizes:
    
    # number of lines by the intensiy after quantization
    for n in range(val): #range(np.int(np.maximum(grad_mag*10,1)))

    # scale length by gradint magnitude 
        vert_x = np.multiply(np.array([-0.5, 0.5]),np.maximum(grad_mag*10,0.1))
        vert_x = np.multiply(vert_x,np.random.uniform(1-randomness_length,1+randomness_length,1))
        vert_y = np.array([0, 0])

        vert_y = vert_y + np.random.uniform(-randomness_vertex,randomness_vertex,2)
        vert_x = vert_x + np.random.uniform(-randomness_vertex,randomness_vertex,2)
        # rotate line by gradient direction so that line is perpendicular to gradients
        p1 = rotate_origin([vert_x[0],vert_y[0]],alpha)
        p2 = rotate_origin([vert_x[1],vert_y[1]],alpha)


        vert_x = np.array([p1[0],p2[0]]) + c + np.random.uniform(-randomness_position,+randomness_position)
        vert_y = np.array([p1[1],p2[1]]) + r + np.random.uniform(-randomness_position,+randomness_position)


        plt.plot(vert_x,vert_y,'k-')


plt.imshow(img_orig,alpha = 0.5, cmap = 'gray')

height, width = img.shape
for c in range(width):
    for r in range(height):
        val = int(img[r,c])
        grad_direction = np.array([img_dx[r,c],img_dy[r,c]]) 
        alpha = np.arctan2(grad_direction[0],grad_direction[1])
        grad_mag = grad_direction[0]*grad_direction[0] + grad_direction[1]*grad_direction[1]
        pixplot()
plt.axis('off') 
plt.gcf().patch.set_visible(False)      
plt.savefig("test.svg", bbox_inches ="tight")
plt.show()




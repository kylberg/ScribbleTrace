import numpy as np
from matplotlib import pyplot as plt
import skimage
from skimage.transform import resize
import random


img = skimage.data.load('P:/projects/AxiDraw/kylberg/wiggly_squares/pipe.jpg',as_gray=True)


output_width = 20.0
quantificationlevels = 7

randomness_vertex = 0.1 # 0 is no randomness, 0.1 is suitable.
randomness_position = 0.0 # 0.05
spiral_theta_resolution = 50
spiral_a = 0
spiral_b = 1
min_pixel_size = 0.1
max_pixel_size = 1.0

normalize_spirals = True

scale_factor = round(img.shape[0]/output_width)


img = resize(img,(round(img.shape[0]/scale_factor), round(img.shape[1]/scale_factor)),anti_aliasing=True)
img_orig = img
img = 1-img
img = np.round(np.multiply(img,quantificationlevels - 1)) + 0


def vertices_on_sin():

    n_vertices = round(spiral_theta_resolution)

    thetas = np.linspace(0,2*np.pi,n_vertices)
    y = np.sin(thetas)

    y = np.multiply(y,0.1)

    vertices = np.transpose(np.vstack((np.linspace(-0.5,0.5,n_vertices),y)))

    return vertices

def vertices_on_archimedian_spiral(theta_range):

    n_vertices = round(spiral_theta_resolution * theta_range/(2*np.pi))

    vertices = np.zeros([n_vertices,2])

    thetas = np.linspace(-theta_range,theta_range,n_vertices)

    for idx, theta in enumerate(thetas):
        if theta > 0:
            radius = spiral_a + spiral_b*theta
            vertices[idx,:] = np.array([radius*np.cos(theta), radius*np.sin(theta)])
        else:
            radius = spiral_a + spiral_b*(-theta)
            vertices[idx,:] = np.array([-radius*np.cos(-theta), -radius*np.sin(-theta)])
        
    

    return vertices



def pixplot():

        
    pixel_sizes = np.multiply(np.linspace(min_pixel_size,max_pixel_size,quantificationlevels),np.random.uniform(1-randomness_position,1+randomness_position,quantificationlevels))
    # if not small_first:
    #     pixel_sizes = pixel_sizes[::-1]

    # pixel_sizes = pixel_sizes[0:val]
    # for n in pixel_sizes:

        # out = circles(a, a, a*0.2, c=a, alpha=0.5, edgecolor='none')
        # circle1 = plt.Circle([c,r],radius=n/2,facecolor='none',edgecolor='k')
        # ax.add_artist(circle1)
        
    if val > 0:    
        theta_range  = val * np.pi
        vert = vertices_on_archimedian_spiral(theta_range)


        if normalize_spirals:
            vert_max = np.max(vert)
        else:
            vert_max = 22
        
        vert = np.multiply(np.divide(vert,vert_max),0.5)


        vert[:,0] = vert[:,0] + c 
        vert[:,1] = vert[:,1] + r

        
        plt.plot(np.array([np.min(vert[:,0]),c-0.5]),np.array([r    ,r]),'k')
        plt.plot(vert[:,0],vert[:,1],'k-')
        plt.plot(np.array([np.max(vert[:,0]),0.5+c]),np.array([r,r]),'k')
    else:
        vert = vertices_on_sin()

        # vert_max = np.max(vert)

        
        # vert = np.divide(vert,vert_max)

        vert[:,0] = vert[:,0] + c 
        vert[:,1] = vert[:,1] + r
        plt.plot(vert[:,0],vert[:,1],'k-')

        # plt.plot(np.array([c-0.5,c+0.5]),np.array([r,r]),'k')

        # vert_x = np.array([0, 0, 1, 1])
        # vert_y = np.array([0, 1, 1, 0])

        # vert_x = vert_x + np.random.uniform(-randomness_vertex,randomness_vertex,4)
        # vert_y = vert_y + np.random.uniform(-randomness_vertex,randomness_vertex,4)

        # vert_x = np.multiply(vert_x-0.5,n) + c
        # vert_y = np.multiply(vert_y-0.5,n) + r

        # vert_x = np.append(vert_x,vert_x[0])
        # vert_y = np.append(vert_y,vert_y[0])

        # plt.plot(vert_x,vert_y,'k-')


fig, ax = plt.subplots()
ax.imshow(img_orig,alpha = 0.5, cmap = 'gray')

height, width = img.shape
for c in range(width):
    for r in range(height):
        val = np.int(img[r,c])
        pixplot()
plt.show()




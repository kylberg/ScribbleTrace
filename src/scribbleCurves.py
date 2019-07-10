import numpy as np
from matplotlib import pyplot as plt
import skimage
from skimage.transform import resize
import random
from scipy import interpolate
import bezier

img = skimage.data.load('C:/Users/GustafK/source/repos/github/kylberg/ScribbleTrace/MagrittePipe.jpg',as_gray=True)


output_width = 40.0
quantificationlevels = 5
quantificationlevels_gradmag = 12
curve_max_steps = 4
curve_step_size = 2
scale_factor = round(img.shape[0]/output_width)

img = resize(img,(round(img.shape[0]/scale_factor), round(img.shape[1]/scale_factor)),anti_aliasing=True)
img_orig = img

img_dy = skimage.filters.sobel_h(img, mask=None)
img_dx = skimage.filters.sobel_v(img, mask=None)
grad_mag = np.multiply(img_dx,img_dx)+np.multiply(img_dy,img_dy)

grad_mag = np.divide(grad_mag,np.max(grad_mag))
grad_mag = np.round(np.multiply(grad_mag,quantificationlevels_gradmag - 1)) + 1

img = 1-img
img = np.round(np.multiply(img,quantificationlevels - 1)) + 1

randomness_angle = 0.01 # 0 is no randomness, 0.01 is suitable.
randomness_position = 0.5
# min_pixel_size = 0.1
# max_pixel_size = 0.9

randomness_length = 0.0

def normalize(v):
    norm=np.linalg.norm(v, ord=1)
    if norm==0:
        norm=np.finfo(v.dtype).eps
    return v/norm

# def rotate_origin(xy, radians):
#     x, y = xy
#     xx = x * np.cos(radians) + y * np.sin(radians)
#     yy = -x * np.sin(radians) + y * np.cos(radians)
#     return xx, yy

def pixplot():
    # pixel_sizes = np.multiply(np.linspace(min_pixel_size,max_pixel_size,quantificationlevels),np.random.uniform(1-randomness_position,1+randomness_position,quantificationlevels))
    # pixel_sizes = pixel_sizes[0:val]
    # for n in pixel_sizes:
    
    # number of lines by the intensiy after quantization
    for n in range(val): #range(np.int(np.maximum(grad_mag*10,1)))

        # scale length by gradint magnitude 


        vert_x = np.array([c]) + np.random.uniform(-randomness_position,+randomness_position)
        vert_y = np.array([r]) + np.random.uniform(-randomness_position,+randomness_position)

        loc_c = c
        loc_r = r

        step_number = np.int(np.minimum(curve_max_steps,local_grad_mag))
        step_length = np.minimum(curve_step_size,local_grad_mag/10)
        # one direction
        for i in range(step_number):
            alpha = np.arctan2(img_dy[loc_r,loc_c],img_dx[loc_r,loc_c]) - np.pi/2

            alpha = alpha*np.random.uniform(1-randomness_angle,1+randomness_angle,1)

            dx = step_length*np.cos(alpha)
            dy = step_length*np.sin(alpha)

            vert_x = np.append(vert_x, vert_x[-1] + dx)
            vert_y = np.append(vert_y, vert_y[-1] + dy)
            loc_c = np.int(np.round(vert_x[-1]))
            loc_r = np.int(np.round(vert_y[-1]))

            loc_c = np.maximum(np.minimum(loc_c,width-1),0)
            loc_r = np.maximum(np.minimum(loc_r,height-1),0)
        # oposit direction

        loc_c = c
        loc_r = r
        for i in range(step_number):
            alpha = np.arctan2(img_dy[loc_r,loc_c],img_dx[loc_r,loc_c]) - 3*np.pi/2

            alpha = alpha*np.random.uniform(1-randomness_angle,1+randomness_angle,1)

            dx = step_length*np.cos(alpha)
            dy = step_length*np.sin(alpha)

            vert_x = np.insert(vert_x, 0, vert_x[0] + dx)
            vert_y = np.insert(vert_y, 0, vert_y[0] + dy)
            loc_c = np.int(np.round(vert_x[0]))
            loc_r = np.int(np.round(vert_y[0]))

            loc_c = np.maximum(np.minimum(loc_c,width-1),0)
            loc_r = np.maximum(np.minimum(loc_r,height-1),0)

        # vert_x = np.multiply(np.array([-0.5, 0.5]),np.maximum(local_grad_mag*10,0.1))

        # vert_x = np.multiply(vert_x,np.random.uniform(1-randomness_length,1+randomness_length,1))

        # vert_y = np.array([0, 0])

        # vert_x = vert_x + np.random.uniform(-randomness_vertex,randomness_vertex,2)
        # vert_y = vert_y + np.random.uniform(-randomness_vertex,randomness_vertex,2)

        # rotate line by gradient direction so that line is perpendicular to gradients
        # p1 = rotate_origin([vert_x[0],vert_y[0]],alpha)
        # p2 = rotate_origin([vert_x[1],vert_y[1]],alpha)




        # vert_x = np.array([p1[0],p2[0]]) + c + np.random.uniform(-randomness_position,+randomness_position)
        # vert_y = np.array([p1[1],p2[1]]) + r + np.random.uniform(-randomness_position,+randomness_position)

        # looking back
        # grad_direction = np.array([img_dx[r,c],img_dy[r,c]]) 
        

        # if 0 < grad_direction[0]*grad_direction[0]+grad_direction[1]+grad_direction[1]:
        #     # grad_direction = normalize(grad_direction)
        #     vert_x = np.insert(vert_x,0,vert_x[0] + grad_direction[0])
        #     vert_x = np.append(vert_x, vert_x[-1] + grad_direction[0])

        #     vert_y = np.insert(vert_y,0,vert_y[0] + grad_direction[1])
        #     vert_y = np.append(vert_y, vert_y[-1] + grad_direction[1])
        
        curve1 = bezier.Curve(np.asfortranarray([vert_x,vert_y]), degree=2)
        # curve1.plot(num_pts=20)
        [vert_x,vert_y] = curve1.evaluate_multi(np.linspace(0,1,15))

        plt.plot(vert_x,vert_y,'k-')


img_angular = np.arctan2(img_dy,img_dx)
plt.imshow(grad_mag,alpha = 0.5, cmap = 'viridis')
plt.quiver(img_dx,img_dy)

height, width = img.shape
for c in range(width):
    for r in range(height):
        # if c==6 and r==21:
        val = np.int(img[r,c])
        local_grad_direction = np.array([img_dx[r,c],img_dy[r,c]]) 
        local_grad_direction = normalize(local_grad_direction)
        # alpha = np.arctan2(local_grad_direction[0],local_grad_direction[1])
        local_grad_mag = grad_mag[r,c]
        pixplot()
plt.show()




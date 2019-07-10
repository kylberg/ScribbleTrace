import numpy as np
from matplotlib import pyplot as plt
import skimage
from skimage.transform import resize
import random
from scipy import interpolate
import bezier
import scipy
from skimage.draw import line


img = skimage.data.load('C:/Users/GustafK/source/repos/github/kylberg/ScribbleTrace/MagrittePipe.jpg',as_gray=True)


output_width = 150.0
quantificationlevels = 20
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
plt.imshow(img,alpha = 0.5, cmap = 'gray')
# plt.quiver(img_dx,img_dy)

height, width = img.shape


# start point

initial_speed = 1
weight = 5
current_pos = np.array([random.uniform(0,width), random.uniform(0,height)])
current_vel = np.array([random.uniform(-initial_speed,initial_speed), random.uniform(-initial_speed,initial_speed)])

trace = np.ndarray(shape=(1,2), dtype=float)
trace[0] = current_pos
max_speed = 3
running = True

while running:

    c,r = np.round(current_pos).astype(int)
    
    dist = np.ones_like(img)
    if r<height and c<width and r>=0 and c>=0:
        dist[r,c] = 0
    else:
        dist[np.int(height/2),np.int(width/2)] = 0

    dist = scipy.ndimage.morphology.distance_transform_edt(dist)
    dist = np.max(dist)-dist
    dist_w = np.multiply(img,dist)
    pos_of_max = np.asarray(np.where(dist_w == np.amax(dist_w)))
    pos_of_max = pos_of_max[:,0]
    force = (pos_of_max - np.array([r,c])).astype(float)
    force_angle = np.arctan2(force[0],force[1])

    current_vel = current_vel + np.multiply(np.array([np.cos(force_angle),np.sin(force_angle)]),0.25)
    current_vel_norm = np.sqrt(current_vel[0]*current_vel[0] + current_vel[1]*current_vel[1])
    if current_vel_norm > 0 and current_vel_norm > max_speed:
        current_vel = np.multiply(np.divide(current_vel,current_vel_norm),max_speed)


    current_pos = current_pos + current_vel
    c,r = np.round(current_pos).astype(int)

    if r<height and c<width and r>=0 and c>=0:
        # img[r,c] = np.maximum(img[r,c] - weight,0)
        A = np.asarray(line(np.round(trace[-1,1]).astype(int), np.round(trace[-1,0]).astype(int), r, c))

        A = np.delete(A, np.where( np.bitwise_or( (A[:,0]<0), (A[:,0]>=height) ) )[0], 0)
        A = np.delete(A, np.where( np.bitwise_or( (A[:,1]<0), (A[:,1]>=width ) ) )[0], 0)
        rr, cc = A

        img[rr, cc] = np.maximum(img[rr,cc] - weight,0)




    print(np.sum(img))

    trace = np.append(trace,[current_pos],axis=0)



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




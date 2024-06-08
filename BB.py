import os

import numpy as np
import random as r
import cv2

from config import *
from utils.utilsybu import *

def xy2box(x, y, r=28):
    p1 = [x - r, y - r]
    p2 = [x + r, y + r]

    return p1, p2
# def xy2box(x, y, r=28):
#
#
#     p1 = [x - r, y - r]
#     p2 = [x + r, y + r]
#
#     return p1, p2
def target_process(bb, grid_number=8, is_torch=True):
    # boundingboxes: B NUM_OBJ 4(p1x,p1y,p2x,p2y)
    batch_size = bb.shape[0]
    number_box = bb.shape[1]
    himg, wimg = 256, 256
    target_batch = np.zeros((batch_size, grid_number, grid_number, 5))

    for i in range(batch_size):
        '''
        T Num xy
        '''

        BB = bb[2]
        # print('frame : ', i)

        for bi in range(number_box):
            p1= [bb[i, bi, 0], bb[i, bi, 1]]
            p2= [bb[i, bi, 2], bb[i, bi, 3]]
            bbox =bb[i,bi]

            # -------------bounding box to yolo target --------------------

            '''
            input: 
            center_x, center_y = (p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2
            grid_x = int(grid_number * float(center_x) / float(himg))
            grid_y = int(grid_number * float(center_y) / float(wimg))
            grid_center_x = int( (grid_x+0.5)/grid_number * himg)
            grid_center_y = int( (grid_y+0.5)/grid_number * wimg)
            center_point=np.array([grid_center_x, grid_center_y,grid_center_x,grid_center_y])
            target_point = (bbox-center_point) / np.array([wimg, himg, wimg, himg])
            
            
            
            '''

            # object 的中心点的绝对坐标
            center_x, center_y = (p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2

            # object 中心点所在的 grid
            grid_x = int(grid_number * float(center_x) / float(himg))
            grid_y = int(grid_number * float(center_y) / float(wimg))

            # object 中心点所在的 grid 的中心点 的坐标
            grid_center_x = int( (grid_x+0.5)/grid_number * himg)
            grid_center_y = int( (grid_y+0.5)/grid_number * wimg)

            center_point=np.array([grid_center_x, grid_center_y,grid_center_x,grid_center_y])

            # yolo model 的 target

            target_point = (bbox-center_point) / np.array([wimg, himg, wimg, himg])
            # ---------------------------------------------------------------------------------------

            # 置信度 confidence 设置为 1
            target_batch[i, grid_y, grid_x, 0] = 1.0
            # 相对于 anchor 的 位置
            target_batch[i, grid_y, grid_x, 1:5] = target_point
            # print(bb[i,bi])
            # print(grid_center_x,grid_center_y)
            # print(target_point)
            # print(target_batch[i, grid_y, grid_x])
            # print('----------------------------------')

    if is_torch:
        import torch
        target_batch = torch.from_numpy(target_batch)

    # pause()
    return target_batch

def target_process_(batch, grid_number=8, is_torch=True):
    # batch: B bb_num xy
    batch_size = batch.shape[0]
    target_batch = np.zeros((batch_size, grid_number, grid_number, 5))
    number_box = batch.shape[1]
    for i in range(batch_size):
        '''
        T Num xy
        '''

        BB = batch[2]
        # print('frame : ', i)

        for bi in range(number_box):
            center_x, center_y = batch[i, bi, 0], batch[i, bi, 1]
            p1, p2 = xy2box(center_x, center_y, r=19)

            bbox = np.array([p1[0], p1[1], p2[0], p2[1]])
            himg, wimg = 256, 256
            bbox = bbox / np.array([wimg, himg, wimg, himg])

            grid_x = int(grid_number * float(center_x) / float(himg))
            grid_y = int(grid_number * float(center_y) / float(wimg))

            target_batch[i, grid_y, grid_x, 0] = 1.0

    if is_torch:
        import torch
        target_batch = torch.from_numpy(target_batch)
    return target_batch

class Ball:
    def __init__(self, x, y, radius, dx, dy):
        self.x = x
        self.y = y
        self.dx = dx
        self.dy = dy
        self.radius = radius
        self.hit_h_wall = 0
        self.hit_v_wall = 0
        self.last_collision = -1


def distance(obj1, obj2):
    return ((obj1.x - obj2.x) ** 2 + (obj1.y - obj2.y) ** 2) ** 0.5


def norm(vector):
    return (vector[0] ** 2 + vector[1] ** 2) ** 0.5


def plot_ball(ball, map):
    X, Y = np.meshgrid(np.arange(WINDOW_WIDTH), np.arange(WINDOW_HEIGHT))
    dist = (X - ball.x) ** 2 + (Y - ball.y) ** 2
    map[np.where(dist < ball.radius ** 2)] = 255
    return map


def to_video(map, filename):
    print('test')
    # image_list = []
    # for i in range(times):
    #     image_list.append(Image.fromarray(map[i, :, :].astype('uint8')).convert('L'))
    filename = filename + '.mp4'
    size = [WINDOW_WIDTH, WINDOW_HEIGHT]
    fps = 5
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(filename, fourcc, fps, size, False)
    for i in range(times):
        # out.write(image_list[i])
        out.write(map[i, :, :])
        # print('test')
    out.release()


def create_objects(NUM_OBJ, default_radius=25, speed_width=[-4, 4]):
    objects = []

    rads=np.zeros((1,NUM_OBJ,2))
    radius_=[]
    speeds_=[]
    for _ in range(NUM_OBJ):
        radius = r.randint(default_radius[0], default_radius[1])
        radius_.append(radius)
        x = r.randint(radius, WINDOW_WIDTH - radius)
        y = r.randint(radius, WINDOW_HEIGHT - radius)
        if len(objects) != 0:
            state = True
            cur_length = len(objects)
            actual_dis = np.zeros(cur_length)
            dis = np.zeros(cur_length)  # dis should less than acutal_dis to avoid overlap
            while state:
                state = False
                for i in range(cur_length):
                    obj = objects[i]
                    actual_dis[i] = (obj.x - x) ** 2 + (obj.y - y) ** 2
                    dis[i] = (radius + obj.radius) ** 2
                    if sum(actual_dis < dis) > 0:
                        state = True
                        x = r.randint(radius, WINDOW_WIDTH - radius)
                        y = r.randint(radius, WINDOW_HEIGHT - radius)
        #dx = r.randint(speed_width[0], speed_width[1])
        #dy = r.randint(speed_width[0], speed_width[1])
        dx = 0
        dy = 0
        # while dx==0:
        #     dx = r.randint(speed_width[0], speed_width[1])
        # while dy==0:
        #     dy = r.randint(speed_width[0], speed_width[1])

        while dx==0:
            dx = r.randint(speed_width[0], speed_width[1])
        while dy==0:
            dy = r.randint(speed_width[0], speed_width[1])
        speeds_.append([dx, dy])

        objects.append(Ball(x, y, radius, dx, dy))
        # rads.append([x,y])
        rads[0,_,0]=x
        rads[0,_,1]=y

    return objects,rads,radius_,speeds_


def animation():
    objects,rads_init,radius_,speeds_ = create_objects(NUM_OBJ, RADIUS_RANGE, SPEED_RANGE)

    print(radius_)
    print(speeds_)
    print(rads_init)
    centers=np.zeros((times,NUM_OBJ,2))
    boundingboxes=np.zeros((times,NUM_OBJ,4),dtype=int)


    # yolo_target= np.zeros((times, NUM_OBJ, 2))

    # centers[0]=rads_init
    # print(rads)
    # 1 I 2
    # T I 2
    # input('')
    map = np.zeros([times, WINDOW_WIDTH, WINDOW_HEIGHT], dtype=np.uint8)
    for obj in objects:
        map[0, :, :] = plot_ball(obj, map[0, :, :])

    collision_ = np.zeros([times], dtype=bool)
    # collision_=[]
    for i in range(1, times):

        ball_index=0
        collision = False

        for obj in objects:  # Moving each object
            obj.x += obj.dx
            obj.y += obj.dy
            # ball center
            centers[i,ball_index]=[obj.x,obj.y]
            # bounding boxes
            boundingboxes[i, ball_index] = [obj.x - obj.radius, obj.y - obj.radius, obj.x + obj.radius, obj.y + obj.radius]
            ball_index=ball_index+1
            # print('obj: ({},{})'.format(obj.dx, obj.dy))
            map[i, :, :] = plot_ball(obj, map[i, :, :])
            # check vertical wall collision
            if obj.x <= obj.radius:
                obj.dx = abs(obj.dx)
                collision=True
            elif obj.x >= WINDOW_WIDTH - obj.radius:
                obj.dx = - abs(obj.dx)
                collision=True

            # check horizontal wall collision
            if obj.y <= obj.radius:
                obj.dy = abs(obj.dy)
                collision=True
            elif obj.y >= WINDOW_HEIGHT - obj.radius:
                obj.dy = - abs(obj.dy)
                collision=True

        collision_[i]=collision
        for index in range(len(objects)):
            for j in range(index, len(objects)):
                obj1 = objects[index]
                obj2 = objects[j]
                dist = distance(obj1, obj2)
                if dist <= (obj1.radius + obj2.radius):
                    obj1.dx, obj2.dx = obj2.dx, obj1.dx
                    obj1.dy, obj2.dy = obj2.dy, obj1.dy

    yolo_target=target_process(boundingboxes,grid_number=8,is_torch=False)

    # yolo_target=target_process(centers,grid_number=8,is_torch=False)
    # CC0= yolo_target[:,:,:,0]
    # grids=enlarge(CC0,32)
    # grids=gray2rgb(grids).astype(np.uint8)*255
    #
    # bb=centers.astype(np.int32)
    # for i in range(times):
    #     for j in range(NUM_OBJ):
    #         p1,p2=xy2box(centers[i,j,0],centers[i,j,1],r=radius_[j])
    #         boundingboxes[i,j]=[p1[0],p1[1],p2[0],p2[1]]
    # for i in range(CC0.shape[0]):
    #     # print_info(grids[i])
    #     # print_info(xrgb[i])
    #     # B bb_num xy
    #     left=grids[i]
    #     right=map[i]
    #     for j in range(bb.shape[1]):
    #         print('--------------')
    #         p1,p2=xy2box(bb[i,j,0],bb[i,j,1],r=radius_[j])
    #         print(p1,p2)
    #
    #         p11,p22=[boundingboxes[i,j,0],boundingboxes[i,j,1]],[boundingboxes[i,j,2],boundingboxes[i,j,3]]
    #         print(p11,p22)
    #         # pause()
    #         left= cv2.circle(left,center=(bb[i,j,0],bb[i,j,1]),radius=2,color=(0,0,255),thickness=5)
    #         # right=cv2.circle(right,center=(bb[i,j,0],bb[i,j,1]),radius=2,color=(0,0,255),thickness=5)
    #         # left=cv2.rectangle(left,p1,p2,color=[0,255,0],thickness=1)
    #         left=cv2.rectangle(left,p11,p22,color=[0,255,255],thickness=1)
    #
    #     left=enlarge(left,ratio=2,C_first=False)
    #     # right=enlarge(right,ratio=2,C_first=False)
    #     # ssshow=np.concatenate([left,right],axis=1)
    #     cv2.imshow('left', left)
    #     cv2.waitKey(1000)

    return map,centers,boundingboxes,radius_,speeds_,collision_,yolo_target


import argparse
import time
import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import numpy as np
from torch.utils.data import Dataset, DataLoader
# from skimage.transform import resize
# from event_utils import make_2channel_events
# import event_utils
import datetime
import os
import sys
import numpy
numpy.set_printoptions(threshold=sys.maxsize)
import cv2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('torch availability',torch.cuda.is_available())

# device = torch.device("cpu")

def pause(showtxt='----------press enter to continue----------'):
    input(showtxt)
    return


def view_as_batch(x):
    # 等于 torch.squeeze(0) 真2
    shape = list(x.shape)
    shape.insert(0, 1)
    tuple(shape)

    return x.reshape(shape)


def print_shape(tensor):
    print('shape: ', tensor.shape)
    return

def save_txt(txt,save_path):
    '''

    :param txt: d
    :param save_path:
    :return:
    '''


    # lines = ['Readme', 'How to write text files in Python']
    with open(save_path, 'w') as f:
        for line in txt:
            print('------',line)
            print('nbvcmncmmmmmmmmmmmmmmmmm')
            # layer,active_num=line
            # f.write(layer,' : ',active_num)
            # f.write('\n')

    return


def print_info(array, name='this array'):
    print('------ info of ', name, '------')
    print('type:', type(array))
    print('shape:', array.shape)
    print('dtype:', array.dtype)
    if type(array) == np.ndarray:
        print('max:', np.max(array))
        print('min:', np.min(array))
    elif type(array) == torch.Tensor:
        print('max:', torch.max(array))
        print('min:', torch.min(array))
        print('device:', array.device)

    print('------print info end---------')
    return


def empty_vram():
    torch.cuda.empty_cache()
    return
def vram(before_sentence=''):

    used=torch.cuda.memory_allocated()/1024/1024/1024
    print(before_sentence+' VRAM used',used, 'GB')
    return used

def Merge(dicts):

    res={}
    for dict in dicts:
        res = {**res, **dict}
    # res = {**dict1, **dict2}
    return res

def index2name(index, seg_per_v=47):
    '''

    index 变成 bouncing ball 的 file name
    eg. 012_23.npy

    :return:
    '''

    seg_per_v = 47
    zheng = index // seg_per_v
    yu = index % seg_per_v

    name = str(zheng).zfill(3) + '_' + str(yu).zfill(2) + '.npy'

    return name
def enlarge(img: np.ndarray,ratio:int,C_first=True):

    '''
    img B C H W

    C_first:
         True:
         C H W
         False:
         H W C
    '''
    if C_first:
        if len(img.shape)==3:

            C,H,W=img.shape
            Hr=H*ratio
            Wr=W*ratio
            resized=np.zeros((C,Hr,Wr),dtype=img.dtype)
            for i in range(ratio):
                for j in range(ratio):
                    resized[:, i:Wr:ratio, j:Hr:ratio] = img[:, 0:H, 0:W]

        elif len(img.shape)== 4:
            B, C, H, W = img.shape

            resized = np.zeros((B, C, H*ratio,W*ratio), dtype=img.dtype)
            for i in range(ratio):
                for j in range(ratio):
                    resized[:, :, i:-1:ratio, j:-1:ratio] = img[:, :, 0:H, 0:W]

        elif len(img.shape)== 2:
            H, W = img.shape

            resized = np.zeros((H*ratio,W*ratio), dtype=img.dtype)
            for i in range(ratio):
                for j in range(ratio):
                    resized[i:-1:ratio, j:-1:ratio] = img[0:H, 0:W]
        else:
            return None

    else:
        if len(img.shape) == 3:

            H, W, C = img.shape
            Hr = H * ratio
            Wr = W * ratio
            resized = np.zeros((Hr, Wr, C), dtype=img.dtype)
            for i in range(ratio):
                for j in range(ratio):

                    # print(resized.shape)
                    # print(img.shape)
                    resized[i:Wr:ratio, j:Hr:ratio, :] = img[0:H, 0:W, :]

        elif len(img.shape) == 4:
            B, H, W, C = img.shape

            resized = np.zeros((B, H * ratio, W * ratio, C), dtype=img.dtype)
            for i in range(ratio):
                for j in range(ratio):
                    resized[:, i:-1:ratio, j:-1:ratio, :] = img[:,0:H, 0:W, :]

        elif len(img.shape) == 2:
            H, W = img.shape

            resized = np.zeros((H * ratio, W * ratio), dtype=img.dtype)
            for i in range(ratio):
                for j in range(ratio):
                    resized[i:-1:ratio, j:-1:ratio] = img[0:H, 0:W]
        else:
            return None
    return resized

def gray2rgb(img):

    '''
    img B H W
    '''
    if len(img.shape)==3:
        B,H,W=img.shape

        rgb=np.expand_dims(img,axis=3)
        rgb=np.concatenate([rgb,rgb,rgb],axis=3)

    elif len(img.shape)== 2:
        H,W=img.shape

        rgb=np.expand_dims(img,axis=2)
        rgb=np.concatenate([rgb,rgb,rgb],axis=2)
    else:
        return None

    return rgb



def index2name33(index, seg_per_v=47):
    '''

    index 变成 bouncing ball 的 file name

    :return:
    '''
    # print(index)
    seg_per_v = seg_per_v
    zheng = index // seg_per_v
    yu = index % seg_per_v
    # print(index & seg_per_v,'-','index & seg_per_v')
    name = str(zheng).zfill(3) + '_' + str(yu).zfill(3) + '.npy'

    return name


def imshow_seq(seq, wait=200):
    # seq : (index ,h, w)
    # cv2 show 一个序列的frame

    if type(seq) == torch.Tensor:
        show = seq.numpy()
    elif type(seq) == np.ndarray:
        show = seq
    else:
        print('invalid show')
        return
    # print(show.dtype)

    if show.dtype == 'int32':
        show = show.astype(np.uint8)

    if show.dtype == np.uint8 and np.max(show) <= 1:
        show = show * 255

    for i in range(show.shape[0]):
        print(show[i].shape)
        # print(show[i])
        cv2.imshow('frame  ' + str(i), show[i])
        cv2.waitKey(wait)

    return


def imshow_seq_rgb(seq, wait=200):
    # seq : (index ,h, w)
    # cv2 show 一个序列的frame

    if type(seq) == torch.Tensor:
        show = seq.numpy()
    elif type(seq) == np.ndarray:
        show = seq
    else:
        print('invalid show')
        return
    # print(show.dtype)

    if show.dtype == 'int32':
        show = show.astype(np.uint8)

    if show.dtype == np.uint8 and np.max(show) <= 1:
        show = show * 255

    for i in range(show.shape[0]):
        # print(show[i].shape)
        # print(show[i])
        # cv2.imshow('frame  ' + str(i), show[i].transpose(1, 2, 0))
        cv2.imshow('frame  ' + str(i), show[i])

        cv2.waitKey(wait)
        cv2.destroyWindow('frame  ' + str(i))

    return

# def add_noise(x,thresh):
#     '''
#
#     numpy 版本
#     :param x:
#     :param thresh:
#     :return:
#     '''
#
#     from event_utils import removeBadPixel
#     T,C,H,W=x.shape # T 2 H W
#     # thresh=0.985
#     noise=np.random.rand(T,C,H,W)
#     noise=np.where(noise>thresh,1.0,0.0)
#
#     noised_x=x+noise
#     noised_x=np.where(noised_x>0,1.0,0.0)
#
#     noised_x=removeBadPixel(noised_x.transpose(0,2,3,1)).transpose(0,3,1,2)
#
#     return noised_x


def add_noise(x, thresh):
    '''

    torch 版本
    :param x:
    :param thresh:
    :return:
    '''
    from utils.event_utils import removeBadPixel

    if type(x) == torch.Tensor:
        T,C,H,W = x.shape  # T 2 H W
        # thresh=0.985
        noise = torch.rand(T, C, H, W).to(device)
        noise = torch.where(noise > thresh, 1.0, 0.0)

        noised_x = x + noise
        noised_x = torch.where(noised_x > 0, 1.0, 0.0)

        noised_x = removeBadPixel(noised_x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
    elif type(x) == np.ndarray:
        T,C,H,W = x.shape  # T 2 H W
        # thresh=0.985
        noise = np.random.rand(T, C, H, W)
        noise = np.where(noise > thresh, 1.0, 0.0)

        noised_x = x + noise
        noised_x = np.where(noised_x > 0, 1.0, 0.0)

        noised_x = removeBadPixel(noised_x.transpose(0, 2, 3, 1)).transpose(0, 3, 1, 2)


    return noised_x,noise

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __repr__(self):
        return '{:.3f} ({:.3f})'.format(self.val, self.avg)


def np2v(seq, outp='output.mp4', fps=24.0, is_color=True):
    '''

    gray image input
    (T,H,W)

    rgb image input
    (T, H, W, 3)


    '''
    # print_info(seq)
    # size = (seq.shape[1],seq.shape[2])
    # size = (seq.shape[2], seq.shape[1])

    T,H,W,C=seq.shape
    # size=(H,W)
    size=(W,H)

    # print(size)
    # T = seq.shape[0]
    # somehow 这个格式有时候有用有时候没用 很奇怪
    fourcc = cv2.VideoWriter_fourcc(*'MP42')
    # fourcc = cv2.VideoWriter_fourcc(*'MJPG')

    '''
        gray: isColor=False
        rgb:  isColor=True
    '''
    print(size)
    out = cv2.VideoWriter(outp, fourcc, fps, size, isColor=is_color)

    for i in range(T):
        # (256, 256, 3) np.uint8
        # print_info(seq[i],'seq[i]')
        out.write(seq[i])
        # out.write(np.random.randint(0,255, (H,W,3), dtype = np.uint8))
    out.release()
    print('saved video in ',out)
    return


def show(a):
    '''

    show a np array
    :param a:
    :return:
    '''

    cv2.imshow('this img', a)
    cv2.waitKey(0)

    return


class Events(object):
    def __init__(self, num_events, width=346, height=260):
        self.data = np.rec.array(None,
                                 dtype=[('x', np.uint16), ('y', np.uint16), ('p', np.bool_), ('ts', np.float64)],
                                 shape=(num_events))
        self.width = width
        self.height = height

    def generate_fimage(self, input_event=0, image_raw_event_inds_temp=0, split=1):
        '''
        split: 每个 gray frame 对应多少个 event frame
            frame_num = gray_num*split

        '''

        # time stamp
        start = input_event[0, 2]
        end = input_event[-1, 2]

        # time length 这一段视频的总时长
        time_all = end - start

        # 一共有多少帧
        gray_num = image_raw_event_inds_temp.shape[0]
        frame_num = gray_num * split
        # print('frame',time_all)
        # print('frame',frame_num)

        # 每一帧分配有多少时长
        interval = time_all / frame_num
        events_num = input_event.shape[0]

        # print("start time: " , start)
        # print("end time: " , end)
        # print("time all: ",time_all)
        # print("frame num: ",frame_num)
        # print("interval: ",interval)
        # print("events num: ",events_num)
        # print('input_event',input_event.shape)

        frame_indexes = input_event[:, 2].reshape([events_num])
        frame_indexes = frame_indexes - start - 1e-5
        frame_indexes = frame_indexes / interval

        frame_indexes = frame_indexes.astype(np.int)
        x = input_event[:, 0].astype(np.int)
        y = input_event[:, 1].astype(np.int)
        p = input_event[:, 3].astype(np.int)

        # 0 neg event; 1 pos event
        p = np.where(p == 1, 1, 0)

        event_frames = np.zeros([frame_num, 346, 260, 2], dtype=np.uint8)

        # 根据 index 赋值
        event_frames[frame_indexes, x, y, p] = 255

        # event_frames
        # print('event_frames',event_frames.shape)

        event_frames = event_frames.reshape(gray_num, split, 346, 260, 2)
        # print('event_frames',event_frames.shape)

        event_frames = event_frames.transpose(0, 1, 3, 2, 4)
        # print('event_frames',event_frames.shape)

        # for i in range(event_frames.shape[0]):
        #     print(i, '/', 2655)
        #     cv2.imshow('dasdsa',
        #                np.concatenate([event_frames[i,1, :, :, 0],
        #                                event_frames[i,1, :, :, 1]], axis=1))
        #     cv2.waitKey(20)

        return event_frames


def events2rgb(events):
    '''

    input   :
    events       : (f/B, 2, w, h) np.uint8

    pos: red
    neg: blue

    black background

    :return:
    '''

    if type(events) == np.ndarray:
        pass
    elif type(events) == torch.Tensor:
        events=events.cpu().numpy()

    from event_utils import removeBadPixel
    T,C,H,W = events.shape

    if np.max(events)>1:
        events=events/255

    # rgb[:,0,:,:]=events[:,0,:,:]
    # rgb[:,2,:,:]=events[:,1,:,:]
    events=events.transpose((0,2,3,1))
    events=removeBadPixel(events)
    events=events.transpose((0,3,1,2))
    # print(events.shape)

    # blank=np.zeros([frame_num, 1, h, w],dtype=np.uint8)
    rgb = np.zeros([T, 3, H, W], dtype=np.uint8)

    rgb[:, 0, :, :] = events[:, 0, :, :]
    rgb[:, 2, :, :] = events[:, 1, :, :]

    if np.max(rgb)<255:
        rgb=rgb*255

    return rgb


def shrink_event(events, ratio=2):

    '''

    :param events:
    :param ratio:
    :return:
    '''

    if len(events.shape)==3:

        C, H, W = events.shape

        H1=int(H*ratio)
        W1=int(W*ratio)

        events_resized = resize(events, (140, 54))
    elif len(events.shape)==4:

        T, C, H, W = events.shape

        events_resized=events[:,:,0:H:ratio,0:W:ratio]
        events_resized=np.zeros_like(events_resized)
        for i in range(ratio):
            for j in range(ratio):
                events_resized=events_resized+events[:,:,i:H:ratio,j:W:ratio]

        events_resized=events_resized/(ratio*ratio)
        events_resized=event_utils.make_2channel_events((events_resized))

    elif len(events.shape) == 5:

        B, T, C, H, W = events.shape

        H1 = int(H * ratio)
        W1 = int(W * ratio)

        events_resized = resize(events, (H1, W1))
    else:
        print('Invalid Shape')
        return events

    return events_resized


def events3_2_rgb(events):
    '''

    input   :
    events       : (f/B, 3, w, h) np.float

    pos: red
    neg: blue

    cv2 好像是 BGR 不是 RGB

    # 0: no event -> 0
    # 1: p        -> blue
    # 2: n        -> red

    :return:
    '''
    print_info(events, 'events')
    frame_num, p, h, w = events.shape

    ggg = np.zeros([3, frame_num * h * w], dtype=np.uint8)
    kkk = np.zeros([frame_num, 3, h, w], dtype=np.uint8)

    xxx = events.transpose(1, 0, 2, 3)
    xxx = xxx.reshape(3, -1)

    ggg[np.argmax(xxx, axis=0), np.arange(frame_num * h * w)] = 255
    ggg = ggg.reshape(3, frame_num, h, w)
    ggg = ggg.transpose(1, 0, 2, 3)

    kkk[:, 0, :, :] = ggg[:, 1, :, :]
    kkk[:, 2, :, :] = ggg[:, 2, :, :]

    return kkk


def torch2np(tensor):
    """
    cuda 上的 tensor 移到 numpy()
    """
    return tensor.cpu().detach().numpy()


def event3to2(e3):
    '''

    ------ info of  pred ------
    type: <class 'numpy.ndarray'>
    shape: (1, 3, 256, 256)
    dtype: float32
    max: 1.0
    min: 4.827225e-12

    # 0: no event -> 0
    # 1: p        -> blue
    # 2: n        -> red

    (3,256,256) -> (2,256,256)
    :return:
    '''

    shape = e3.shape

    if len(shape) == 3:
        # print('length is 3')
        return e3[1:3, :, :]
    elif len(shape) == 4:
        # print('length is 4')

        return e3[:, 1:3, :, :]
    else:
        # print('length is',len(shape))
        return

    # return e3[1:3,:,:]


def np2events(v):
    '''

    :param v:   (480,256,256)
    :return:    (480, 2, 256, 256)
    '''
    frame_num, h, w = v.shape
    copy = np.array(v, dtype=np.float)
    blank_frame = np.zeros([1, h, w], dtype=np.float)
    # first_frame=v[0]
    # v[0]=0
    # cv2.imshow('fff',v[0]*255)
    # cv2.waitKey(0)
    res = np.concatenate([blank_frame, copy], axis=0)
    res = res[0:frame_num]

    # 后一帧 减去 前一帧
    res = v - res
    res[0] = 0
    # print_info(res,'res')

    events = np.zeros([2, frame_num, h, w])

    pos_event = np.where(res > 0, 1, 0).astype(np.uint8)
    neg_event = np.where(res < 0, 1, 0).astype(np.uint8)
    events[0] = pos_event
    events[1] = neg_event
    events = events.transpose(1, 0, 2, 3)
    return events


def np2events_batch(v):
    '''

    :param v:   (B, T, H, W)
    :return:    (B, T, 2, H, W)
    '''
    B, T, H, W = v.shape
    copy = np.array(v, dtype=np.float)
    blank_frame = np.zeros([B, 1, H, W], dtype=np.float)
    # print_info(blank_frame,'blank_frame')
    # print_info(copy,'copy')

    # first_frame=v[0]
    # v[0]=0
    # cv2.imshow('fff',v[0]*255)
    # cv2.waitKey(0)
    res = np.concatenate([blank_frame, copy], axis=1)
    res = res[:, 0:T]

    # 后一帧 减去 前一帧
    res = v - res
    res[:, 0] = 0
    # print_info(res,'res')

    events = np.zeros([B, 2, T, H, W])

    pos_event = np.where(res > 0, 1, 0).astype(np.uint8)
    neg_event = np.where(res < 0, 1, 0).astype(np.uint8)
    events[:, 0] = pos_event
    events[:, 1] = neg_event
    events = events.transpose(0, 2, 1, 3, 4)
    return events


def gen_events_from_video(v, save=''):
    '''

    input   :
    v       : (f, w, h) np.uint8

    :return:
    '''
    print_info(v, 'v')

    events = np2events(v)
    print_info(events, 'events')

    rgb = events2rgb(events)
    rgb = rgb.transpose(0, 2, 3, 1)
    print_info(rgb, 'rgb')

    # imshow_seq_rgb(rgb)
    np2v(rgb * 255, out='rgb_event.mp4', is_color=True)

    return


def get_input_representation(index=150, load_path='G:/Cynthia/Q/dataset/visevent/train_numpy/event/2.npy'):
    # self.traindir_gray = 'G:/Cynthia/Q/dataset/visevent/train_numpy/gray'

    load_path = 'G:/Cynthia/Q/dataset/mvsec/full/processed/indoor/0/count_data/' + str(index) + '.npy'
    load_path_gary = 'G:/Cynthia/Q/dataset/mvsec/full/processed/indoor/0/gray_data/' + str(index) + '.npy'

    load_path_next = 'G:/Cynthia/Q/dataset/mvsec/full/processed/indoor/0/count_data/' + str(index + 1) + '.npy'
    load_path_gary_next = 'G:/Cynthia/Q/dataset/mvsec/full/processed/indoor/0/gray_data/' + str(index + 1) + '.npy'
    # MVSEC     x shape (2, 260, 346, 10)
    # VisEvent  x shape (l,10,260,346,2)
    x = np.load(load_path)
    x_gray = np.load(load_path_gary)

    x = torch.from_numpy(x)
    print('x shape ', x.shape)
    print('x shape ', x_gray.shape)

    next_representation = np.load(load_path_gary_next)

    # x shape (l,2,260,346,10)

    # visevent
    # x = x.permute(0, 4, 2, 3, 1)

    # MVSEC
    # x = x.permute(3,1,2,0)
    # ([10, 260, 346, 2])
    # print('x shape ',x.shape)

    # visevent ([105, 2, 260, 346, 10])
    # print('x shape ',x.shape)

    # x = x.transpose(0, 4, 2, 3, 1)

    # event list 中每一个 元素是一个numpy array(l,10,260,346,2) l是这个data被分为了多少个frame
    # gray list每个元素 (l,260,346)

    xoff = 45
    yoff = 2
    # input_representation = torch.zeros(x.shape[0], 4, image_resize, image_resize,
    #                                    5).float()

    input_representation = torch.zeros(1, 4, image_resize, image_resize, 5).float()
    input_representation[:, 0, :, :, :] = x[0, yoff:-yoff, xoff:-xoff, 0:5]
    input_representation[:, 1, :, :, :] = x[1, yoff:-yoff, xoff:-xoff, 0:5]
    input_representation[:, 2, :, :, :] = x[0, yoff:-yoff, xoff:-xoff, 5:10]
    input_representation[:, 3, :, :, :] = x[1, yoff:-yoff, xoff:-xoff, 5:10]

    input_representation = input_representation.to(device)
    # x=torch.ones([11,4,128,128,1],dtype=torch.float)
    # input_representation=torch.from_numpy(input_representation)
    # print('input_representation',input_representation.shape)
    # input_representation=torch.ones([1,4,256,256,5],dtype=torch.float)

    # 检查一下灰度图是不是对的
    # for j in range(self.length):
    #     cv.imshow('dd', self.grays[j, 1])
    #     cv.waitKey(100)

    # cv2.imshow('x_gray', x_gray)
    # cv2.waitKey(0)

    # 检查一下event 是不是对的
    # for j in range(10):
    #     show=input_representation[0,j,:,:,0].cpu().numpy()
    #     cv2.imshow('spikes', show)
    #     cv2.waitKey(800)

    return input_representation, x_gray, next_representation

def plot_list(y_list,legend_list):

    import matplotlib.pyplot as plt
    fig,ax=plt.subplots()
    plt.xlabel("time stamp")
    plt.ylabel("Average IoU")
    xxx=np.arange(len(y_list[0]))
    plt.ylim((0.0, 1.0))
    for i in range(len(y_list)):
        ax.scatter(xxx,y_list[i],s=3,label=legend_list[i])
    # ax.scatter(xxx,iou_region2,label=' region size 2')
    # ax.scatter(xxx,iou_region4,label=' region size 4')
    ax.legend()
    # plt.savefig('books_read.png')
    plt.show()

    return

def plot_loss(loss, saveplt='losses.png'):
    if type(loss[0]) == torch.Tensor:
        loss = torch.Tensor(loss)

        # print(loss.device)
        if loss.device == torch.device("cuda:0"):
            loss = loss.cpu().numpy()
        else:
            loss = loss.numpy()

    import matplotlib.pyplot as plt
    plt.clf()
    x_points = np.arange(len(loss))
    y_points = np.array(loss)

    plt.plot(x_points, y_points)
    plt.savefig(saveplt)

    plt.show()
    return

def plot_loss3(losses, saveplt='losses.png'):
    ys=[]
    for loss in losses:
        if type(loss[0]) == torch.Tensor:
            loss = torch.Tensor(loss)

            # print(loss.device)
            if loss.device == torch.device("cuda:0"):
                loss = loss.cpu().numpy()
            else:
                loss = loss.numpy()
        ys.append(loss)

    import matplotlib.pyplot as plt
    plt.clf()
    x_points = np.arange(len(loss))
    y_1 = np.array(ys[0])
    y_2 = np.array(ys[1])
    y_3 = np.array(ys[2])

    p1,=plt.plot(x_points, y_1)
    p2,=plt.plot(x_points, y_2)
    p3,=plt.plot(x_points, y_3)
    plt.legend([p1,p2,p3],['combined','pred','attention'])
    plt.savefig(saveplt)

    plt.show()
    return

def scatter_loss(loss, saveplt='losses.png'):
    if type(loss[0]) == torch.Tensor:
        loss = torch.Tensor(loss)

        # print(loss.device)
        if loss.device == torch.device("cuda:0"):
            loss = loss.cpu().numpy()
        else:
            loss = loss.numpy()

    import matplotlib.pyplot as plt

    x_points = np.arange(len(loss))
    y_points = np.array(loss)

    plt.scatter(x_points, y_points)
    plt.savefig(saveplt)

    plt.show()
    return
def crop_256(before):
    '''
    (260,346) -> (256,256)
    '''

    return before[2:-2, 45:-45]


def addmargin(img, margin=1,color=(255,255,255)):
    '''

    img (3,w,h)

    given a rgb image, add margin around it
    :return:
    '''
    # print_info(img,'img')
    C, W, H = img.shape
    d = img.dtype

    gray = color[0]
    top = np.ones([C, W, margin], dtype=d) * gray
    bot = np.ones([C, W, margin], dtype=d) * gray
    left = np.ones([C, margin, H + margin * 2], dtype=d) * gray
    rght = np.ones([C, margin, H + margin * 2], dtype=d) * gray

    margined = np.concatenate([top, img, bot], axis=2)
    # print_info(margined,'margined')

    margined = np.concatenate([left, margined, rght], axis=1)
    # print_info(margined,'margined')

    return margined


def addmargin_v(v, margin=1):
    '''

    img (T, W, H, 3)

    given a rgb video, add margin around it
    :return:
    '''
    # print_info(v,'v')
    T, W, H, C = v.shape
    d = v.dtype
    # print(T, W,margin, C,d)
    gray = 128
    top = np.ones([T, W, margin, C], dtype=d) * gray
    bot = np.ones([T, W, margin, C], dtype=d) * gray
    left = np.ones([T, margin, H + margin * 2, C], dtype=d) * gray
    rght = np.ones([T, margin, H + margin * 2, C], dtype=d) * gray
    # print_info(top,'top')

    margined = np.concatenate([top, v, bot], axis=2)
    # print_info(margined,'margined')

    margined = np.concatenate([left, margined, rght], axis=1)
    # print_info(margined,'margined')
    # pause()

    return margined

# def cat_two_image()

def show10event(event):
    '''

    :param event: (10,2,256,256)
    :return:
    '''
    rgb_pred = events2rgb(event) * 255

    for i in range(2):

        for j in range(5):

            if j == 0:
                line = addmargin(rgb_pred[0])

            else:
                line = np.concatenate([line, addmargin(rgb_pred[j])], axis=2)

        if i == 0:
            whole_pack = line
        else:
            whole_pack = np.concatenate([whole_pack, line], axis=1)

    #  (3, W, H)
    return whole_pack


def show10event2(event):
    '''

    :param event: (10,2,256,256)
    :return:
    '''
    rgb_pred = events2rgb(event) * 255

    for i in range(10):

        if i == 0:
            whole_pack = addmargin(rgb_pred[0])
        else:
            whole_pack = np.concatenate([whole_pack,
                                         addmargin(rgb_pred[i])], axis=2)

    #  (3, W, H)
    return whole_pack


def event2video(seq, out='output.mp4', fps=24, is_color=True):
    '''
    event   shape: (480, 2, 256, 256)
    rgb     shape: (480, 3, 256, 256)
    '''
    rgbnp = events2rgb(seq)
    # print_info(rgbnp,'rgbnp')
    # shape: (480, 3, 256, 256)

    size = (seq.shape[2], seq.shape[3])
    duration = rgbnp.shape[0]
    fps = fps
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    '''
        gray: isColor=False
        rgb:  isColor=True
    '''
    out = cv2.VideoWriter(out, fourcc, fps, size, isColor=is_color)

    for i in range(duration):
        # np.uint8
        # (h, w, 3)
        data = rgbnp[i].transpose(1, 2, 0) * 255
        # print_info(data,'data')
        # input('-----')
        out.write(data)
    out.release()
    return rgbnp


def two_event2v(seq1, seq2, out='combined.mp4'):
    rgbnp1 = events2rgb(seq1)
    rgbnp2 = events2rgb(seq2)

    marginsize = 5
    size1 = (seq1.shape[2], seq1.shape[3])
    size2 = (seq2.shape[2], seq2.shape[3])
    size = (size1[0] + marginsize + size2[0], size2[1])
    duration = rgbnp1.shape[0]
    if rgbnp1.shape[0] != rgbnp2.shape[0]:
        print('invalid input')
        return

    fps = 24.0
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    '''
        gray: isColor=False
        rgb:  isColor=True
    '''
    margin = np.ones([size1[0], marginsize, 3], dtype=np.uint8)
    margin = margin * 128
    out = cv2.VideoWriter(out, fourcc, fps, size, isColor=True)

    for i in range(duration):
        # np.uint8
        # (h, w, 3)
        data1 = rgbnp1[i].transpose(1, 2, 0) * 255
        data2 = rgbnp2[i].transpose(1, 2, 0) * 255

        data = np.concatenate([data1, margin, data2], axis=1)
        # print_info(data,'data')
        # input('-----')
        out.write(data)
    out.release()
    # print(out)
    return

def make_folder_index(root,index,format=4):

    if type(index)==int:
        folder_name=str(index).zfill(format)
    elif type(index)==str:
        folder_name=index

    newf=os.path.join(root,folder_name)
    if not os.path.exists(newf):
        os.mkdir(newf)
        print('make dictory: ', newf)
    else:
        print(newf,' exists.')
    return newf


def save_video_jpgs(x,folder,format=2):
    '''
    x (T,3,H,W)

    :return:
    '''
    T, C, H, W=x.shape
    for i in range(T):
        # print(i)
        # print_shape(x[i])
        cv2.imwrite(filename=os.path.join(folder, str(i).zfill(format) + '.jpg'), img=x[i])


    return

def cat_two_image(x1,x2,border_color=128,border_width=2):
    # # addmargin_v
    # x1=addmargin_v(x1,margin=1,color=(255,255,255))
    # x2=addmargin_v(x2,margin=1,color=(255,255,255))
    x1=addmargin_v(x1,margin=border_width,color=border_color)
    x2=addmargin_v(x2,margin=border_width,color=border_color)
    return np.concatenate([x1,x2],axis=2)

def four_event2v(seq1, seq2, seq3, seq4, out='combined.mp4'):
    rgbnp1 = events2rgb(seq1)
    rgbnp2 = events2rgb(seq2)
    rgbnp3 = events2rgb(seq3)
    rgbnp4 = events2rgb(seq4)

    marginsize = 5
    size1 = (seq1.shape[2], seq1.shape[3])
    size2 = (seq2.shape[2], seq2.shape[3])
    size = (4 * size1[0] + 3 * marginsize, size2[1])
    duration = rgbnp1.shape[0]
    if rgbnp1.shape[0] != rgbnp2.shape[0]:
        print('invalid input')
        return

    fps = 24.0
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    '''
        gray: isColor=False
        rgb:  isColor=True
    '''
    margin = np.ones([size1[0], marginsize, 3], dtype=np.uint8)
    margin = margin * 128
    out = cv2.VideoWriter(out, fourcc, fps, size, isColor=True)

    for i in range(duration):
        # np.uint8
        # (h, w, 3)
        data1 = rgbnp1[i].transpose(1, 2, 0) * 255
        data2 = rgbnp2[i].transpose(1, 2, 0) * 255
        data3 = rgbnp3[i].transpose(1, 2, 0) * 255
        data4 = rgbnp4[i].transpose(1, 2, 0) * 255

        # data=np.concatenate([data1,margin,
        #                      data2,margin,
        #                      data3,margin,
        #                      data4],axis=1)

        data = np.concatenate([data1, margin,
                               data2, margin,
                               data3, margin,
                               data4], axis=1)
        # print_info(data,'data')
        # input('-----')
        out.write(data)
    out.release()
    # print(out)
    return


def new_folder(root=''):

    now=time.strftime("%Y%m%d-%H%M%S")

    newf=os.path.join(root,now)
    print('make dictory: ',newf)
    os.mkdir(newf)

    return newf

# from event_utils import ModelOut2Video


def change_color(frame):
    '''

    把三种yanse 情况 改成另外三种

    mask 是原来的颜色 分成三类(任意个类)
    '''
    b, g, r = cv2.split(frame)

    mask1 = b > 200
    mask2 = g > 200

    mask3 = np.ones([438, 440])
    mask3 = mask3 - mask1 - mask2
    mask3 = np.where(mask3 > 0, True, False)

    # mask3 = b.any()<200 or g.any()<200
    print_shape(mask1)
    # frame[mask] = [0,255,255]
    frame[:, :, :3][mask1] = [0, 0, 255]
    frame[:, :, :3][mask2] = [255, 0, 0]
    frame[:, :, :3][mask3] = [200, 200, 200]

    return frame

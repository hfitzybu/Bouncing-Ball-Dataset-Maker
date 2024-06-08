import torch
import numpy as np
from utils.utilsybu import *

# from patchify import patchify
from spikingjelly.datasets import play_frame

# from utilybu import print_info,pause
import cv2


def C_last(e):
    '''
    e: T C H W

    把 channel 放在最后
    '''
    if len(e.shape) == 4:
        T, C, H, W = e.shape
        return e.permute(0, 2, 3, 1).contiguous().view(T * H * W, C)

    elif len(e.shape) == 5:
        T, B, C, H, W = e.shape
        return e.permute(0, 1, 3, 4, 2).contiguous().view(T * B * H * W, C)


def event2onehot_torch(event, device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")):
    '''
    输入输出都是 torch
    (B, C1,W,   H)   -> (B, C2, W,   H)
    (B, 2, 256, 256) -> (B, 3, 256,256)
    把每个 pixel 的 两个channel (None, P, N) one hot
    '''
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # B = event.size(0)
    B, C, W, H = event.shape

    one = torch.ones([B, W, H], dtype=torch.float).to(device)
    one_hot = torch.zeros([B, 3, W, H], dtype=torch.float).to(device)

    # 0: no event
    # 1: p
    # 2: n
    # print_info(event,'event')
    # print_info(one,'one')
    # print_info(torch.sum(event,axis=1),'torch.sum(event,axis=0)')
    # print_info(one_hot,'onehot')
    # print_info(torch.sum(event, axis=1),'torch.sum(event, axis=1)')

    # channel wise sum
    one_hot[:, 0] = one - torch.sum(event, axis=1)
    one_hot[:, 1] = event[:, 0]
    one_hot[:, 2] = event[:, 1]
    # print_info(one_hot,'onehot')

    return one_hot


def events3_2_rgb(events):
    '''

    input   :
    events       : (f/B, 3, W, H) np.float

    pos: red
    neg: blue

    cv2 好像是 BGR 不是 RGB

    # 0: no event -> 0
    # 1: p        -> blue
    # 2: n        -> red

    我最厉害的地方就是自己写的代码过一个月自己都看不懂

    max value -> 1
    others    -> 0

    :return:
    '''
    # print_info(events,'events')
    frame_num, p, h, w = events.shape

    ggg = np.zeros([3, frame_num * h * w], dtype=np.uint8)
    # kkk = np.zeros([frame_num, 3, h, w], dtype=np.uint8)

    xxx = events.transpose(1, 0, 2, 3)
    xxx = xxx.reshape(3, -1)

    ggg[np.argmax(xxx, axis=0), np.arange(frame_num * h * w)] = 255
    ggg = ggg.reshape(3, frame_num, h, w)
    ggg = ggg.transpose(1, 0, 2, 3)

    kkk = event2rgb_(ggg[:, 1:3])  # (30, 256, 256, 3)

    # kkk[:,0,:,:]=ggg[:,1,:,:]
    # kkk[:,2,:,:]=ggg[:,2,:,:]
    c3=np.concatenate([kkk[0,:,:,0],
               kkk[0,:,:,1],
               kkk[0,:,:,2]],axis=0)
    print_info(c3)
    cv2.imwrite('event2rgb.png',c3
               )
    # cv2.waitKey(0)
    # print_info(kkk[:,0], 'kkk')
    # print_info(kkk[:,1], 'kkk')
    # print_info(kkk[:,2], 'kkk')
    # print_info(kkk[:], 'kkk')

    return kkk


def get_a_multistep_input(xpath='', pred_len=10, device=torch.device("cuda:0")):
    '''

    for training

    get an input for sPredNetBeta1 load numpy array and convert to torch tensor
    pred len max = 480

    :param xpath:
    :param pred_len:
    :return:

    shape: (480, 2, 256, 256)
    dtype: float64
    max: 1.0
    min: 0.0

    '''
    xgt = np.load(xpath)  # (480, 2, 256, 256)
    print_info(xgt, 'xgt')

    xgt = torch.from_numpy(xgt).type(torch.float).to(device)
    print_info(xgt, 'xgt')

    xgt = torch.unsqueeze(xgt, 0)  # (1, 480, 2, 256, 256)

    x = xgt[:, 1:pred_len + 1]  # x [1,20,2,256,256]
    target = xgt[0, 2:pred_len + 2]
    # target=xgt[0,1:pred_len+1]

    target = torch.unsqueeze(event2onehot_torch(target), 0)
    # target=torch.randn([1, 20, 3, 256, 256]).to(device)
    '''
    此时 x:  [B, T, C, H, W] 对于单个输入来说 B=1 
    x shape应该是 [T, B, C, H, W]
    '''
    x = x.permute(1, 0, 2, 3, 4)
    target = target.permute(1, 0, 2, 3, 4)
    '''
    x       :torch.Size([T, 1, 2, 256, 256])
    target  :torch.Size([T, 1, 3, 256, 256])
    '''
    return x, target


def removeBadPixel(orignal):
    '''

    T,H,W,C=orignal.shape

    不知道为什么这个event 有的像素点既有pos event 又有neg event
    接下来的操作收是把这种像素点视为没有 event
    '''

    if type(orignal) == torch.Tensor:

        T, H, W, C = orignal.shape
        # 先把array 拉直一下 方便操作 后面再reshape 回去
        orignal = orignal.reshape(T * H * W, C)

        # 找到那些 既有pos event 又有neg event的位置
        adddd = torch.sum(orignal, dim=1)

        # 搞一个 坏的像素点的位置全部为1 其他位置为0 的同shape 的array
        pixelIs2 = torch.where(adddd > 1.5, 1, 0).reshape(-1, 1)
        pixelIs2 = torch.cat([pixelIs2, pixelIs2], dim=1)

        # 原来的array 减去上面那个array 就把坏的 像素点全变成0
        orignal = orignal - pixelIs2
        orignal = orignal.view(T, H, W, C)

    elif type(orignal) == np.ndarray:
        T, H, W, C = orignal.shape
        # 先把array 拉直一下 方便操作 后面再reshape 回去
        orignal = orignal.reshape(T * H * W, C)

        # 找到那些 既有pos event 又有neg event的位置
        adddd = np.sum(orignal, axis=1)

        # 搞一个 坏的像素点的位置全部为1 其他位置为0 的同shape 的array
        pixelIs2 = np.where(adddd > 1.5, 1, 0).reshape(-1, 1)
        pixelIs2 = np.concatenate([pixelIs2, pixelIs2], axis=1)

        # 原来的array 减去上面那个array 就把坏的 像素点全变成0
        orignal = orignal - pixelIs2
        orignal = orignal.reshape(T, H, W, C)
    return orignal


def crop_256(before):
    '''
        (B,C,260,346) -> (B,C,256,256)
        (B,T,C,260,346) -> (B,T,C,256,256)
    '''
    if len(before.shape) == 4:
        return before[:, :, 2:-2, 45:-45]
    elif len(before.shape) == 5:
        return before[:, :, :, 2:-2, 45:-45]
    else:
        return None


def get_a_multistep_input_visevent(xpath='', pred_len=10, return_shape="T1CHW", device=torch.device("cuda:0")):
    '''

    for training

    get an input for sPredNetBeta1 load numpy array and convert to torch tensor
    pred len max = 480

    :param xpath:
    :param pred_len:
    :return:.
    (106, 260, 346, 2)
    dtype: uint8
    max: 255
    min: 0
    '''
    xgt = np.load(xpath) / 255  # (T, 260, 346, 2)
    T, H, W, C = xgt.shape

    '''
    不知道为什么这个event 有的像素点既有pos event 又有neg event
    接下来的操作收是把这种像素点视为没有 event

    +
    用 removeBadPixel
    '''
    xgt = removeBadPixel(xgt)
    xgt = xgt.transpose(0, 3, 1, 2)  # (B, 2, 260, 346)
    xgt = crop_256(xgt)  # (B,C,260,346) -> (B,C,256,256)
    xgt = torch.from_numpy(xgt).type(torch.float).to(device)
    xgt = torch.unsqueeze(xgt, 0)  # (1, 480, 2, 256, 256)

    if return_shape == "T1CHW":

        x = xgt[:, 1:pred_len + 1]  # x [1,20,2,256,256]
        target = xgt[0, 2:pred_len + 2]

        # 2 channel 变成 onehot 的3 channel：
        # 0: non event
        # 1: pos event
        # 2: neg event
        # 然后添加一个batch 维度
        target = torch.unsqueeze(event2onehot_torch(target), 0)
        # print_info(target,'target')

        # target=torch.randn([1, 20, 3, 256, 256]).to(device)
        '''
        此时 x:  [B, T, C, H, W] 对于单个输入来说 B=1 
        x shape应该是 [T, B, C, H, W]
        '''
        x = x.permute(1, 0, 2, 3, 4)
        target = target.permute(1, 0, 2, 3, 4)
        '''
        x       :torch.Size([T, 1, 2, 256, 256])
        target  :torch.Size([T, 1, 3, 256, 256])
        '''
        return x, target

    elif return_shape == "TCHW":
        x = xgt[0, 1:pred_len + 1]  # x [T,2,256,256]
        target = xgt[0, 2:pred_len + 2]  # x [T,2,256,256]
        return x, target
    else:
        print("Invalid Shape")

        return None, None


def make_events(xxx):
    '''

    xxx shape [T, B, C, H, W]或者 [T, C, H, W]

    把batch 的 2 channel array 变成events

    一个pixel 有两个channel 一个pos 一个neg
    比较两个哪个大让哪个变成 1 另一个变成 0

    multi step mode 的 out
    model out shape [T, B, C, H, W] = xxx.shape
    [T*B, C, H, W]
    where C = 2

    一般用来计算 IoU （或其他 metric， 因为 x/target 是 2 channel）
    :return:
    '''

    # xxx = torch2np(xxx)
    if len(xxx.shape) == 4:
        xxx = np.expand_dims(xxx, 0)

    [T, B, C, H, W] = xxx.shape

    '''
    3 channel

    0: non B
    1: pos G
    2: neg R

    events3_2_rgb 中搞成了

    0: pos B
    1: non G
    2: neg R

    所以这个函数里统一一下 把 0 1 channel 对调一下， 反正只是作为rgb输出 只是个显示问题
    btw, cv2 的channel 是 BGR 不是 RGB 

    '''

    ''' -----------------------------------------------------'''
    xxx = xxx.transpose(0, 1, 3, 4, 2)
    xxx = xxx.reshape(-1, C)

    arg = np.argmax(xxx, axis=1)
    # print(arg.shape)
    # pause()
    arg = np.expand_dims(arg, axis=1)
    # ind=xxx.argsort(0)
    batch_onehot = np.zeros_like(xxx)
    np.put_along_axis(batch_onehot, arg, 1, axis=1)
    from utils.utilsybu import print_info
    batch_onehot = batch_onehot.reshape(T * B, H, W, C)
    xxx = batch_onehot.transpose(0, 3, 1, 2)
    ''' -----------------------------------------------------'''
    # xxx_copy = np.zeros_like(xxx)
    # xxx_copy[:, 0] = xxx[:, 1]  # B
    # # xxx_copy[:,1]=0       # G
    # xxx_copy[:, 2] = xxx[:, 2]  # R
    # print_info(xxx[:,0])
    # print_info(xxx[:,1])
    # import cv2
    # from test import PATH_BOUNCINGBALL
    # cv2.imshow(PATH_BOUNCINGBALL,xxx[1,1])
    # cv2.waitKey(0)
    xxx = xxx[:, 1:3]
    # print(xxx.shape)
    xxx_copy = xxx
    # xxx_copy[:,0]=xxx[:,1]
    # xxx_copy[:,1]=xxx[:,0]
    return xxx_copy


# def make_2channel_events(arr, th=1e-1):
#     '''
#
#     [B, 2, H, W]
#
#     :param arr:
#     :return:
#     '''
#     T, C, W, H = arr.shape
#     # print_info(arr)
#     # from utilybu import print_info,pause
#     # import cv2
#     events = arr.transpose(1, 0, 2, 3)  # 2,T,H,W
#     # pos=events[0:1]
#     # neg=events[1:2]
#
#     pp = np.where(events[0:1] - events[1:2] > th, 1.0, 0.0)
#     nn = np.where(events[1:2] - events[0:1] > th, 1.0, 0.0)
#
#     # print_info(pp,'pp')
#     # print_info(nn,'nn')
#
#     event = np.concatenate([pp, nn], axis=0)
#     event = event.reshape(2, T, H, W)
#     event = event.transpose(1, 0, 2, 3)
#     # cv2.imshow('ddddd',event[2,1])
#     # cv2.waitKey(0)
#     # pause()
#
#     # event2onehot_torch()
#
#     return event


def gray2event(gray, th=1e-1):
    '''
    [B, T, H, W]
    '''
    B, T, W, H = gray.shape

    gray = gray.transpose(1, 0, 2, 3).astype(np.float64)  # T, B, H, W
    future=gray[1:]
    resi=future-gray[:-1]

    pp = np.where(resi >  th, 1.0, 0.0) # T B H W
    nn = np.where(resi < -th, 1.0, 0.0) # T B H W

    pp=np.expand_dims(pp,axis=2)
    nn=np.expand_dims(nn,axis=2) # T B 1 H W

    events=np.concatenate([pp,nn],axis=2) # T B 2 H W

    events = events.transpose(1, 0, 2, 3, 4) # B T 2 H W

    return events # B T 2 H W

def gray2event_(gray, th=1e-1,gap=1):
    '''
    [T, H, W]
    '''
    T, W, H = gray.shape

    gray = gray.astype(np.float64)  # T, H, W

    # gray=gray[0:T:gap]
    future=gray[gap:T:gap]
    resi=future-gray[0:-gap:gap]

    # print('',gray.shape)
    # print('resi',resi.shape)

    pp = np.where(resi >  th, 1.0, 0.0) # T H W
    nn = np.where(resi < -th, 1.0, 0.0) # T H W

    pp=np.expand_dims(pp,axis=1)
    nn=np.expand_dims(nn,axis=1) # T 1 H W

    events=np.concatenate([pp,nn],axis=1) # T 2 H W
    # print('events',resi.shape)

    return events # T 2 H W


# def gray2event_(gray, th=1e-1,gap=1):
#     '''
#     [T, H, W]
#     '''
#     T, W, H = gray.shape
#
#     gray = gray.astype(np.float64)  # T, H, W
#     future=gray[1:]
#     resi=future-gray[:-1]
#
#     pp = np.where(resi >  th, 1.0, 0.0) # T H W
#     nn = np.where(resi < -th, 1.0, 0.0) # T H W
#
#     pp=np.expand_dims(pp,axis=1)
#     nn=np.expand_dims(nn,axis=1) # T 1 H W
#
#     events=np.concatenate([pp,nn],axis=1) # T 2 H W
#
#     return events # T 2 H W
def get_a_multistep_input2(xpath='', pred_len=10, device=torch.device("cuda:0")):
    '''

    for testing

    get an input for sPredNetBeta1 load numpy array and convert to torch tensor
    pred len max = 480

    :param xpath:
    :param pred_len:
    :return:
    '''
    xgt = np.load(xpath)  # (480, 2, 256, 256)
    xgt = torch.from_numpy(xgt).type(torch.float).to(device)

    xgt = torch.unsqueeze(xgt, 0)  # (1, 480, 2, 256, 256)

    x = xgt[:, 1:pred_len + 1]  # x [1,20,2,256,256]
    target = xgt[0, 2:pred_len + 2]
    # target=xgt[0,1:pred_len+1]

    target = torch.unsqueeze(target, 0)
    # target=torch.randn([1, 20, 3, 256, 256]).to(device)
    '''
    此时 x:  [B, T, C, H, W] 对于单个输入来说 B=1 
    x shape应该是 [T, B, C, H, W]
    '''
    x = x.permute(1, 0, 2, 3, 4)
    target = target.permute(1, 0, 2, 3, 4)
    '''
    x       :torch.Size([T, 1, 2, 256, 256])
    target  :torch.Size([T, 1, 3, 256, 256])
    '''
    return x, target


def ModelOut2Events(xxx):
    '''

    model out shape [T, B, C, H, W] = xxx.shape
    [T*B, C, H, W]
    :return:
    '''

    xxx = torch2np(xxx)

    [T, B, C, H, W] = xxx.shape

    '''
    3 channel

    0: non B
    1: pos G
    2: neg R

    events3_2_rgb 中搞成了

    0: pos B
    1: non G
    2: neg R

    所以这个函数里统一一下 把 0 1 channel 对调一下， 反正只是作为rgb输出 只是个显示问题
    btw, cv2 的channel 是 BGR 不是 RGB 

    '''

    ''' -----------------------------------------------------'''
    xxx = xxx.transpose(0, 1, 3, 4, 2)
    xxx = xxx.reshape(-1, C)

    arg = np.argmax(xxx, axis=1)
    # print(arg.shape)
    # pause()
    arg = np.expand_dims(arg, axis=1)
    # ind=xxx.argsort(0)
    batch_onehot = np.zeros_like(xxx)
    np.put_along_axis(batch_onehot, arg, 1, axis=1)

    batch_onehot = batch_onehot.reshape(T * B, H, W, C)
    xxx = batch_onehot.transpose(0, 3, 1, 2)
    ''' -----------------------------------------------------'''
    xxx_copy = np.zeros_like(xxx)
    xxx_copy[:, 0] = xxx[:, 1]  # B
    # xxx_copy[:,1]=0       # G
    xxx_copy[:, 2] = xxx[:, 2]  # R

    xxx_copy=xxx

    return xxx_copy
def ModelOut2Events3(xxx):
    '''

    model out shape [T, B, C, H, W] = xxx.shape
    [T*B, C, H, W]
    :return:
    '''

    xxx = torch2np(xxx)

    [T, B, C, H, W] = xxx.shape

    '''
    3 channel

    0: non B
    1: pos G
    2: neg R

    events3_2_rgb 中搞成了

    0: pos B
    1: non G
    2: neg R

    所以这个函数里统一一下 把 0 1 channel 对调一下， 反正只是作为rgb输出 只是个显示问题
    btw, cv2 的channel 是 BGR 不是 RGB 

    '''

    ''' -----------------------------------------------------'''
    xxx = xxx.transpose(0, 1, 3, 4, 2)
    xxx = xxx.reshape(-1, C)

    arg = np.argmax(xxx, axis=1)
    # print(arg.shape)
    # pause()
    arg = np.expand_dims(arg, axis=1)
    # ind=xxx.argsort(0)
    batch_onehot = np.zeros_like(xxx)
    np.put_along_axis(batch_onehot, arg, 1, axis=1)

    batch_onehot = batch_onehot.reshape(T * B, H, W, C)
    xxx = batch_onehot.transpose(0, 3, 1, 2)
    ''' -----------------------------------------------------'''
    # xxx_copy = np.zeros_like(xxx)
    # xxx_copy[:, 0] = xxx[:, 1]  # B
    # # xxx_copy[:,1]=0       # G
    # xxx_copy[:, 2] = xxx[:, 2]  # R
    #
    # xxx_copy=xxx

    return xxx

def ModelOut2Events2(xxx):
    '''
    multi step mode 的 out
    model out shape [T, B, C, H, W] = xxx.shape
    [T*B, C, H, W]
    where C = 2

    一般用来计算 IoU （或其他 metric， 因为 x/target 是 2 channel）
    :return:
    '''

    xxx = torch2np(xxx)

    [T, B, C, H, W] = xxx.shape

    '''
    3 channel

    0: non B
    1: pos G
    2: neg R

    events3_2_rgb 中搞成了

    0: pos B
    1: non G
    2: neg R

    所以这个函数里统一一下 把 0 1 channel 对调一下， 反正只是作为rgb输出 只是个显示问题
    btw, cv2 的channel 是 BGR 不是 RGB 

    '''

    ''' -----------------------------------------------------'''
    xxx = xxx.transpose(0, 1, 3, 4, 2) # T B H W C
    xxx = xxx.reshape(-1, C) # T*B*H*W C

    arg = np.argmax(xxx, axis=1)
    # print(arg.shape)
    # pause()
    arg = np.expand_dims(arg, axis=1)
    # ind=xxx.argsort(0)
    batch_onehot = np.zeros_like(xxx)
    np.put_along_axis(batch_onehot, arg, 1, axis=1)

    batch_onehot = batch_onehot.reshape(T * B, H, W, C)
    xxx = batch_onehot.transpose(0, 3, 1, 2)
    ''' -----------------------------------------------------'''

    xxx = xxx[:, 1:3] # (T*B,C,H,W)

    return xxx


def warmup_forward(pred_model, x, warmup=1, device=torch.device("cuda:0")):
    '''

    :param model:
    :param x:
    :param warmup:
    :return:
    '''
    B, T, C, H, W = x.shape

    pred_model.set_step_mode('s')

    y = torch.zeros((B, T, 3, H, W), dtype=torch.float)

    warmup = 999999999
    for i in range(T):

        if i < warmup:
            x_this = x[:, i]
        else:
            x_this = ModelOut2ModelIn(y_this, device=device)

        y_this = pred_model(x_this)

        y[:, i] = y_this

    print_info(x)
    this_frame = ModelOut2Video(y)
    # np2v(this_frame)
    asd = x[0].cpu().numpy()
    gt_frame = events2rgb(asd)
    print_info(gt_frame)

    # for i in range(T):
    #     cv2.imshow(str(i),this_frame[i])
    #     cv2.waitKey(200)
    for i in range(T):
        cv2.imwrite('paper_result/visout/mmgt' + str(i) + '.png', gt_frame[i].transpose(1, 2, 0))
        cv2.imwrite('paper_result/visout/mmpred' + str(i) + '.png', this_frame[i])
        cv2.waitKey(200)
    cv2.imshow(str(5), this_frame[5])
    cv2.waitKey(200)
    print_info(this_frame)

    return


def ModelOut2ModelIn(xxx, device):
    '''

    用 argmax 使 event tensor 只有一个为 1 其余两个为 0

    single step mode 的 out to
    single step mode 的 in
    model out shape [B, C, H, W] = xxx.shape
    [T*B, C, H, W]
    where C = 2

    一般用来计算 IoU （或其他 metric， 因为 x/target 是 2 channel）
    :return:
    '''

    xxx = torch2np(xxx)

    [B, C, H, W] = xxx.shape

    ''' -----------------------------------------------------'''
    xxx = xxx.transpose(0, 2, 3, 1)  # channel 移到最后
    xxx = xxx.reshape(-1, C)  # ( B*H*W , C)

    arg = np.argmax(xxx, axis=1)
    # print(arg.shape)
    # pause()
    arg = np.expand_dims(arg, axis=1)
    # ind=xxx.argsort(0)
    batch_onehot = np.zeros_like(xxx)
    np.put_along_axis(batch_onehot, arg, 1, axis=1)

    batch_onehot = batch_onehot.reshape(B, H, W, C)
    xxx = batch_onehot.transpose(0, 3, 1, 2)
    ''' -----------------------------------------------------'''

    # print_info(xxx[:,0],'non')
    # print_info(xxx[:,1],'pos')
    # print_info(xxx[:,2],'neg')
    xxx = xxx[:, 1:3]
    xxx = torch.from_numpy(xxx).to(device)
    # print_info(xxx)
    #
    # pause()

    return xxx


def ModelOut2Video(xxx):
    '''

    这个是好用的 好用个坤把 --2024.04.14
    xxx: torch.Size([T, B, 3, 256, 256]) with gradient and on cuda

    out: numpy [T*B, 256, 256, 3]
    :return:
    '''

    # xxx = torch2np(xxx)
    # [T, B, C, H, W] = xxx.shape

    # xxx = xxx.reshape(T * B, C, H, W)

    # xxx=events3_2_rgb(xxx)
    # xxx = events3_2_rgb(xxx)
    # xxx=event2rgb_(xxx)
    '''
    '''
    xxx=ModelOut2Events2(xxx)
    xxx = event2rgb_(xxx)

    '''
    '''

    # event2rgb_-1
    # xxx=xxx.transpose(0, 2, 3, 1)

    return xxx

# def getX(sample='bb64',len=30):
#
#     bb64='E:/su/snn/code/dataset/bouncingball/bb64'
#
#     if sample == 'bb64':
#         sample_gt = np.load(bb64+'/7.npy')[0:len + 2]
#         sample = sample_gt[1:len + 1]
#         gt = sample_gt[2:len + 2]



    return sample,gt
# def ModelOut2RGBs(xxx):
#     xxx = torch2np(xxx)
#     [T, B, C, H, W] = xxx.shape
#
#     xxx=xxx.reshape(T*B,C,H,W)
#
#     xxx=events3_2_rgb(xxx)
#
#     xxx=xxx.transpose(0, 2, 3, 1)
#
#     return


'''
下面两个函数是把 pred net 的输出 保存到硬盘上（以MP4或者jpg 的格式）

'''


def save_v(y, root='modelout/mp4'):
    y_v = ModelOut2Video(y)  # ([T, B, 3, 256, 256]) with gradient and on cuda ------>  numpy [T*B, 256, 256, 3]
    savev = os.path.join(new_folder(root), 's.mp4')
    np2v(seq=y_v, out=savev, fps=2.0)  # numpy [T*B, 256, 256, 3] 保存到硬盘

    return


def save_images(y, root='modelout/jpg'):
    y_v = ModelOut2Video(y)  # ([T, B, 3, 256, 256]) with gradient and on cuda ------>  numpy [T*B, 256, 256, 3]
    newf = new_folder(root)

    for i in range(y_v.shape[0]):
        cv2.imwrite(filename=os.path.join(newf, str(i) + '.jpg'), img=y_v[i])

    return


def save_images_and_v(y, gt, save_folder):
    y_v = ModelOut2Video(
        y)  # ([T, B, 3, 256, 256]) with gradient and on cuda ------>  numpy [T*B, 256, 256, 3] (30, 256, 256, 3)
    gt = ModelOut2Video(gt)
    print_info(y_v, 'y_v')
    print_info(gt, 'gt')
    whole = np.concatenate([y_v, gt], axis=2)
    print_info(whole, 'whole')

    newf = new_folder(save_folder)
    savev = os.path.join(newf, 's.mp4')

    np2v(seq=whole, out=savev, fps=2.0)  # numpy [T*B, 256, 256, 3] 保存到硬盘

    # for i in range(whole.shape[0]):
    #     cv2.imwrite(filename=os.path.join(newf,str(i)+'.jpg'),img=whole[i])

    #
    for i in range(whole.shape[0]):
        cv2.imwrite(filename=os.path.join(newf, 'y_' + str(i) + '.jpg'), img=y_v[i])
    for i in range(whole.shape[0]):
        cv2.imwrite(filename=os.path.join(newf, 'gt_' + str(i) + '.jpg'), img=gt[i])
    return

def binary2float(e):



    return


def compute_active(e):
    '''


    :param events:
    :return:
    '''
    # print_info(e)

    T, C, H, W = e.shape

    s = torch.sum(e.view(T, -1), dim=1)
    # print_info(noise.view(T,-1))
    # print(s)

    return s


def concat_and_save(np2, torch3, save_folder):
    np2 = events2rgb(torch.squeeze(np2))

    y_v = ModelOut2Video(torch3)  # ([T, B, 3, 256, 256]) with gradient and on cuda ------>  numpy [T*B, 256, 256, 3]
    # gt=ModelOut2Video(gt)
    # print_info(y_v,'y_v')
    # print_info(gt,'gt')
    # whole=np.concatenate([y_v,torch3],axis=2)
    # print_info(whole,'whole')
    # newf = new_folder(save_folder)
    # savev = os.path.join(newf, 's.mp4')

    # np2v(seq=whole, out=savev, fps=2.0)  # numpy [T*B, 256, 256, 3] 保存到硬盘
    #
    # for i in range(whole.shape[0]):
    #     cv2.imwrite(filename=os.path.join(newf,str(i)+'.jpg'),img=whole[i])
    for i in range(np2.shape[0]):
        # print_info(whole[i].transpose(1,2,0),'whole[i]')
        # cv2.imwrite(filename=os.path.join('noise_', str(i) + '.jpg'), img=np2[i].transpose(1, 2, 0))
        img = np2[i].transpose(1, 2, 0)
        whole = np.concatenate([img, y_v[i]], axis=1)
        # print_info(whole,'whole')
        cv2.imwrite(filename=os.path.join('noise_', str(i) + '.jpg'), img=whole)

    return


def IoU(y, gt):
    '''

    :param y:      (B, 2, 256, 256) dtype: float64 0.0 or 1.0
    :param gt:     (B, 2, 256, 256) dtype: float64 0.0 or 1.0
    :return:        1 - error_num/active_num
    '''
    B = y.shape[0]
    # print(B)
    # print_info(y,'y')
    # print_info(gt,'gt')
    sum = y + gt
    bing = np.where(sum > 0.99, 1.0, 0)
    jiao = np.where(sum > 1.99, 1.0, 0)

    # xx=events2rgb(jiao)
    # cv2.imshow(',',255*xx[0].transpose(1,2,0))
    # cv2.waitKey(0)

    jiao = jiao.reshape(B, -1)
    bing = bing.reshape(B, -1)
    # print_info(jiao,'jiao')

    # sum_jiao=np.sum(jiao, axis=1)
    # sum_bing=np.sum(bing, axis=1)
    # print('sum_jiao:',sum_jiao)
    # print('sum_bing:',sum_bing)

    if np.sum(bing, axis=1).all() != 0:

        acc = np.sum(jiao, axis=1) \
              / np.sum(bing, axis=1)
    else:
        acc = np.zeros_like(np.sum(jiao, axis=1))

    ave_this = acc[~numpy.isnan(acc)]
    ave_this = np.mean(ave_this)
    # print(ave_this)

    acc[np.isnan(acc)] = ave_this

    # print_info(acc,'acc')

    return acc


def two_event2v(seq1, seq2, out='combined.mp4'):
    '''


    :param seq1: ()
    :param seq2:
    :param out:
    :return:
    '''

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


def region_polarity(x):
    '''

    :param x: numpy array, T,C,H,W
    :return:
    '''
    T, C, H, W = x.shape

    x_r = x.transpose(1, 0, 2, 3).reshape(C, T, H * W)  # (2,T,H*W)

    pos = np.sum(x_r[0], axis=1)  # (T)
    neg = np.sum(x_r[1], axis=1)  # (T)

    regionP = (pos - neg) / (H * W)  # (T)

    return regionP


def regionfy(x, region_size=2, thresh=0.2501):
    T, C, H, W = x.shape
    H_p = int(H / region_size)
    W_p = int(W / region_size)
    from patchify import patchify

    # (T,C,H,W) -> (1,1,H_p,W_p,T,C,region_size,region_size),
    # where T_p=1, W_p =1, H_p=H/region_size,W_P=W/region_size
    # e.g (30,2,256,256) -> (1, 1, 128, 128, 30, 2, 2, 2)
    x_p = patchify(x, (T, C, region_size, region_size), step=region_size)  # x patches
    # print(x_p.shape)

    x_p = np.squeeze(x_p, axis=(0, 1))  # (H_p,W_p,T,2,region_size,region_size),
    # print(x_p.shape)

    x_p = x_p.reshape(H_p * W_p * T, C, region_size, region_size)

    rp = region_polarity(x_p)

    rp = rp.reshape(1, H_p, W_p, T)

    pos = np.where(rp > thresh, 1.0, 0.0)
    neg = np.where(rp < -thresh, 1.0, 0.0)

    regioned = np.concatenate([pos, neg], axis=0)

    # x_B=x_p.reshape(H_p,W_P)
    regioned = regioned.transpose(3, 0, 1, 2)

    return regioned


def region_IoU(x1, x2, region_size=2, th='auto'):
    '''

    :param x1:
    :param x2:
    :param region_size:
    :return:
    '''
    if th == 'auto':
        th = 1 / (region_size * region_size) + 1e-5
    # print(th)
    if x1.shape != x2.shape:
        print('two tensor have different shape.')
        return 0.0

    regioned_x1 = regionfy(x1, region_size=region_size, thresh=th)
    regioned_x2 = regionfy(x2, region_size=region_size, thresh=th)

    # print_info(regioned_x1,'regioned_x1')
    # # play_frame(regioned_x1)
    #
    # print_info(regioned_x2,'regioned_x2')

    iou = IoU(regioned_x1, regioned_x2)

    # print('this iou:',iou)
    # play_frame(np.concatenate([regioned_x1,regioned_x2],axis=3))

    # pause()

    # from spikingjelly.datasets import play_frame
    #
    # pause()
    # print(x1)
    # polarity=region_polarity(x1)
    # print(polarity)

    return iou
def white_bg(image):
    '''

    :param image: T
    :return:
    '''
    # invert = cv2.bitwise_not(image)

    mask = np.sum(image, axis=3)
    # print('mask shape ',mask.shape)
    c_first = image.transpose(0, 3, 1, 2)

    b = np.where(mask > 0, c_first[:,0], 255)
    g = np.where(mask > 0, c_first[:,1], 255)
    r = np.where(mask > 0, c_first[:,2], 255)

    white_return = np.concatenate([np.expand_dims(r, 3), np.expand_dims(g, 3), np.expand_dims(b, 3)], axis=3)
    # cv2.imshow('cc,', white_return)
    # cv2.waitKey()
    #
    # print(mask.shape)
    # pause()
    return white_return
def event2rgb_(events,bg='w',mode='rgb'):
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
        events = events.cpu().numpy()

    # from event_utils import removeBadPixel
    T, C, H, W = events.shape

    # n mnist beizhushi diao le
    # if np.max(events) > 1:
    #     events = events / 255

    # rgb[:,0,:,:]=events[:,0,:,:]
    # rgb[:,2,:,:]=events[:,1,:,:]
    events = events.transpose((0, 2, 3, 1))
    events = removeBadPixel(events)
    events = events.transpose((0, 3, 1, 2))
    # print(events.shape)

    # blank=np.zeros([frame_num, 1, h, w],dtype=np.uint8)
    color_events = np.zeros([T, 3, H, W], dtype=np.uint8)

    color_events[:, 0, :, :] = events[:, 0, :, :]
    color_events[:, 2, :, :] = events[:, 1, :, :]
    color_events=color_events.transpose(0,2, 3, 1)
    if np.max(color_events) < 255:
        color_events = color_events * 255

    if bg=='w' or bg=='white':
        # print_info(color_events)
        # (H W 3) -> ( )
        color_events=white_bg(color_events)
    elif bg =='b' or bg =='black':
        pass
    else :
        color_events=None
        print('bg error')


    return color_events
def all2numpy(x):

    if type(x) == torch.Tensor:
        x=x.cpu().numpy()


    return x
# from utilybu import pause

def play(x):

    x=all2numpy(x)

    # if type(x) == torch.Tensor:
    #     T, H, W, C = x.shape
    #
    #     x=x.cpu().numpy
    #
    # elif type(x) == np.ndarray:

    T, H, W, C = x.shape

    x=event2rgb_(x)

    for i in range(T):

        cv2.imshow('frame: '+str(i),x[i])
        cv2.waitKey(100)
        cv2.destroyWindow('frame: '+str(i))
        # pause()


    return

def play_bb(x,bb):


    def gravityball_bouncingbox(gt):
        print(gt)

        r,x,y=gt

        # r=8

        return (x-r,y-r),(x+r,y+r)


    x=all2numpy(x)

    # if type(x) == torch.Tensor:
    #     T, H, W, C = x.shape
    #
    #     x=x.cpu().numpy
    #
    # elif type(x) == np.ndarray:

    T, H, W, C = x.shape
    print('T',T)

    x=event2rgb_(x)

    for i in range(T):
        p1,p2=gravityball_bouncingbox(bb[i])
        # print(p1,p2)

        show=cv2.rectangle(x[i],p1,p2,(0,0,0),1,1)

        cv2.imshow('frame: '+str(i),show)
        cv2.waitKey(100)
        # pause()


    return





def play_bb2(x,bb):


    def gravityball_bouncingbox(gt):
        print(gt)

        # r,x,y=gt
        x,y,h,w=gt

        x=int(x)
        y=int(y)
        h=int(h)
        w=int(w)

        # r=8

        return (x-w,y-h),(x+w,y+h)


    x=all2numpy(x)

    # if type(x) == torch.Tensor:
    #     T, H, W, C = x.shape
    #
    #     x=x.cpu().numpy
    #
    # elif type(x) == np.ndarray:

    T, H, W, C = x.shape

    x=event2rgb_(x)

    for i in range(T):
        p1,p2=gravityball_bouncingbox(bb[i])
        # print(p1,p2)

        show=cv2.rectangle(x[i],p1,p2,(0,0,0),1,1)

        cv2.imshow('frame: '+str(i),show)
        cv2.waitKey(100)
        # pause()


    return

def save_bb(x,bb,save_path):

    def gravityball_bouncingbox(gt,r=8):
        # print(gt)

        # r,x,y=gt
        x,y=gt

        x=int(x)
        y=int(y)

        # r=8

        return (x-r,y-r),(x+r,y+r)


    T, H, W, C = x.shape

    x=event2rgb_(x)

    p1, p2 = gravityball_bouncingbox(bb[0])
    print(p1,p2)
    # print('x', x.shape)

    show = cv2.rectangle(x[0], p1, p2, (0, 0, 0), 1, 1)

    # cv2.imshow('frame: ' + str(0), show)
    cv2.imwrite(filename=save_path,img=show)
    # cv2.waitKey(0)

    return


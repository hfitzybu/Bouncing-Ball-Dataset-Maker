from utils.utilsybu import AverageMeter,print_info,pause
from utils.event_utils import event2onehot_torch,C_last

from utils.event_utils import event2onehot_torch, ModelOut2Events
from model.spikedriven.spikeformerpredictor import SDTPredictor

from utils.event_utils import ModelOut2Events2
from spikingjelly.activation_based import neuron, functional, surrogate, layer
import torch
from utils.dataloader import *

from torch.utils.data import Dataset, DataLoader

def train_one_epoch_sPredNet(train_loader, model, loss_f, optimizer):
    losses = AverageMeter()
    # loss_plot = []

    for ww, data in enumerate(train_loader, 0):
        x, gt = data
        [B, T, C, H, W] = x.shape
        x = x.type(torch.float)
        '''
        x shape应该是 [T, B, C, H, W]
        '''
        x = x.permute(1, 0, 2, 3, 4)
        # print_info(x,'x')
        gt = gt.view(B * T, C, H, W)  # [B*T, 2, H, W]
        gt = event2onehot_torch(gt)  # [B*T, 3, H, W]
        gt = gt.view(B, T, 3, H, W)  # [B, T, 3, H, W]
        gt = gt.permute(1, 0, 2, 3, 4)  # [T, B, 3, H, W]

        # print_info(x,'x')
        # print_info(gt,'gt')
        # pause()

        y = model(x) # ([T, B, 3, H W)
        # print_info(y,'y')
        # pause()
        y=C_last(y)
        gt=C_last(gt)

        # print_info(y,'y')
        # print_info(gt,'gt')
        # loss = loss_simple_pre(pred=y, gt=gt)
        # ([T, B, 3, H W)
        loss = loss_f(y, gt)
        print('loss',loss)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # print_info(loss,'loss')
        losses.update(loss.item(), n=1)
        # loss_plot.append(loss)
        functional.reset_net(model)
        # pause()

    return losses.avg

def train_sPredNet(train_loader, model, loss_f, optimizer, epoch,is_evaluate=True, is_save=False):
    '''

    train sPredNet23

    '''
    losses = AverageMeter()

    reon_loss = []

    for ee in range(0, epoch):
        # if is_evaluate:
        #     evalueate_during_training(model,save_v='play/'+str(ee)+'.mp4')

        loss = train_one_epoch_sPredNet(train_loader=train_loader,
                                        model=model,
                                        loss_f=loss_f,
                                        optimizer=optimizer)
        losses.update(loss, n=1)
        reon_loss.append(loss)
        # evaluate_trained()
        with open('model_save/train_log/log.txt',"a") as log:
            log.write(str(ee)+", "+str(loss)+",\n")
        print('epoch ', ee, ', loss:', loss)
        if ee%20==0:
            if is_save:
                saveee = 'model_save/'
                saveee_latest = 'model_save/latest.pth'

                torch.save(model.state_dict(), saveee + str(ee).zfill(4) + '.pth')
                torch.save(model.state_dict(), saveee_latest)
                print('saved model in ',saveee_latest)
            # cv2.imwrite(filename='train_save/all/' + str(ee) + '.jpg', img=save_all)
            if is_evaluate:
                # model.set_step_mode('s')
                eval_path='model_save/eval/'+str(ee).zfill(4)
                if not os.path.exists(eval_path):
                    os.makedirs(eval_path)
                    # print('model saved at ', eval_path)
                evaluate_trained(eval_path,model)
                # evaluate_trained(eval_path,patch_size=model.patch_size)

                # model.set_step_mode('m')
    from utils.utilsybu import plot_loss
    plot_loss(reon_loss)

    return
def evalueate_during_training(model,
                              save_rgb='play/eva.jpg',
                              save_v='play/eva.mp4',
                              xpath='G:/Cynthia/Q/dataset/visevent/train_numpy/event0917/108.npy'):

    x, target=get_a_multistep_input_visevent(xpath=xpath,pred_len=model.T,return_shape='T1CHW')

    # [60, 2, 256, 256])
    y=single_forward(model,x)

    y=np.squeeze(y,axis=1) # (60, 1, 256, 256, 3) -> (60, 256, 256, 3)
    # y=y.transpose(0,3,1,2).astype(np.uint8)
    y=y.astype(np.uint8)

    # (frame, 256, 256, 3) np.uint8
    np2v(seq=y,out=save_v,is_color=True)

    return

def evaluate_trained(eval_path='',model=None):
    '''

    moving mnist

    :return:
    '''

    pred_len = 18
    set_size = 2**5
    batch_size = 1
    # from utils.train_util import train_sPredNet, train_loading_sPredNet_visevent, train_loading_sPredNet,train_loading_sPredNet_mm
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # model = SDTPredictor(img_size_h=64,img_size_w=64,patch_size=patch_size,in_channels=2,T=30).cuda()
    functional.reset_net(model)
    functional.set_step_mode(model, 'm')
    from resource import PATH_BB64,PATH_BB256
    train_set = DL_BouncingBall64(HW=[64,64],
                    input_len=30,
                    set_size=1,
                    events_folder=PATH_BB64,
                    )
    trainloader = DataLoader(dataset=train_set,
                                 batch_size=batch_size,
                                 shuffle=False)

    model.set_step_mode('m')

    PATH = 'model_save/latest.pth'
    model.load_state_dict(torch.load(PATH))

    iou = AverageMeter()
    iou_1 = AverageMeter()
    iou_2 = AverageMeter()
    iou_4 = AverageMeter()

    for ww, data in enumerate(trainloader, 0):
        x, gt = data
        # print_info(x,'x')
        # print_info(gt,'gt')
        [B, T, C, H, W] = x.shape
        x = x.type(torch.float)

        # pause()
        '''
        x shape应该是 [T, B, C, H, W]
        '''
        x = x.permute(1, 0, 2, 3, 4)
        # print_info(x,'x')
        gt = gt.view(B * T, C, H, W)  # [B*T, 2, H, W]
        gt = event2onehot_torch(gt)  # [B*T, 3, H, W]
        gt = gt.view(B, T, 3, H, W)  # [B, T, 3, H, W]
        gt = gt.permute(1, 0, 2, 3, 4)  # [T, B, 3, H, W]

        y = model(x)

        y_v = ModelOut2Video(y)  # ([T, B, 3, 256, 256]) -> [T*B, 256, 256, 3]
        gt_v = ModelOut2Video(gt)  # ([T, B, 3, 256, 256]) -> [T*B, 256, 256, 3]

        margined_y_v = addmargin_v(y_v, margin=5)
        margined_gt_v = addmargin_v(gt_v, margin=5)

        concatenated = np.concatenate([margined_y_v, margined_gt_v], axis=2)
        if ww==0:
            for jj in range(concatenated.shape[0]):

                cv2.imwrite(eval_path+'/'+str(jj).zfill(2)+'.png',concatenated[jj])

        y_2 = ModelOut2Events2(y)
        gt_2 = ModelOut2Events2(gt)
        iou_this_1 = region_IoU(y_2, gt_2,region_size=1)
        iou_this_2 = region_IoU(y_2, gt_2,region_size=2)
        iou_this_4 = region_IoU(y_2, gt_2,region_size=4)

        # pause()

        # print(iou_this)
        iou_1.update(iou_this_1)
        iou_2.update(iou_this_2)
        iou_4.update(iou_this_4)
        # print_info(x,'x')
        # print_info(gt,'gt')
        functional.reset_net(model)
        # pause()
    print(iou_1.avg)
    print(iou_2.avg)
    print(iou_4.avg)
    # plot_list(y_list=[iou_1.avg,iou_2.avg,iou_4.avg],
    #           legend_list=['iou_1','iou_2','iou_4'])
    functional.reset_net(model)

    return


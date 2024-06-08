import cv2
import numpy as np
import pickle
import json

from utils.event_utils import *
from utils.utilsybu import *
from BB import *

def main():
    SAMPLES=64

    '''
    
    save list
    ------------------------------------------------------------
    numpy npy:
    
    1. events                  [T, 2, H, W]        bool
    2. events5                 [T/5, 2, H, W]      bool
    
    3. gary                    [T, 2, H, W]        bool

    4. events_noise            [T, 2, H, W]        bool
    # events5_noise            [T/5, 2, H, W]      bool
    
    5. collision_              [T]                 bool
    6. boundingboxes           [T, NUM_OBJ, 4]     float
    7. centers                 [T, NUM_OBJ, 2]     int
               
    8. yolo_target             [T, 2, H, W]        float
    
        ---------------------seq-------------------------------
    
    
    
    ------------------------------------------------------------
    png:
    
    events_noise_rgb        [T, 3, H, W]
    # events_rgb              [T, 3, H, W]
    events5_noise_rgb       [T/5, 3, H,W]
    ------------------------------------------------------------
    dictionary txt
    
    figure_num
    radius_                 
    speeds_                 
    time
    ------------------------------------------------------------
   
    '''
    for i in range(SAMPLES):
        info_dict = {}
        print('sample:' , i)

        _ = animation() # T H W, T ball_num 2
        output_this, centers, boundingboxes, radius_, speeds_, collision_, yolo_target=_
        # print(collision_)
        info_dict['figures']='ball'
        info_dict['radius'] = radius_
        info_dict['speed'] = speeds_
        info_dict['time'] = times
        centers=centers.astype(np.int32) # [T, H, W]

        events = gray2event_(output_this,gap=1) # [T, 2, H, W]
        events5 = gray2event_(output_this,gap=5) # [T, 2, H, W]

        root = 'saved/BALL'+str(NUM_OBJ)
        if not os.path.exists(root):
            os.makedirs(root)

        save_path = os.path.join(root, str(i).zfill(4))

        print(save_path)
        # pause()
        #output,centerss #B T H W, B T ball_num 2

        en,noise = add_noise(events, thresh=1 - 1e-3)

        # noise_num=np.sum(noise)
        # active_num=np.sum(events)
        # print('noise_num',noise_num)
        # print('active_num',active_num) # 27.7%
        # print(noise_num/active_num)

        # pause()

        events_binary = np.where(events>0,True,False)
        events5_binary = np.where(events5>0,True,False)
        en_binary = np.where(en>0,True,False)
        gray_binary = np.where(output_this > 0, True, False)
        # collision_binary = np.where(collision_ > 0, True, False)

        # print_info(events_binary,'events_binary')
        # print_info(en_binary,'en_binary')
        # print_info(collision_binary,'collision_binary')
        # print_info(yolo_target_binary,'yolo_target_binary')
        # pause()
        rgb_noise = event2rgb_(en)
        rgb = event2rgb_(events)
        rgb5 = event2rgb_(events5)

        # for j in range(rgb.shape[0]):
        #     # cv2.imshow(save_path + '/rgb/' + str(j).zfill(4) + '.png', rgb[j])
        #     # cv2.waitKey(1000)
        #     cv2.imwrite('XXX/' + str(i).zfill(3) + str(j).zfill(4) + '.png', rgb_noise[j])

        events_path=os.path.join(save_path,'events.npy')
        events5_path=os.path.join(save_path,'events5.npy')
        bb_path=os.path.join(save_path,'boundingboxes.npy')
        yolo_target_path=os.path.join(save_path,'yolo_target.npy')
        centers_path=os.path.join(save_path,'centers.npy')
        gray_path=os.path.join(save_path,'gray.npy')
        events_p=os.path.join(save_path,'en.npy')
        collision_binary_path = os.path.join(save_path, 'collision_binary.npy')

        # save = False
        save = True

        if save:

            if not os.path.exists(save_path):
                os.makedirs(save_path)
                os.makedirs(save_path+'/rgb')
                os.makedirs(save_path+'/rgb5')
                os.makedirs(save_path+'/rgb_noised')
                print('make ',save_path)
            with open(save_path + '/info.pkl', 'wb') as f:
                pickle.dump(info_dict, f)

            info_js = json.dumps(info_dict)
            with open(save_path + '/info.json', 'w') as f:
                json.dump(info_js, f)

            # print_info(yolo_target,'yolo_target')
            # print_info(events_binary,'events_binary')
            # print_info(boundingboxes,'boundingboxes')
            # print_info(gray_binary,'gray_binary')
            # print_info(en_binary,'en_binary')
            # print_info(collision_,'collision_')


            # 1. events   [T, 2, H, W]  bool
            np.save(events_path,events_binary)
            # 2. events   [T, 2, H, W]  bool
            np.save(events5_path,events5_binary)
            # 3. [T, 2, H, W]        bool
            np.save(gray_path,gray_binary)
            # 4. events_noise  [T, 2, H, W]  bool
            np.save(events_p,en_binary)
            # 5. collision_   [T]    bool
            np.save(collision_binary_path, collision_)
            # 6. boundingboxes [T, NUM_OBJ, 4]  float
            np.save(bb_path,boundingboxes)
            # 7. centers    [T, NUM_OBJ, 2]  int
            np.save(centers_path, centers)
            # 8. yolo_target  [T, 2, H, W]  float
            np.save(yolo_target_path,yolo_target)
            for j in range(rgb.shape[0]):
                cv2.imwrite(save_path+'/rgb/'+str(j).zfill(4)+'.png',rgb[j])
                cv2.imwrite(save_path+'/rgb_noised/'+str(j).zfill(4)+'.png',rgb_noise[j])

            for j in range(rgb5.shape[0]):
                cv2.imwrite(save_path + '/rgb5/' + str(5*j).zfill(4) + '.png', rgb5[j])

if __name__ == '__main__':
    main()

# import torch.utils.data as data

# from PIL import Image
# import os
# import os.path

# IMG_EXTENSIONS = [
    # '.jpg', '.JPG', '.jpeg', '.JPEG',
    # '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
# ]


# def is_image_file(filename):
    # return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

# def dataloader(filepath):

 # classes = [d for d in os.listdir(filepath) if os.path.isdir(os.path.join(filepath, d))]
 # image = [img for img in classes if img.find('frames_cleanpass') > -1]
 # disp  = [dsp for dsp in classes if dsp.find('disparity') > -1]

 # monkaa_path = filepath + [x for x in image if 'monkaa' in x][0]
 # monkaa_disp = filepath + [x for x in disp if 'monkaa' in x][0]

 
 # monkaa_dir  = os.listdir(monkaa_path)

 # all_left_img=[]
 # all_right_img=[]
 # all_left_disp = []
 # test_left_img=[]
 # test_right_img=[]
 # test_left_disp = []


 # for dd in monkaa_dir:
   # for im in os.listdir(monkaa_path+'/'+dd+'/left/'):
    # if is_image_file(monkaa_path+'/'+dd+'/left/'+im):
     # all_left_img.append(monkaa_path+'/'+dd+'/left/'+im)
     # all_left_disp.append(monkaa_disp+'/'+dd+'/left/'+im.split(".")[0]+'.pfm')

   # for im in os.listdir(monkaa_path+'/'+dd+'/right/'):
    # if is_image_file(monkaa_path+'/'+dd+'/right/'+im):
     # all_right_img.append(monkaa_path+'/'+dd+'/right/'+im)

 # flying_path = filepath + [x for x in image if x == 'frames_cleanpass'][0]
 # flying_disp = filepath + [x for x in disp if x == 'frames_disparity'][0]
 # flying_dir = flying_path+'/TRAIN/'
 # subdir = ['A','B','C']

 # for ss in subdir:
    # flying = os.listdir(flying_dir+ss)

    # for ff in flying:
      # imm_l = os.listdir(flying_dir+ss+'/'+ff+'/left/')
      # for im in imm_l:
       # if is_image_file(flying_dir+ss+'/'+ff+'/left/'+im):
         # all_left_img.append(flying_dir+ss+'/'+ff+'/left/'+im)

       # all_left_disp.append(flying_disp+'/TRAIN/'+ss+'/'+ff+'/left/'+im.split(".")[0]+'.pfm')

       # if is_image_file(flying_dir+ss+'/'+ff+'/right/'+im):
         # all_right_img.append(flying_dir+ss+'/'+ff+'/right/'+im)

 # flying_dir = flying_path+'/TEST/'

 # subdir = ['A','B','C']

 # for ss in subdir:
    # flying = os.listdir(flying_dir+ss)

    # for ff in flying:
      # imm_l = os.listdir(flying_dir+ss+'/'+ff+'/left/')
      # for im in imm_l:
       # if is_image_file(flying_dir+ss+'/'+ff+'/left/'+im):
         # test_left_img.append(flying_dir+ss+'/'+ff+'/left/'+im)

       # test_left_disp.append(flying_disp+'/TEST/'+ss+'/'+ff+'/left/'+im.split(".")[0]+'.pfm')

       # if is_image_file(flying_dir+ss+'/'+ff+'/right/'+im):
         # test_right_img.append(flying_dir+ss+'/'+ff+'/right/'+im)



 # driving_dir = filepath + [x for x in image if 'driving' in x][0] + '/'
 # driving_disp = filepath + [x for x in disp if 'driving' in x][0]

 # subdir1 = ['35mm_focallength','15mm_focallength']
 # subdir2 = ['scene_backwards','scene_forwards']
 # subdir3 = ['fast','slow']

 # for i in subdir1:
   # for j in subdir2:
    # for k in subdir3:
        # imm_l = os.listdir(driving_dir+i+'/'+j+'/'+k+'/left/')    
        # for im in imm_l:
          # if is_image_file(driving_dir+i+'/'+j+'/'+k+'/left/'+im):
            # all_left_img.append(driving_dir+i+'/'+j+'/'+k+'/left/'+im)
          # all_left_disp.append(driving_disp+'/'+i+'/'+j+'/'+k+'/left/'+im.split(".")[0]+'.pfm')

          # if is_image_file(driving_dir+i+'/'+j+'/'+k+'/right/'+im):
            # all_right_img.append(driving_dir+i+'/'+j+'/'+k+'/right/'+im)


 # return all_left_img, all_right_img, all_left_disp, test_left_img, test_right_img, test_left_disp

import torch.utils.data as data

from PIL import Image
import os
import os.path
from sklearn.model_selection import train_test_split

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]


def is_image_file(filename): 
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def dataloader(filepath):  # /media/hugonie/Hhome/dataset/SceneFlowData/


    all_left_img = []
    all_right_img = []
    all_left_disp = []
    test_left_img = []
    test_right_img = []
    test_left_disp = []
    train_left_img = []
    train_right_img = []
    train_left_disp = []

    
    # driving
    # driving_dir = filepath + [x for x in image if 'driving' in x][0] + '/'
    # driving_disp = filepath + [x for x in disp if 'driving' in x][0]
    driving_dir = filepath + '/frames_finalpass/driving/'
    driving_disp = filepath + '/disparity/driving'

    subdir1 = ['15mm_focallength', '35mm_focallength']
    subdir2 = ['scene_backwards', 'scene_forwards']
    subdir3 = ['fast', 'slow']

    for i in subdir1:
        for j in subdir2:
            for k in subdir3:
                imm_l = os.listdir(driving_dir + i + '/' + j + '/' + k + '/left/')
                for im in imm_l:
                    if is_image_file(driving_dir + i + '/' + j + '/' + k + '/left/' + im):
                        all_left_img.append(driving_dir + i + '/' + j + '/' + k + '/left/' + im)
                    all_left_disp.append(
                        driving_disp + '/' + i + '/' + j + '/' + k + '/left/' + im.split(".")[0] + '.pfm')

                    if is_image_file(driving_dir + i + '/' + j + '/' + k + '/right/' + im):
                        all_right_img.append(driving_dir + i + '/' + j + '/' + k + '/right/' + im)
    train_left_img,test_left_img,train_right_img,test_right_img,train_left_disp,test_left_disp=train_test_split(all_left_img,all_right_img,all_left_disp,test_size=0.1,random_state=42)

    train_left_img.sort()
    train_right_img.sort()
    train_left_disp.sort()

    test_left_img.sort()
    test_right_img.sort()
    test_left_disp.sort()
    
    file=open('sceneflow_test.txt','w')
    for l,r,d in zip(test_left_img,test_right_img,test_left_disp):
        stri=[str(l)+' ',str(r)+' ',str(d)+' ']
        file.writelines(stri)
        file.write('\n')
        # file.seed(0,0)
        # for lines in file.readlines():
            # print(lines)
        # file.write('\v')
        # file.write(str(r))
        # file.write('\v')
        # file.write(str(d))
        # file.write('\n')
    file.close()
    '''
    file=open('sceneflow_train.txt','w')
    for l,r,d in zip(train_left_img,train_right_img,train_left_disp):
        stri=[str(l)+' ',str(r)+' ',str(d)+' ']
        file.writelines(stri)
        file.write('\n')
        # file.seed(0,0)
        # for lines in file.readlines():
            # print(lines)
        # file.write('\v')
        # file.write(str(r))
        # file.write('\v')
        # file.write(str(d))
        # file.write('\n')
    file.close()
    # return all_left_img, all_right_img, all_left_disp, test_left_img, test_right_img, test_left_disp
    '''
    print(len(test_left_img))
    return train_left_img, train_right_img, train_left_disp, test_left_img, test_right_img, test_left_disp

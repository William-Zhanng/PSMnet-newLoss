from __future__ import print_function
import argparse
import os
import sys

sys.path.append('./dataloader')
sys.path.append('./models')
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import time
import math
# KITTI用
# from dataloader import KITTIloader2015_pri as lt 
from dataloader import KITTILoader as DA

# 三选一
# from dataloader import listflowfile as lt
# from dataloader import listflowfile_flying_test as lt
# from dataloader import listflowfile_monkaa_sceneflow as lt
# from dataloader import listflowfile_only_monkaa as lt

from dataloader import readlist as lt
# from dataloader import SecenFlowLoader as DA

from models import *
import cv2 as cv
from os.path import join, split, isdir, isfile, splitext, split, abspath, dirname

parser = argparse.ArgumentParser(description='PSMNet')
parser.add_argument('--maxdisp', type=int, default=192,
                    help='maxium disparity')
parser.add_argument('--model', default='stackhourglass',
                    help='select model')
# parser.add_argument('--datapath', default='/data/yyx/data/sceneflow',help='datapath')
parser.add_argument('--datapath', default='.', help='datapath')
# parser.add_argument('--datapath', default='/data/yyx/data/kitti/2015/data_scene_flow/training', help='datapath')
parser.add_argument('--trainlist', default='/data/yyx/zwt/PSMNet-master/PSMNet-master/filename/sceneflow_test.txt',
                    help='training list')
parser.add_argument('--testlist', default='/data/yyx/zwt/PSMNet-master/PSMNet-master/filename/kitti_list.txt',
                    help='testing list')
# parser.add_argument('--testlist', default='/data/yyx/StereoNet-master_pri/filenames/monkaa_test_list.txt', help='testing list')
# parser.add_argument('--testlist', default='/data/yyx/zwt/PSMNet-master/PSMNet-master/filename/kitti_test.txt', help='testing list')

parser.add_argument('--epochs', type=int, default=55,
                    help='number of epochs to train')
# parser.add_argument('--loadmodel', default= None,
# help='load model')
# parser.add_argument('--loadmodel', default= './result_320/checkpoint_55.tar',
#                     help='load model')
parser.add_argument('--loadmodel', default='./pretrained_sceneflow.tar',
                    help='load model')
parser.add_argument('--save_path', default='./pretrain_result',
                    help='save path')
# parser.add_argument('--save_path', default= './new_loss_result',

# help='save path')
parser.add_argument('--savemodel', default='./',
                    help='save model')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

# set gpu id used
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

# all_left_img, all_right_img, all_left_disp, test_left_img, test_right_img, test_left_disp = lt.dataloader(args.datapath)

all_left_img, all_right_img, all_left_disp = lt.dataloader(args.trainlist)
test_left_img, test_right_img, test_left_disp = lt.dataloader(args.testlist)
print('len of train', len(all_left_img))
print('len of test', len(test_left_img))

TrainImgLoader = torch.utils.data.DataLoader(
    DA.myImageFloder(all_left_img, all_right_img, all_left_disp, True, args.datapath),
    batch_size=1, shuffle=True, num_workers=8, drop_last=False)

TestImgLoader = torch.utils.data.DataLoader(
    DA.myImageFloder(test_left_img, test_right_img, test_left_disp, False, args.datapath),
    batch_size=1, shuffle=False, num_workers=4, drop_last=False)

if args.model == 'stackhourglass':
    model = stackhourglass(args.maxdisp)
elif args.model == 'basic':
    model = basic(args.maxdisp)
else:
    print('no model')

if args.cuda:
    model = nn.DataParallel(model)
    model.cuda()

if args.loadmodel is not None:
    state_dict = torch.load(args.loadmodel)
    model.load_state_dict(state_dict['state_dict'])
    print("!!!!!!!!!!!!!")

print('Number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))

optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999))

'''
保存路径设置
'''

# driving and monkaa test
# test_gt_path=args.save_path+'/results320_drivingmonkaa_test_gt/'
# test_imgL_path=args.save_path+'/results320_drivingmonkaa_test_imgL/'
# the_gtsavedir_list =[test_gt_path,test_imgL_path]
# test_pred1_path=args.save_path+'/results320_drivingmonkaa_test_pred1/'
# test_pred2_path=args.save_path+'/results320_drivingmonkaa_test_pred2/'
# test_pred3_path=args.save_path+'/results320_drivingmonkaa_test_pred3/'
# train_pred3_path=args.save_path+'/driving_train_pred3_320/'
# test_path_dirs = [test_pred1_path,test_pred2_path,test_pred3_path,train_pred3_path]
# drving test
test_gt_path = args.save_path + '/results320_driving_test_gt/'
test_imgL_path = args.save_path + '/results320_driving_test_imgL/'
the_gtsavedir_list = [test_gt_path, test_imgL_path]
test_pred1_path = args.save_path + '/results320_driving_test_pred1/'
test_pred2_path = args.save_path + '/results320_driving_test_pred2/'
test_pred3_path = args.save_path + '/kitti_test_pred/'
train_pred3_path = args.save_path + '/driving_train_pred3_320/'
test_path_dirs = [test_pred1_path, test_pred2_path, test_pred3_path, train_pred3_path]

# KITTI test
# test_gt_path=args.save_path+'/KITTI_test_gt/'
# test_imgL_path=args.save_path+'/KITTI_test_imgL/'
# the_gtsavedir_list =[test_gt_path,test_imgL_path]
# test_pred3_path=args.save_path+'/kitti_test_pred/'
# test_path_dirs = [test_pred3_path]
for i in the_gtsavedir_list:
    if not os.path.exists(i):
        os.makedirs(i)

        '''
        存储路径设置

        '''
    # driving final version

for dir in test_path_dirs:
    if not os.path.exists(dir):
        os.makedirs(dir)


def train(imgL, imgR, disp_L, bx):
    model.train()
    imgL = Variable(torch.FloatTensor(imgL))
    imgR = Variable(torch.FloatTensor(imgR))
    disp_L = Variable(torch.FloatTensor(disp_L))

    if args.cuda:
        imgL, imgR, disp_true = imgL.cuda(), imgR.cuda(), disp_L.cuda()

    # ---------
    mask = disp_true < args.maxdisp
    mask.detach_()
    # ----
    optimizer.zero_grad()

    if args.model == 'stackhourglass':
        output1, output2, output3 = model(imgL, imgR)
        output1 = torch.squeeze(output1, 1)
        output2 = torch.squeeze(output2, 1)
        output3 = torch.squeeze(output3, 1)
        if (bx % 100 == 0):
            im_pred3 = output3[0, :, :].detach().cpu().numpy()
            cv.imwrite(join(train_pred3_path, "driving_train_pred3-%04d.png" % bx), im_pred3)
        loss = 0.5 * F.smooth_l1_loss(output1[mask], disp_true[mask], size_average=True) + 0.7 * F.smooth_l1_loss(
            output2[mask], disp_true[mask], size_average=True) + F.smooth_l1_loss(output3[mask], disp_true[mask],
                                                                                  size_average=True)
    elif args.model == 'basic':
        output = model(imgL, imgR)
        output = torch.squeeze(output, 1)
        loss = F.smooth_l1_loss(output[mask], disp_true[mask], size_average=True)

    loss.backward()
    optimizer.step()

    return loss.item()


def adjust_learning_rate(optimizer, epoch):
    lr = 0.001
    print(lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def test(imgL, imgR, disp_true, bx):
    model.eval()
    imgL = Variable(torch.FloatTensor(imgL))

    imgR = Variable(torch.FloatTensor(imgR))
    if args.cuda:
        imgL, imgR = imgL.cuda(), imgR.cuda()

    # ---------
    mask = (disp_true < 192) & (disp_true > 0)
    # mask = (disp_true < 192)
    # ----

    with torch.no_grad():
        output3 = model(imgL, imgR)
    # 裁剪，KITTI用
    output = torch.squeeze(output3.data.cpu(), 1)[:, :, :] * 1.17
    # 裁剪，driving用
    # output = torch.squeeze(output3.data.cpu(),1)[:,4:,:]*1.17
    im_pred3 = np.array(output[0, :, :].cpu() * 256, dtype=np.uint16)
    cv.imwrite(join(test_pred3_path, "kitti_test_pred-%04d.png" % bx), im_pred3)
    im = np.array(imgL[0, :, :, :].permute(1, 2, 0).cpu() * 255, dtype=np.uint8)
    cv.imwrite(join(test_pred2_path, "flying-colorimg-%04d.jpg" % bx), im)
    # cv.imwrite(join(test_pred3_path, "flying_test_pred3-%d.png" % bx),im_pred3)
    # print("disp_true",disp_true[mask])
    if len(disp_true[mask]) == 0:
        loss1 = 0.0
        loss2 = 0.0
        loss3 = 0.0
    else:
        loss1 = torch.mean(torch.abs(output[mask] - disp_true[mask]))  # end-point-error
        loss2 = Thres_metric(output, disp_true, mask, 1.0)
        loss3 = Thres_metric(output, disp_true, mask, 3.0)
    return loss1, loss2, loss3


def Thres_metric(D_est, D_gt, mask, thres):
    assert isinstance(thres, (int, float))
    D_est, D_gt = D_est[mask], D_gt[mask]
    E = torch.abs(D_gt - D_est)
    err_mask = (E > thres) & (E / D_gt.abs() > 0.05)
    return torch.mean(err_mask.float())


def main():
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    # ---------------------------train--------------------------------------

    # start_full_time = time.time()
    # for epoch in range(48, args.epochs+1):
    #     print('This is %d-th epoch' %(epoch))
    #     total_train_loss = 0
    #     adjust_learning_rate(optimizer,epoch)
    #     wrong_num = 0

    #     for batch_idx, (imgL_crop, imgR_crop, disp_crop_L) in enumerate(TrainImgLoader):
    #         start_time = time.time()

    #         loss = train(imgL_crop,imgR_crop, disp_crop_L,batch_idx)
    #         if math.isnan(loss):
    #             wrong_num += 1
    #         else:
    #             print('Iter %d training loss = %.3f , time = %.2f' %(batch_idx, loss, time.time() - start_time))
    #             if(batch_idx %1000 == 0):
    #                 savefilename = args.save_path+'/checkpoint_'+str(epoch)+'.tar'
    #                 torch.save({
    #                 'epoch': epoch,
    #                 'state_dict': model.state_dict(),
    #                 'train_loss': total_train_loss/len(TrainImgLoader),
    #                  }, savefilename)
    #             total_train_loss += loss
    #     total_num = len(TrainImgLoader)-wrong_num
    #     print('epoch %d total training loss = %.3f' %(epoch, total_train_loss/total_num))

    #     savefilename = args.save_path+'/checkpoint_'+str(epoch)+'.tar'
    #     torch.save({
    #         'epoch': epoch,
    #     'state_dict': model.state_dict(),
    #                 'train_loss': total_train_loss/len(TrainImgLoader),
    #     }, savefilename)

    # print('full training time = %.2f HR' %((time.time() - start_full_time)/3600))

    # ------------- TEST ------------------------------------------------------------
    total_test_loss1 = 0
    total_test_loss2 = 0
    total_test_loss3 = 0
    for batch_idx, (imgL, imgR, disp_L) in enumerate(TestImgLoader):
        # driving
        # 保存源左图
        # print("disp_L",disp_L[0,959])
        # print("disp2",disp_L[0,511,955:959])
        # im_imgL = np.array(imgL[0,:,:,:].permute(1,2,0).cpu()*2.0, dtype=np.uint8)
        # cv.imwrite(join(test_imgL_path, "driving_test_imgL-%d.jpg" % batch_idx),im_imgL)
        # #cv.imwrite(join(test_imgL_path, "KITTI_test_imgL-%d.jpg" % batch_idx),im_imgL)
        # _,H, W = disp_L.shape
        # im_gt = torch.zeros((H, W))

        # im_gt = np.array(disp_L[0,:,:].cpu()*256, dtype=np.uint16)
        # # print(im_gt[511,955:959])
        # #print(im_gt.dtype)
        # #保存grundtruth深度图
        # cv.imwrite(join(test_gt_path, "drving_test_gt_%d.png" % batch_idx),im_gt)
        # #cv.imwrite(join(test_gt_path, "KITTI_test_gt_%d.png" % batch_idx),im_gt)
        test_loss1, test_loss2, test_loss3 = test(imgL, imgR, disp_L, batch_idx)
        print('Iter %d test loss = %.3f' % (batch_idx, test_loss1))
        total_test_loss1 += test_loss1
        total_test_loss2 += test_loss2
        total_test_loss3 += test_loss3

    print('total test epe = %.4f' % (total_test_loss1 / len(TestImgLoader)))
    print('total test d1 = %.4f' % (total_test_loss2 / len(TestImgLoader)))
    print('total test d3 = %.4f' % (total_test_loss3 / len(TestImgLoader)))
    # ----------------------------------------------------------------------------------
    # SAVE test information
    savefilename = args.savemodel + 'testinformation.tar'
    torch.save({
        'test_loss': total_test_loss1 / len(TestImgLoader),
    }, savefilename)


if __name__ == '__main__':
    main()


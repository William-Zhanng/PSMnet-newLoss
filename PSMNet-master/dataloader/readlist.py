import torch.utils.data as data

from PIL import Image
import os
import os.path
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from PIL import Image
import numpy as np

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]


def is_image_file(filename): 
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def read_all_lines(filename):
    with open(filename) as f:
        lines = [line.rstrip() for line in f.readlines()]
    return lines

def dataloader(filelist):  
    lines = read_all_lines(filelist)
    splits = [line.split() for line in lines]
    left_images = [x[0] for x in splits]
    #left_images=os.path.join(datapath, left_images)
    right_images = [x[1] for x in splits]
    #right_images=os.path.join(datapath, right_images)
    if len(splits[0]) == 2:  # ground truth not available
        return left_images, right_images, None
    else:
        disp_images = [x[2] for x in splits]
        #disp_images=os.path.join(datapath, disp_images)
        return left_images, right_images, disp_images
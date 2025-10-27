import cv2
import glob
import numpy as np
import os
import pandas as pd
import re
from torch.utils.data import Dataset
from itertools import compress
from annotator.hed import HEDdetector
from annotator.util import HWC3
import torch

#from decimal import Decimal

device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

class MyDataset(Dataset):
    def __init__(self, image_dir, data_dir, hint_dir):

        self.image_dir = image_dir
        self.data_dir = data_dir
        self.hint_dir = hint_dir

        df = pd.DataFrame() 
        for f in self.data_dir: 
            dfi = pd.read_csv(f)
            df = pd.concat([df, dfi[(dfi['Train']==1)&(dfi['img']==1)]])
            #df = pd.concat([df, dfi[(dfi['Train']==1)&(dfi['img']==1)&(dfi['withBuilding']==1)]])


        self.description = df   
        self.image_city = self.description['city'].to_list()

        self.image_list_xtile = self.description['row'].to_list()
        self.image_list_ytile = self.description['col'].to_list()

        self.image_list_right = self.description['r'].to_list()
        self.image_list_down = self.description['d'].to_list()
        #self.image_list_rotate = self.description['rt'].to_list()

        self.description = self.description['descriptions_num_gc'].to_list()
        
    def __len__(self):
        return len(self.image_list_xtile)

    def __getitem__(self, item):

        xtile = self.image_list_xtile[item]
        ytile = self.image_list_ytile[item]
        r = self.image_list_right[item]
        d = self.image_list_down[item]
        #rt = self.image_list_rotate[item]
        city = self.image_city[item]
        #city = self.image_city
        
        hint_name = self.image_dir + f'{city}_bg_Stage_2_{r}_{d}/{str(xtile)}_{str(ytile)}_{r}_{d}.tif'
        img_name =  self.image_dir + f'{city}_gc_Stage_3_{r}_{d}/{str(xtile)}_{str(ytile)}_{r}_{d}.tif'
        #hint_name = self.image_dir + city + '_4lu_Stage_1_' + r + '_' + d + '/' + str(xtile) + '_' + str(ytile) + '_' + r + '_' + d + '.tif'
        #img_name =  self.image_dir + city + '_4lu_Stage_2_' + r + '_' + d + '/' + str(xtile) + '_' + str(ytile) + '_' + r + '_' + d + '.tif'     

        #print('-----img_name-----', img_name)
        #print('-----hint_name-----', hint_name)

        target = cv2.imread(img_name)
        source = cv2.imread(hint_name, cv2.IMREAD_UNCHANGED)
        if source is None:
            print(hint_name)
        
        if source.shape[2] == 4:
            # convert 4-channel source image to 3-channel
            #make mask of where the transparent bits are
            trans_mask = source[:,:,3] == 0

            #replace areas of transparency with white and not transparent
            source[trans_mask] = [255, 255, 255, 255]

            #new image without alpha channel...
            source = cv2.cvtColor(source, cv2.COLOR_BGRA2BGR)

        #OpenCV read images in BGR order.
        source = cv2.cvtColor(source, cv2.COLOR_BGR2RGB)
        target = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)

        # Normalize source images to [0, 1].
        source = source.astype(np.float32) / 255.0

        # Normalize target images to [-1, 1].
        target = (target.astype(np.float32) / 127.5) - 1.0

        prompt = self.description[item]
        
        # to(device)
        target = torch.tensor(target).to(device)
        source = torch.tensor(source).to(device)
        
        return dict(jpg=target, txt=prompt, hint=source)


# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Routine for decoding the CIFAR-10 binary file format."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
import numpy as np
from PIL import Image
import random
import re
import csv
# Process images of this size. Note that this differs from the original CIFAR
# image size of 32 x 32. If one alters this number, then the entire model
# architecture will change and any model would need to be retrained.
IMAGE_SIZE = 128
# Global constants describing the CIFAR-10 data set.
EYE_H = 40; EYE_W = 40;
NOSE_H = 32; NOSE_W = 40;
MOUTH_H = 32; MOUTH_W = 48;
re_pose = re.compile('_\d{3}_')
re_poseIllum = re.compile('_\d{3}_\d{2}_')
#眼嘴鼻的高宽都是固定的,如上

class MultiPIE():
    """Reads and parses examples from MultiPIE data filelist
    """
    def __init__(self, datasplit='train', Random=True, LOAD_60_LABEL=False, MIRROR_TO_ONE_SIDE=True, RANDOM_VERIFY=False,
                 GENERATE_MASK=False, source='without90', testing = False):
        self.testing = testing
        self.split = datasplit
        self.random = Random
        self.seed = None
        self.LOAD_60_LABEL = LOAD_60_LABEL
        self.MIRROR_TO_ONE_SIDE = MIRROR_TO_ONE_SIDE
        self.RANDOM_VERIFY = RANDOM_VERIFY
        self.GENERATE_MASK = GENERATE_MASK
        if not testing:
            #split_f = '/home/TP-GAN/data/train.csv'
            #split_f_test = '/home/TP-GAN/data/test.csv'
            #self.indices是图片的名称大全,原本是csv文件里读取的
            #self.indices = open(split_f, 'r').read().splitlines()
            #self.indices_test = open(split_f_test, 'r').read().splitlines()
            #self.size = len(self.indices)
            #self.test_size = len(self.indices_test)
            # make eval deterministic
            if 'train' not in self.split:
                self.random = False

        else:#only load test images for a separate list file.
            split_f_test = '/home/TP-GAN/data/test.csv'
            self.indices_test = open(split_f_test, 'r').read().splitlines()
            self.size = 0
            self.test_size = len(self.indices_test)
        self.idx = 0


    def test_batch(self, test_batch_size=100,Random = True, Pose = -1):
        test_batch_size = min(test_batch_size, len(self.indices_test))
        images = np.empty([test_batch_size, IMAGE_SIZE, IMAGE_SIZE, 3])
        eyel = np.empty([test_batch_size, EYE_H, EYE_W, 3], dtype=np.float32)
        eyer = np.empty([test_batch_size, EYE_H, EYE_W, 3], dtype=np.float32)
        nose = np.empty([test_batch_size, NOSE_H, NOSE_W, 3], dtype=np.float32)
        mouth = np.empty([test_batch_size, MOUTH_H, MOUTH_W, 3],dtype=np.float32)
        if not self.testing:
            idenlabels = np.empty([test_batch_size], dtype=np.int32)
            leyel = np.empty([test_batch_size, EYE_H, EYE_W, 3], dtype=np.float32)
            leyer = np.empty([test_batch_size, EYE_H, EYE_W, 3], dtype=np.float32)
            lnose = np.empty([test_batch_size, NOSE_H, NOSE_W, 3], dtype=np.float32)
            lmouth = np.empty([test_batch_size, MOUTH_H, MOUTH_W, 3],dtype=np.float32)
            labels = np.empty([test_batch_size, IMAGE_SIZE, IMAGE_SIZE, 3])
        filenames = list()
        if Random:
            random.seed(2017)#make testing batch deterministic
            random.shuffle(self.indices_test)
            #resume randomeness for training
            random.seed(self.seed)
        #这个是什么?LOAD_60_LABEL,TODO:如果LOAD_60_LABEL，则跳过小于45度的图片；Pose为指定角度
        j = 0
        for i in range(test_batch_size):
            print(j, end=' ')
            images[i, ...], feats = self.load_image(self.indices_test[j % len(self.indices_test)])
            eyel[i,...] = feats[1]
            eyer[i,...] = feats[2]
            nose[i,...] = feats[3]
            mouth[i, ...] = feats[4]
            filename = self.indices_test[j % len(self.indices_test)]
            filenames.append(filename)
            if not self.testing:
                labels[i,...], _, leyel[i,...], leyer[i,...], lnose[i,...], lmouth[i, ...] = self.load_label_mask(filename)
                identity = int(filename[0:3])
                idenlabels[i] = identity
            j += 1
        print('\n')
        if not self.testing:
            #labels应该是正面的图片,leyel是label_eye_left
            return images, filenames, eyel, eyer, nose, mouth,\
            labels, leyel, leyer, lnose, lmouth, idenlabels
        else:
            return images, filenames, eyel, eyer, nose, mouth, None, None, None, None, None, None



    def load_image(self, filename):
        #读取一个图片
            """
            Load input image & codemap and preprocess:
            - cast to float
            - subtract mean divide stdadv
            - concatenate together
            """
            im = Image.open(filename)
            in_ = np.array(im, dtype=np.float32)
            in_ /= 256
            features = self.GetFeatureParts(in_, filename)
            return in_, features


    #input 图片路径,return 监督图片,TODO:这个要改
    def findSameIllumCodeLabelpath(self, fullpath):
        #输入路径，提取路径中某个东西
        labelpath = fullpath.replace('cropped','cropped_test')
        codepath='_'
        return (codepath, labelpath)
        
    #这个函数意思是
    def findPose(self, fullpath):
        return +60
        
    ##输入img_resize是128x128的Image.open,filename是用来寻找标注文件;返回trans_points和图片各个部位的裁剪
    def GetFeatureParts(self, img_resize, filename, label=False):
        #crop four parts
        trans_points = np.empty([5,2],dtype=np.int32)
        #把这里改成mtcnn
        if True:
            path_5pt='./test_tem.5pt'
            featpath = path_5pt
            with open(featpath, 'r') as csvfile:
                reader = csv.reader(csvfile, delimiter=' ')
                for ind,row in enumerate(reader):
                    if not ind >=5:
                        trans_points[ind,:] = row 
        #把这里改成mtcnn
        eyel_crop = np.zeros([EYE_H,EYE_W,3], dtype=np.float32);
        crop_y = int(trans_points[0,1] - EYE_H / 2);
        crop_y_end = crop_y + EYE_H;
        crop_x = int(trans_points[0,0] - EYE_W / 2);
        crop_x_end = crop_x + EYE_W;
        
        eyel_crop[...] = img_resize[crop_y:crop_y_end,crop_x:crop_x_end,:];
        ##########################################################
        eyer_crop = np.zeros([EYE_H,EYE_W,3], dtype=np.float32);
        crop_y = int(trans_points[1,1] - EYE_H / 2)
        crop_y_end = crop_y + EYE_H;
        crop_x = int(trans_points[1,0] - EYE_W / 2);
        crop_x_end = crop_x + EYE_W;
        eyer_crop[...] = img_resize[crop_y:crop_y_end,crop_x:crop_x_end,:];
        #####################################################
        month_crop = np.zeros([MOUTH_H,MOUTH_W,3], dtype=np.float32);
        crop_y = int((trans_points[3,1] + trans_points[4,1]) // 2 - MOUTH_H / 2);
        crop_y_end = crop_y + MOUTH_H;
        crop_x = int((trans_points[3,0] + trans_points[4,0]) // 2 - MOUTH_W / 2);
        crop_x_end = crop_x + MOUTH_W;
        month_crop[...] = img_resize[crop_y:crop_y_end,crop_x:crop_x_end,:];
        ##########################################################
        nose_crop = np.zeros([NOSE_H,NOSE_W,3], dtype=np.float32);
        crop_y_end = int(crop_y_end)
        crop_x = int(crop_x)
        crop_y = crop_y_end - NOSE_H;
        crop_x_end = crop_x + NOSE_W;
        #import pdb; pdb.set_trace()
        nose_crop[...] = img_resize[crop_y:crop_y_end,crop_x:crop_x_end,:];

        if not label and self.MIRROR_TO_ONE_SIDE:
            teml = eyel_crop[:,::-1,:]
            eyel_crop = eyer_crop[:,::-1,:]
            eyer_crop = teml
            month_crop = month_crop[:,::-1,:]
            nose_crop = nose_crop[:,::-1,:]
            trans_points[:,0] = IMAGE_SIZE - trans_points[:,0]
            #exchange eyes and months
            teml = trans_points[0,:].copy()
            trans_points[0, :] = trans_points[1, :]
            trans_points[1, :] = teml
            teml = trans_points[3,:].copy()
            trans_points[3, :] = trans_points[4, :]
            trans_points[4, :] = teml
        
        return trans_points, eyel_crop, eyer_crop, nose_crop, month_crop

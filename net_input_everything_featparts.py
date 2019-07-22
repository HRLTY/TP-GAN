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

class MultiPIE():
    """Reads and parses examples from MultiPIE data filelist
    """
    def __init__(self, datasplit='train', Random=True, LOAD_60_LABEL=False, MIRROR_TO_ONE_SIDE=True, RANDOM_VERIFY=False,
                 GENERATE_MASK=False, source='without90', testing = False):
        self.dir = '/home/ruihuang/data/FS_aligned/'
        self.csvpath = '/home/ruihuang/data/{}/{}.csv'
        self.feat5ptDir = '/home/shu.zhang/ruihuang/data/FS_t5pt/'
        self.test_dir = '/home/shu.zhang/ruihuang/data/testlist/FS/{}.csv'
        self.testing = testing

        self.split = datasplit
        self.random = Random
        self.seed = None
        self.LOAD_60_LABEL = LOAD_60_LABEL
        self.MIRROR_TO_ONE_SIDE = MIRROR_TO_ONE_SIDE
        self.RANDOM_VERIFY = RANDOM_VERIFY
        self.GENERATE_MASK = GENERATE_MASK
        #key->value第一个是角度,第二个是光线最好的编号
        self.cameraPositions = {'24_0': (+90, '10'),'01_0' : (+75, '08'), '20_0' : (+60, '08'), '19_0' : (+45, '09'), '04_1' : (+30, '07'), '05_0' : (+15, '06'), #left
                    '05_1' : (0,'06'), #center
                    '14_0' : (-15,'06'), '13_0' : (-30, '05'), '08_0' : (-45, '15'),'09_0' : (-60, '15'),'12_0' : (-75, '15'),'11_0' : (-90, '15')} #right

        if not testing:
            split_f = self.csvpath.format(source, self.split)
            split_f_test = self.csvpath.format(source, 'test')
            self.indices = open(split_f, 'r').read().splitlines()
            self.indices_test = open(split_f_test, 'r').read().splitlines()
            self.size = len(self.indices)
            self.test_size = len(self.indices_test)
            # make eval deterministic
            if 'train' not in self.split:
                self.random = False
            # randomization: seed and pick
            if self.random:
                random.seed(self.seed)
                self.idx = random.randint(0, len(self.indices)-1)
        else:#only load test images for a separate list file.
            split_f_test = self.test_dir.format(source)
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
        j = 0
        for i in range(test_batch_size):
            if self.LOAD_60_LABEL:
                while True:
                    pose = abs(self.findPose(self.indices_test[j % len(self.indices_test)]))
                    if pose >= 45:
                            break
                    j += 1
            if Pose != -1:
                while True:
                    pose = abs(self.findPose(self.indices_test[j % len(self.indices_test)]))
                    if pose == Pose:
                        break
                    j += 1
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
            return images, filenames, eyel, eyer, nose, mouth,\
            labels, leyel, leyer, lnose, lmouth, idenlabels
        else:
            return images, filenames, eyel, eyer, nose, mouth, None, None, None, None, None, None

    def next_image_and_label_mask_batch(self, batch_size, imageRange=-1,imageRangeLow = 0, labelnum=None):
        """Construct a batch of images and labels masks.

        Args:
        batch_size: Number of images per batch.
        shuffle: boolean indicating whether to use a shuffling queue.
        Returns:
        ndarray feed.
        images: Images. 4D of [batch_size, height, width, 6] size.
        labels: Labels. 4D of [batch_size, height, width, 3] size.
        masks: masks. 4D of [batch_size, height, width, 3] size.
        verifyImages: Images. 4D of [batch_size, height, width, 3] size.
        verifyLabels: 1D of [batch_size] 0 / 1 classification label
        """
        assert batch_size >= 1
        images = np.empty([batch_size, IMAGE_SIZE, IMAGE_SIZE, 3])
        labels = np.empty([batch_size, IMAGE_SIZE, IMAGE_SIZE, 3])

        poselabels = np.empty([batch_size],dtype=np.int32)
        idenlabels = np.empty([batch_size],dtype=np.int32)
        landmarklabels = np.empty([batch_size, 5*2],dtype=np.float32)
        eyel = np.empty([batch_size, EYE_H, EYE_W, 3], dtype=np.float32)
        eyer = np.empty([batch_size, EYE_H, EYE_W, 3], dtype=np.float32)
        nose = np.empty([batch_size, NOSE_H, NOSE_W, 3], dtype=np.float32)
        mouth = np.empty([batch_size, MOUTH_H, MOUTH_W, 3],dtype=np.float32)
        leyel = np.empty([batch_size, EYE_H, EYE_W, 3], dtype=np.float32)
        leyer = np.empty([batch_size, EYE_H, EYE_W, 3], dtype=np.float32)
        lnose = np.empty([batch_size, NOSE_H, NOSE_W, 3], dtype=np.float32)
        lmouth = np.empty([batch_size, MOUTH_H, MOUTH_W, 3],dtype=np.float32)

        if self.GENERATE_MASK:
            masks = np.empty([batch_size, IMAGE_SIZE, IMAGE_SIZE, 3])
        else:
            masks = None
            
        #verifyImages
        if self.RANDOM_VERIFY:
            verifyImages = np.empty([batch_size, IMAGE_SIZE, IMAGE_SIZE, 3])
            #verifyLabels = np.empty([batch_size, 1], dtype=np.float32)
            verifyLabels = np.empty([batch_size], dtype=np.int32)
        else:
            verifyImages = None; verifyLabels = None
            
        #循环一个batchsize,返回的结果也都是一个batchsize的
        for i in range(batch_size):
            if imageRange != -1:
                while True:
                    pose = abs(self.findPose(self.indices[self.idx]))
                    if not self.LOAD_60_LABEL:
                        if  pose <= imageRange and pose >= imageRangeLow:
                            break
                    else:
                        if pose <= imageRange and pose >= 45:
                            break
                    self.updateidx()
            images[i, ...], feats = self.load_image(self.indices[self.idx])
            filename = self.indices[self.idx]
            labels[i,...], _, leyel[i,...], leyer[i,...], lnose[i,...], lmouth[i, ...] = self.load_label_mask(filename)
            #poselabel是图片对应摄像机的角度
            pose = abs(self.findPose(filename))
            poselabels[i] = int(pose/15)
            identity = int(filename[0:3])
            #返回的idenlabel由文件名的1到4位定义
            idenlabels[i] = identity
            landmarklabels[i,:] = feats[0].flatten()
            eyel[i,...] = feats[1]
            eyer[i,...] = feats[2]
            nose[i,...] = feats[3]
            mouth[i, ...] = feats[4]
            self.updateidx()
       
        return images, labels, masks, verifyImages, verifyLabels, poselabels, idenlabels, landmarklabels,\
               eyel, eyer, nose, mouth, leyel, leyer, lnose, lmouth

    def updateidx(self):
        if self.random:
            self.idx = random.randint(0, len(self.indices)-1)
        else:
            self.idx += 1
            if self.idx == len(self.indices):
                self.idx = 0
    def load_image(self, filename):
            """
            Load input image & codemap and preprocess:
            - cast to float
            - subtract mean divide stdadv
            - concatenate together
            """
            im = Image.open(self.dir + filename)
            in_ = np.array(im, dtype=np.float32)
            in_ /= 256
            features = self.GetFeatureParts(in_, filename)
            if self.MIRROR_TO_ONE_SIDE and self.findPose(filename) < 0:
                in_ = in_[:,::-1,:]
            return in_, features

    def load_label_mask(self, filename, labelnum=-1):

        _, labelname = self.findSameIllumCodeLabelpath(filename)
        im = Image.open(self.dir + labelname)
        if self.MIRROR_TO_ONE_SIDE and self.findPose(labelname) < 0:
            im = im.transpose(Image.FLIP_LEFT_RIGHT)
            if self.GENERATE_MASK:
                codeImg = codeImg.transpose(Image.FLIP_LEFT_RIGHT)

        if self.GENERATE_MASK:
            code = np.array(codeImg, dtype=np.float32)
            mask = (code != np.zeros(code.shape, code.dtype))
            mask = mask.astype(np.float32, copy=False)
            RATIO = 0.1
            mask *= 1-RATIO
            mask += RATIO
        else:
            mask = None

        label = np.array(im, dtype=np.float32)
        #label -= label.mean()
        #label /= label.std()
        label /= 256
        #label -= 1
        feats = self.GetFeatureParts(label, labelname, label=True)
        if not self.LOAD_60_LABEL:
            if self.findPose(filename) < 0:#random.random() < 0.5:
                #print("fipping!")
                label = label[:,::-1,:]
                feats[1][...] = feats[1][:,::-1,:]
                feats[2][...] = feats[2][:,::-1,:]
                feats[3][...] = feats[3][:,::-1,:]
                feats[4][...] = feats[4][:,::-1,:]
                return label, mask, feats[2], feats[1], feats[3], feats[4]
        #print("not flipping!")
        return label, mask, feats[1], feats[2], feats[3], feats[4]
        #use coler code to generate mask
        #background area weights 0.2, face area weights 1.0

        return label, mask, feats

    def load_label_mask_with_veri(self, filename, labelnum=-1):
        label, mask = self.load_label_mask(filename, labelnum)

        #if(random.random() > 0.5):#positive
        if self.RANDOM_VERIFY:
            if True:
                return label, mask, label, int(filename[0:3])
            else:
                randomSubjectPath = self.indices[random.randint(0, len(self.indices)-1)]
                _, veryPath = self.findCodeLabelpath(randomSubjectPath)
                veryIm = Image.open(self.codeLabelDir + veryPath)
                veryImArray = np.array(veryIm, dtype=np.float32)
                veryImArray /= 256
                #veryImArray -= 1
                return label, mask, veryImArray, int(randomSubjectPath[0:3])
        else:
            return label, mask, None, None
        
    #根据图片名字,返回同一组里光照条件最好的图片路径
    def findBestIllumCodeImagepath(self, fullpath):
        span = re_pose.search(fullpath).span()
        camPos = list(fullpath[span[0]+1:span[1]-1])
        camPos.insert(2,'_')
        camPos = ''.join(camPos)
        #get 01_0 like string
        bestIllum = self.cameraPositions[camPos][1]

        labelpath = list(fullpath)
        labelpath[span[1]:span[1]+2] = bestIllum[:]
        labelpath = ''.join(labelpath)
        codepath = str(labelpath).replace('cropped', 'code')
        return (codepath, labelpath)

    def findCodeLabelpath(self, fullpath, labelnum):
        span = re_poseIllum.search(fullpath).span()
        #print span
        #camPosIllu =fullpath[span[0]+1:span[1]-1]
        #print camPosIllu
        #labelpath = fullpath.replace(camPosIllu, '051_06')
        tempath = list(fullpath)
        if self.LOAD_60_LABEL:
            camPos = fullpath[span[0]+1:span[0]+4]
            if(camPos == '240' or camPos == '010'): #+90/75
                tempath[span[0]+1:span[1]-1] = '200_08' #+60
            elif (camPos == '120' or camPos == '110'): #-90/75
                tempath[span[0]+1:span[1]-1] = '090_15' #-60
            else:
                tempath[span[0]+1:span[1]-1] = '051_06'
        else:
            tempath[span[0]+1:span[1]-1] = '051_06'
        labelpath = ''.join(tempath)
        codepath = str(labelpath).replace('cropped', 'code')
        if labelnum != -1:
            replace = None
            for i in self.cameraPositions.items():
                if i[1][0] == labelnum:
                    replace = ''.join([i[0][0:2],i[0][3],'_',i[1][1]])
                    tempath[span[0]+1:span[1]-1] = replace
                    labelpath = ''.join(tempath)
            if replace == None:
                print('damn labelnum bug!')
        return (codepath, labelpath)
    
   
    def findSameIllumCodeLabelpath(self, fullpath):
        span = re_poseIllum.search(fullpath).span()
        tempath = list(fullpath)
        if self.LOAD_60_LABEL:
            if self.findPose(fullpath) >= 0:
                tempath[span[0]+1:span[0]+4] = '190' #+45
            else:
                tempath[span[0]+1:span[0]+4] = '080' #-45
        else:
            tempath[span[0]+1:span[0]+4] = '051'
        labelpath = ''.join(tempath)
        codepath = str(labelpath).replace('cropped', 'code')
        return (codepath, labelpath)

    #根据图片名字来确定此图片的拍摄角度
    def findPose(self, fullpath):
        span = re_pose.search(fullpath).span()
        camPos = list(fullpath[span[0]+1:span[1]-1])
        camPos.insert(2,'_')
        camPos = ''.join(camPos)
        #get 01_0 like string
        return self.cameraPositions[camPos][0]

    
    #输入已裁剪的图片的Image.open格式和图片名,返回裁剪的眼嘴鼻和标注的眼嘴鼻的位置
    def GetFeatureParts(self, img_resize, filename, label=False):
        #crop four parts
        trans_points = np.empty([5,2],dtype=np.int32)
        if True:#not label:
            featpath = self.feat5ptDir + filename[0:-15] + '_trans.5pt'
            #print(filename)
            with open(featpath, 'r') as csvfile:
                reader = csv.reader(csvfile, delimiter=' ')
                for ind,row in enumerate(reader):
                    trans_points[ind,:] = row
        #print(trans_points)
        eyel_crop = np.zeros([EYE_H,EYE_W,3], dtype=np.float32);
        crop_y = int(trans_points[0,1] - EYE_H / 2);
        crop_y_end = crop_y + EYE_H;
        crop_x = int(trans_points[0,0] - EYE_W / 2);
        crop_x_end = crop_x + EYE_W;
        eyel_crop[...] = img_resize[crop_y:crop_y_end,crop_x:crop_x_end,:];

        eyer_crop = np.zeros([EYE_H,EYE_W,3], dtype=np.float32);
        crop_y = int(trans_points[1,1] - EYE_H / 2)
        crop_y_end = crop_y + EYE_H;
        crop_x = int(trans_points[1,0] - EYE_W / 2);
        crop_x_end = crop_x + EYE_W;
        eyer_crop[...] = img_resize[crop_y:crop_y_end,crop_x:crop_x_end,:];


        month_crop = np.zeros([MOUTH_H,MOUTH_W,3], dtype=np.float32);
        crop_y = int((trans_points[3,1] + trans_points[4,1]) // 2 - MOUTH_H / 2);
        crop_y_end = crop_y + MOUTH_H;
        crop_x = int((trans_points[3,0] + trans_points[4,0]) // 2 - MOUTH_W / 2);
        crop_x_end = crop_x + MOUTH_W;
        month_crop[...] = img_resize[crop_y:crop_y_end,crop_x:crop_x_end,:];


        nose_crop = np.zeros([NOSE_H,NOSE_W,3], dtype=np.float32);
        
        #pose specific crop for MultiPIE. But it doesn't affect much as long as you train and test with a consistant crop configuration.
        # A general crop for nose is provicded below as comment.
        pos_s = filename[-18:-15]
        if label or pos_s == '051':#frontal
            crop_y_end = trans_points[2,1] + 10 + 1;
            crop_x = trans_points[2,0] - 20 + 1;
        elif pos_s == '050':#+15 degrees
            crop_y_end = trans_points[2,1] + 10 + 1;
            crop_x = trans_points[2,0] - 25 + 1;
        elif pos_s == '140':#-15 degrees
            crop_y_end = trans_points[2,1] + 10 + 1;
            crop_x = trans_points[2,0] - (NOSE_W-25) + 1;
        elif pos_s ==  '041' or pos_s == '190':#+30/45 degrees
            crop_y_end = trans_points[2,1] + 10 + 1;
            crop_x = trans_points[2,0] - 30 + 1;
        elif pos_s == '130' or pos_s == '080':#-30/45 degrees
            crop_y_end = trans_points[2,1] + 10 + 1;
            crop_x = trans_points[2,0] - (NOSE_W-30) + 1;
        elif pos_s == '010' or pos_s =='200' or pos_s == '240':#+60/75/90 degrees
            crop_y_end = trans_points[2,1] + 10 + 1;
            crop_x = trans_points[2,0] - 32 + 1;
        elif pos_s == '120' or pos_s == '090' or pos_s == '110': #-60/75/90 degrees
            crop_y_end = trans_points[2,1] + 10 + 1;
            crop_x = trans_points[2,0] - (NOSE_W-32) + 1;
        else:
            print("BUG from nose selection!")

        #Or you can just use this general position for nose crop, that works for most poses.
        # crop_y_end = trans_points[2,1] + 10 + 1
        # crop_x = trans_points[2,0] - 20 + 1
        # crop_y_end = int(crop_y_end)
        # crop_x = int(crop_x)
        # crop_y = crop_y_end - NOSE_H;
        # crop_x_end = crop_x + NOSE_W;


        crop_y_end = int(crop_y_end)
        crop_x = int(crop_x)
        crop_y = crop_y_end - NOSE_H;
        crop_x_end = crop_x + NOSE_W;
        #import pdb; pdb.set_trace()
        nose_crop[...] = img_resize[crop_y:crop_y_end,crop_x:crop_x_end,:];

        if not label and self.MIRROR_TO_ONE_SIDE and self.findPose(filename) < 0:
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

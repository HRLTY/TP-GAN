from __future__ import division
from __future__ import print_function
import os,sys
import time
from glob import glob
import tensorflow as tf
import numpy as np
from six.moves import xrange

from ops import *
from utils import *

from net_input_everything_featparts import *
from time import localtime, strftime
import random
import pickle
import subprocess
relu = tf.nn.relu
import scipy.misc
import shutil
from utils import pp, visualize, to_json

import tensorflow as tf

#These parameters should provide a good initialization, but if for specific refinement, you can adjust them during training.

ALPHA_ADVER = 2e1
BELTA_FEATURE = 4e1 #3800 inital loss
UPDATE_G = 2  #optimize D once and UPDATE_G times G
UPDATE_D = 1
PART_W = 3
IDEN_W = 1e1
TV_WEIGHT = 1e-3
COND_WEIGHT = 0.3
L1_1_W = 1#0.5
L1_2_W = 1
L1_3_W = 1.5
RANGE= 60
RANGE_LOW = 0
SYM_W =  3e-1
CLIP_D = 0.1
L1 = True
MODE = 'fs60'  #'f' feature loss enabled.        'v' -verification enanbled. 'o' original, 'm' masked is mandatory and no need to specify
UPDATE_DV = 1 #optimize DV net
DF = True   #local discriminator 4x4
LOAD_60_LABEL = False #otherwise load frontal label
WITHOUT_CODEMAP = True
USE_MASK = False
ENABLE_SELECTION = True
RANDOM_VERIFY = False
RANK_MUL = 6
if WITHOUT_CODEMAP:
    CHANNEL = 3
else:
    CHANNEL = 6

flags = tf.app.flags
flags.DEFINE_integer("epoch", 250, "Epoch to train [25]")
flags.DEFINE_float("learning_rate", 1e-4, "Learning rate of for adam [0.0002]")
flags.DEFINE_float("beta1", 0.9, "Momentum term of adam [0.5]")
flags.DEFINE_integer("train_size", np.inf, "The size of train images [np.inf]")
flags.DEFINE_integer("batch_size", 20, "The size of batch images [64]")
flags.DEFINE_integer("image_size", 128, "The size of image to use (will be center cropped) [108]")
flags.DEFINE_integer("output_size", 128, "The size of the output images to produce [64]")
flags.DEFINE_integer("c_dim", 3, "Dimension of image color. [3]")
flags.DEFINE_string("dataset", "MultiPIE", "The name of dataset [celebA, mnist, lsun]")
flags.DEFINE_string("checkpoint_dir", "checkpoint60", "Directory name to save the checkpoints [checkpoint]")
flags.DEFINE_string("sample_dir", "samples", "Directory name to save the image samples [samples]")
flags.DEFINE_boolean("is_train", True, "True for training, False for testing [False]")
flags.DEFINE_boolean("is_crop", False, "True for training, False for testing [False]")
flags.DEFINE_boolean("visualize", False, "True for visualizing, False for nothing [False]")
FLAGS = flags.FLAGS

class DCGAN(object):
    def __init__(self, sess, image_size=128, is_crop=True,
                 batch_size=10, sample_size = 100, output_size=128,
                 y_dim=None, z_dim=100, gf_dim=64, df_dim=64,
                 gfc_dim=1024, dfc_dim=1024, c_dim=3, dataset_name='MultiPIE',
                 checkpoint_dir=None, sample_dir=None):
        """
        Args:
            sess: TensorFlow session
            batch_size: The size of batch. Should be specified before training.
            output_size: (optional) The resolution in pixels of the images. [64]
            y_dim: (optional) Dimension of dim for y. [None]
            z_dim: (optional) Dimension of dim for Z. [100]
            gf_dim: (optional) Dimension of gen filters in first conv layer. [64]
            df_dim: (optional) Dimension of discrim filters in first conv layer. [64]
            gfc_dim: (optional) Dimension of gen units for for fully connected layer. [1024]
            dfc_dim: (optional) Dimension of discrim units for fully connected layer. [1024]
            c_dim: (optional) Dimension of image color. For grayscale input, set to 1. [3]
        """
        self.test_batch_size = batch_size
        self.save_interval = 300
        self.sample_interval = 150
        self.sess = sess
        self.is_grayscale = (c_dim == 1)
        self.batch_size = 10
        self.sample_run_num = 15
        self.testing = False
        self.testingphase = 'FS'
        self.testimg = True
        if self.testing:
            #self.batch_size = 10
            self.testingphase = '60'#'gt50'
            self.sample_run_num = 99999999

        self.test_batch_size = self.batch_size
        self.image_size = image_size
        self.sample_size = sample_size
        self.output_size = output_size

        self.gf_dim = gf_dim
        self.df_dim = df_dim

        self.gfc_dim = gfc_dim
        self.dfc_dim = dfc_dim
        self.z_dim = 100
        self.c_dim = c_dim

        # batch normalization : deals with poor initialization helps gradient flow
        random.seed()
        self.DeepFacePath = '/home/shu.zhang/ruihuang/data/DeepFace.pickle'
        self.dataset_name = dataset_name
        self.checkpoint_dir = checkpoint_dir
        self.loadDeepFace(self.DeepFacePath)
        self.build_model()

    def build_model(self):

        #hold all four
        #Note: not true, if WITHOUT_CODEMAP is true, then here is pure images without codemap and 3 channels
        #mirror concatenate
        mc = lambda left : tf.concat_v2([left, left[:,:,::-1,:]], 3)
        self.images_with_code = tf.placeholder(tf.float32, [self.batch_size] + [self.output_size, self.output_size, CHANNEL], name='images_with_code')
        self.sample_images = tf.placeholder(tf.float32, [self.test_batch_size] + [self.output_size, self.output_size, CHANNEL], name='sample_images')

        if WITHOUT_CODEMAP:
            self.images = self.images_with_code
            self.sample_images_nocode = self.sample_images
        else:
            self.images = tf.split(3, 2, self.images_with_code)[0]
            self.sample_images_nocode = tf.split(3, 2, self.sample_images)[0]

        self.g_images = self.images #tf.reduce_mean(self.images, axis=3, keep_dims=True)
        self.g_samples = self.sample_images_nocode #tf.reduce_mean(self.sample_images_nocode, axis=3, keep_dims=True)

        self.g32_images_with_code = tf.image.resize_bilinear(self.images_with_code, [32, 32])
        self.g64_images_with_code = tf.image.resize_bilinear(self.images_with_code, [64, 64])

        self.g32_sampleimages_with_code = tf.image.resize_bilinear(self.sample_images, [32, 32])
        self.g64_sampleimages_with_code = tf.image.resize_bilinear(self.sample_images, [64, 64])

        self.labels = tf.placeholder(tf.float32, [self.batch_size] + [self.output_size, self.output_size, 3], name='label_images')
        self.poselabels = tf.placeholder(tf.int32, [self.batch_size])
        self.idenlabels = tf.placeholder(tf.int32, [self.batch_size])
        self.landmarklabels = tf.placeholder(tf.float32, [self.batch_size, 5*2])
        self.g_labels = self.labels #tf.reduce_mean(self.labels, 3, keep_dims=True)
        self.g8_labels = tf.image.resize_bilinear(self.g_labels, [8, 8])
        self.g16_labels = tf.image.resize_bilinear(self.g_labels, [16, 16])
        self.g32_labels = tf.image.resize_bilinear(self.g_labels, [32, 32])
        self.g64_labels = tf.image.resize_bilinear(self.g_labels, [64, 64])

        self.eyel = tf.placeholder(tf.float32, [self.batch_size, EYE_H, EYE_W, 3])
        self.eyer = tf.placeholder(tf.float32, [self.batch_size, EYE_H, EYE_W, 3])
        self.nose = tf.placeholder(tf.float32, [self.batch_size, NOSE_H, NOSE_W, 3])
        self.mouth = tf.placeholder(tf.float32, [self.batch_size, MOUTH_H, MOUTH_W, 3])

        self.eyel_label = tf.placeholder(tf.float32, [self.batch_size, EYE_H, EYE_W, 3])
        self.eyer_label = tf.placeholder(tf.float32, [self.batch_size, EYE_H, EYE_W, 3])
        self.nose_label = tf.placeholder(tf.float32, [self.batch_size, NOSE_H, NOSE_W, 3])
        self.mouth_label = tf.placeholder(tf.float32, [self.batch_size, MOUTH_H, MOUTH_W, 3])

        self.eyel_sam = tf.placeholder(tf.float32, [self.batch_size, EYE_H, EYE_W, 3])
        self.eyer_sam = tf.placeholder(tf.float32, [self.batch_size, EYE_H, EYE_W, 3])
        self.nose_sam = tf.placeholder(tf.float32, [self.batch_size, NOSE_H, NOSE_W, 3])
        self.mouth_sam = tf.placeholder(tf.float32, [self.batch_size, MOUTH_H, MOUTH_W, 3])


        #feats contains: self.feat128, self.feat64, self.feat32, self.feat16, self.feat8, self.feat
        self.G_eyel,self.c_eyel = self.partRotator(self.eyel, "PartRotator_eyel")
        self.G_eyer,self.c_eyer = self.partRotator(tf.concat_v2([self.eyer, self.eyel], axis=3), "PartRotator_eyer")
        self.G_nose,self.c_nose = self.partRotator(self.nose, "PartRotator_nose")
        self.G_mouth,self.c_mouth = self.partRotator(self.mouth, "PartRotator_mouth")

        self.G_eyel_sam, self.c_eyel_sam = self.partRotator(self.eyel_sam, "PartRotator_eyel", reuse=True)
        self.G_eyer_sam, self.c_eyer_sam = self.partRotator(tf.concat_v2([self.eyer_sam, self.eyel_sam],axis=3), "PartRotator_eyer", reuse=True)
        self.G_nose_sam, self.c_nose_sam = self.partRotator(self.nose_sam, "PartRotator_nose", reuse=True)
        self.G_mouth_sam, self.c_mouth_sam = self.partRotator(self.mouth_sam, "PartRotator_mouth", reuse=True)

        self.z = tf.random_normal([self.batch_size, self.z_dim], mean=0.0, stddev=0.02, seed=2017)

        #tf.placeholder(tf.float32, [self.batch_size, self.z_dim], name='z')

        self.feats = self.generator(mc(self.images_with_code), self.batch_size, name="encoder")
        self.feats += (mc(self.images_with_code), mc(self.g64_images_with_code), mc(self.g32_images_with_code),
                       self.G_eyel, self.G_eyer, self.G_nose, self.G_mouth,
                       self.c_eyel, self.c_eyer, self.c_nose, self.c_mouth,)
        self.check_sel128, self.check_sel64, self.check_sel32, self.check_sel16, self.check_sel8, self.G, self.G2, self.G3 = \
        self.decoder(*self.feats, batch_size=self.batch_size)
        self.poselogits, self.identitylogits, self.Glandmark = self.FeaturePredict(self.feats[5])

        sample_feats = self.generator(mc(self.sample_images),self.test_batch_size, name="encoder", reuse=True)
        self.sample512 = sample_feats[-1]
        sample_feats += (mc(self.sample_images), mc(self.g64_sampleimages_with_code), mc(self.g32_sampleimages_with_code),
                         self.G_eyel_sam, self.G_eyer_sam, self.G_nose_sam, self.G_mouth_sam,
                         self.c_eyel_sam, self.c_eyer_sam, self.c_nose_sam, self.c_mouth_sam,)
        self.sample_generator = self.decoder(*sample_feats, batch_size=self.test_batch_size, reuse=True)
        if not DF:
            self.D, self.D_logits = self.discriminator(self.g_labels)
            self.D_, self.D_logits_ = self.discriminator(self.G, reuse=True)
        else:
            print("Using local discriminator!")
            self.D, self.D_logits = self.discriminatorLocal(self.g_labels)
            self.D_, self.D_logits_ = self.discriminatorLocal(self.G, reuse=True)
        self.logfile = 'loss.txt'

        if 'f' in MODE:
            #self.verify_images_masked = tf.mul(self.verify_images, self.masks_binary)
            #can not apply mask !!!
            # self.Dv, self.Dv_logits = self.discriminatorVerify(self.labels, self.verify_images)
            _,_,_,_, self.G_pool5, self.Gvector = self.FeatureExtractDeepFace(tf.reduce_mean(self.G, axis=3, keep_dims=True))
            _,_,_,_, self.label_pool5, self.labelvector = self.FeatureExtractDeepFace(tf.reduce_mean(self.g_labels, axis=3, keep_dims=True), reuse=True)
            _,_,_,_, _, self.samplevector = self.FeatureExtractDeepFace(tf.reduce_mean(self.sample_images_nocode, axis=3, keep_dims=True), reuse=True)
            #self.Dv, self.Dv_logits = self.discriminatorClassify(self.Gvector)
            #self.dv_loss = tf.reduce_sum(tf.nn.sparse_softmax_cross_entropy_with_logits(self.Dv_logits, self.verify_labels))
            self.dv_loss = tf.reduce_mean(tf.abs(self.Gvector-self.labelvector))
            self.dv_loss += tf.reduce_mean(tf.reduce_sum(tf.reduce_sum(tf.abs(self.G_pool5-self.label_pool5),1),1))
            self.logfile = 'loss_verify.txt'
            #self.dv_sum = histogram_summary("dv_", self.Dv)

        # self.d__sum = histogram_summary("d_", self.D_)
        # self.d_sum = histogram_summary("d", self.D)
        # self.G_sum = image_summary("G", self.G)

        #basic loss

        # self.d_loss_real = tf.reduce_mean(self.D_logits)
        # self.d_loss_fake = -tf.reduce_mean(self.D_logits_)
        # self.g_loss_adver = -tf.reduce_mean(self.D_logits_)
        self.d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(self.D_logits, tf.ones_like(self.D) * 0.9))
        self.d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(self.D_logits_, tf.zeros_like(self.D_)))
        self.g_loss_adver = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(self.D_logits_, tf.ones_like(self.D_) * 0.9))

        #self.mark_regression_loss = tf.reduce_mean(tf.square(tf.abs(self.landmarklabels-self.Glandmark)))
        #self.poseloss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(self.poselogits, self.poselabels))
        self.idenloss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(self.identitylogits, self.idenlabels))

        self.eyel_loss = tf.reduce_mean(tf.reduce_sum(tf.reduce_sum(tf.abs(self.c_eyel - self.eyel_label), 1), 1))
        self.eyer_loss = tf.reduce_mean(tf.reduce_sum(tf.reduce_sum(tf.abs(self.c_eyer - self.eyer_label), 1), 1))
        self.nose_loss = tf.reduce_mean(tf.reduce_sum(tf.reduce_sum(tf.abs(self.c_nose - self.nose_label), 1), 1))
        self.mouth_loss = tf.reduce_mean(tf.reduce_sum(tf.reduce_sum(tf.abs(self.c_mouth - self.mouth_label), 1), 1))
        #rotation L1 / L2 loss in g_loss
        # one8 = tf.ones([1,8,4,1],tf.float32)
        # mask8 = tf.concat_v2([one8, one8], 2)
        # mask16 = tf.image.resize_nearest_neighbor(mask8, size=[16, 16])
        # mask32 = tf.image.resize_nearest_neighbor(mask8, size=[32, 32])
        # mask64 = tf.image.resize_nearest_neighbor(mask8, size=[64, 64])
        # mask128 = tf.image.resize_nearest_neighbor(mask8, size=[128, 128])
        #use L2 for 128, L1 for others. mask emphasize left side.
        errL1 = tf.abs(self.G - self.g_labels) #* mask128
        errL1_2 = tf.abs(self.G2 - self.g64_labels) #* mask64
        errL1_3 = tf.abs(self.G3 - self.g32_labels) #* mask32
        #errcheck8 = tf.abs(self.check_sel8 - self.g8_labels) #* mask8
        #errcheck16 = tf.abs(self.check_sel16 - self.g16_labels) #* mask16
        errcheck32 = tf.abs(self.check_sel32 - self.g32_labels) #* mask32
        errcheck64 = tf.abs(self.check_sel64 - self.g64_labels) #* mask64
        errcheck128 = tf.abs(self.check_sel128 - self.g_labels) #* mask128

        self.weightedErrL1 = tf.reduce_mean(tf.reduce_sum(tf.reduce_sum(errL1, 1), 1))
        self.symErrL1 = tf.reduce_mean(tf.reduce_sum(tf.reduce_sum(symL1(#self.processor(self.G)
            tf.nn.avg_pool(self.G, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
           ), 1), 1))
        self.weightedErrL2 = tf.reduce_mean(tf.reduce_sum(tf.reduce_sum(errL1_2, 1), 1))
        self.symErrL2 = tf.reduce_mean(tf.reduce_sum(tf.reduce_sum(symL1(self.processor(self.G2)), 1), 1))
        self.weightedErrL3 = tf.reduce_mean(tf.reduce_sum(tf.reduce_sum(errL1_3, 1), 1))
        self.symErrL3 = tf.reduce_mean(tf.reduce_sum(tf.reduce_sum(symL1(self.processor(self.G3, reuse=True)), 1), 1))

        cond_L12 = tf.abs(tf.image.resize_bilinear(self.G, [64,64]) - tf.stop_gradient(self.G2))
        #self.condErrL12 = tf.reduce_mean(tf.reduce_sum(tf.reduce_sum(cond_L12, 1), 1))
        #cond_L23 = tf.abs(tf.image.resize_bilinear(self.G2, [32,32]) - tf.stop_gradient(self.G3))
        #self.condErrL23 = tf.reduce_mean(tf.reduce_sum(tf.reduce_sum(cond_L23, 1), 1))
        self.tv_loss = tf.reduce_mean(total_variation(self.G))
        #self.weightedErr_check8 = tf.reduce_mean(tf.reduce_sum(tf.reduce_sum(errcheck8, 1), 1))
        #self.weightedErr_check16 = tf.reduce_mean(tf.reduce_sum(tf.reduce_sum(errcheck16, 1), 1))
        # self.weightedErr_check32 = tf.reduce_mean(tf.reduce_sum(tf.reduce_sum(errcheck32, 1), 1))
        # self.weightedErr_check64 = tf.reduce_mean(tf.reduce_sum(tf.reduce_sum(errcheck64, 1), 1))
        # self.weightedErr_check128 = tf.reduce_mean(tf.reduce_sum(tf.reduce_sum(errcheck128, 1), 1))
        # mean = tf.reduce_mean(tf.reduce_mean(self.G, 1,keep_dims=True), 2, keep_dims=True)
        # self.stddev = tf.reduce_mean(tf.squared_difference(self.G, mean))

        self.d_loss = self.d_loss_real + self.d_loss_fake
        self.g_loss = L1_1_W * (self.weightedErrL1 + SYM_W * self.symErrL1) + L1_2_W * (self.weightedErrL2 + SYM_W * self.symErrL2) \
                      + L1_3_W * (self.weightedErrL3 + SYM_W * self.symErrL3)
        self.g_loss += BELTA_FEATURE * self.dv_loss + ALPHA_ADVER * self.g_loss_adver + IDEN_W * self.idenloss + self.tv_loss * TV_WEIGHT

        self.rot_loss = PART_W * (self.eyel_loss + self.eyer_loss + self.nose_loss + self.mouth_loss)
        #self.sel_loss = self.weightedErr_check32 + self.weightedErr_check64 + self.weightedErr_check128
        #self.g_loss += self.sel_loss

        self.var_file = open('var_log.txt', mode='a')
        t_vars = [var for var in tf.trainable_variables() if 'FeatureExtractDeepFace' not in var.name \
                                                                                    and 'processor' not in var.name]
        def isTargetVar(name, tokens):
            for token in tokens:
                if token in name:
                    return True
            return False
        dec128toks = ['dec128', 'recon128', 'check_img128']
        self.d_vars = [var for var in t_vars if 'discriminatorLocal' in var.name]
        self.all_g_vars = [var for var in t_vars if 'discriminatorLocal' not in var.name]
        self.rot_vars = [var for var in t_vars if 'Rotator' in var.name]
        self.sel_vars = [var for var in t_vars if 'select' in var.name]
        self.dec_vars = [var for var in t_vars if 'decoder' in var.name and 'select' not in var.name]
        self.enc_vars = [var for var in t_vars if 'encoder' in var.name]
        self.pre_vars = [var for var in t_vars if 'FeaturePredict' in var.name]
        #
        self.se_vars = list(self.enc_vars); self.se_vars.extend(self.sel_vars)

        self.ed_vars = list(self.dec_vars); self.ed_vars.extend(self.enc_vars);
        self.ed_vars.extend(self.pre_vars); self.ed_vars.extend(self.rot_vars);
        self.ed_vars.extend(self.sel_vars)

        #self.rd_vars = list(self.dec_vars); self.rd_vars.extend([var for var in self.d_vars if isTargetVar(var.name, dec128toks)])

        #print("-----enc and dec ---->", map(lambda x:x.name, self.ed_vars), sep='\n', file=var_file)
        #print("-----enc and sel ---->", map(lambda x:x.name, self.se_vars), sep='\n',  file=var_file)
        #print("-----discrim ---->", map(lambda x:x.name, self.d_vars),sep='\n',  file=var_file)

        self.saver = tf.train.Saver(t_vars, max_to_keep=2)

    def train(self, config):
        """Train DCGAN"""
        #data = glob(os.path.join("./data", config.dataset, "*.jpg"))
        data = MultiPIE(LOAD_60_LABEL=LOAD_60_LABEL, GENERATE_MASK=USE_MASK, RANDOM_VERIFY=RANDOM_VERIFY, MIRROR_TO_ONE_SIDE = True, source = self.testingphase)
        #np.random.shuffle(data)
        config.sample_dir += '{:05d}'.format(random.randint(1,100000))

        d_optim = tf.train.AdamOptimizer(config.learning_rate, beta1=config.beta1) \
                          .minimize(self.d_loss, var_list=self.d_vars)
        #clip_D = [p.assign(tf.clip_by_value(p, -CLIP_D, CLIP_D)) for p in self.d_vars]

        g_dec_optim = tf.train.AdamOptimizer(config.learning_rate, beta1=config.beta1) \
                           .minimize(self.g_loss, var_list=self.ed_vars)
        #g_enc_optim = tf.train.AdamOptimizer(config.learning_rate * 0.001, beta1=config.beta1) \
        #                  .minimize(self.g_loss, var_list=self.enc_vars)
        # s_optim = tf.train.AdamOptimizer(config.learning_rate, beta1=config.beta1) \
        #                   .minimize(self.sel_loss, var_list=self.se_vars)
        #g_sel_dec_optim = tf.train.RMSPropOptimizer(config.learning_rate) \
        #                 .minimize(self.g_loss + self.sel_loss + self.rot_loss, var_list=self.all_g_vars)
        rot_optim = tf.train.AdamOptimizer(config.learning_rate, beta1=config.beta1) \
                          .minimize(self.rot_loss, var_list=self.rot_vars)

        init_op = tf.global_variables_initializer()
        self.sess.run(init_op)

        counter = random.randint(1,30)
        start_time = time.time()

        if self.load(self.checkpoint_dir):
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")

        sample_images, filenames ,sample_eyel, sample_eyer, sample_nose, sample_mouth, \
            sample_labels, sample_leyel, sample_leyer, sample_lnose, sample_lmouth, sample_iden = data.test_batch(self.test_batch_size * self.sample_run_num)
        if not self.testing:
            sample_imagesT, filenamesT ,sample_eyelT, sample_eyerT, sample_noseT, sample_mouthT, \
            sample_labelsT, sample_leyelT, sample_leyerT, sample_lnoseT, sample_lmouthT, sample_idenT = data.test_batch(self.test_batch_size * self.sample_run_num * RANK_MUL, Pose=RANGE)
            if WITHOUT_CODEMAP:
                sample_images = sample_images[..., 0:3]
            #append loss log to file
            self.f = open(self.logfile, mode='a')
            self.f.write('----'+strftime("%a, %d %b %Y %H:%M:%S +0000", localtime())+' BEGINS----MODE:'+MODE+'-----\n')
            print("start training!")
            for epoch in xrange(config.epoch):
                #data = glob(os.path.join("./data", config.dataset, "*.jpg"))
                batch_idxs = min(data.size, config.train_size) // self.batch_size

                for idx in xrange(0, batch_idxs):
                    #load data from MultiPIE
                    batch_images_with_code, batch_labels, batch_masks, verify_images, verify_labels, \
                    batch_pose, batch_iden, batch_landmarks,\
                    batch_eyel, batch_eyer, batch_nose, batch_mouth,\
                    batch_eyel_label, batch_eyer_label, batch_nose_label, batch_mouth_label \
                        = data.next_image_and_label_mask_batch(self.batch_size, imageRange=RANGE, imageRangeLow=RANGE_LOW)
                    # batch_images = batch_images_with_code[:,:,:,0:3] #discard codes
                    if WITHOUT_CODEMAP:
                        batch_images_with_code = batch_images_with_code[..., 0:3]

                    # needs self.G(needing images with code) and real images
                    for _ in range(UPDATE_D):
                        # Update D network
                        _ = self.sess.run([d_optim,],
                            feed_dict={ self.images_with_code: batch_images_with_code,
                                        self.labels : batch_labels,
                                        self.eyel : batch_eyel, self.eyer : batch_eyer, self.nose: batch_nose, self.mouth: batch_mouth,
                                        })
                    for _ in range(UPDATE_G):
                        # Update G network
                        # Run g_optim twice to make sure that d_loss does not go to zero (different from paper)
                        _ = self.sess.run([rot_optim, g_dec_optim,],
                        # _ = self.sess.run([g_sel_dec_optim],
                            feed_dict={self.images_with_code: batch_images_with_code,
                                       self.labels : batch_labels,
                                       self.eyel : batch_eyel, self.eyer : batch_eyer, self.nose: batch_nose, self.mouth: batch_mouth,
                                       self.poselabels : batch_pose, self.idenlabels: batch_iden, self.landmarklabels: batch_landmarks,
                                       self.eyel_label : batch_eyel_label, self.eyer_label : batch_eyer_label,
                                       self.nose_label: batch_nose_label, self.mouth_label : batch_mouth_label
                                       })

                    counter += 1
                    print('.',end='');sys.stdout.flush()
                    if(counter % 5 == 0):
                        self.evaluate(epoch, idx, batch_idxs, start_time, 'train',
                                       batch_images_with_code,  batch_eyel, batch_eyer, batch_nose, batch_mouth,
                                       batch_labels, batch_eyel_label, batch_eyer_label, batch_nose_label, batch_mouth_label, batch_iden);
                    if np.mod(counter, self.sample_interval) == self.sample_interval-1:
                        for i in range(self.sample_run_num):
                            print(i, end=' ')
                            currentBatchSamples = sample_images[i*self.test_batch_size:(i+1)*self.test_batch_size,...]
                            currentBatchEyel =  sample_eyel[i*self.test_batch_size:(i+1)*self.test_batch_size,...]
                            currentBatchEyer =  sample_eyer[i*self.test_batch_size:(i+1)*self.test_batch_size,...]
                            currentBatchNose = sample_nose[i*self.test_batch_size:(i+1)*self.test_batch_size,...]
                            currentBatchMouth = sample_mouth[i*self.test_batch_size:(i+1)*self.test_batch_size,...]
                            samples = self.sess.run(
                                self.sample_generator,
                                feed_dict={ self.sample_images: currentBatchSamples,
                                            self.eyel_sam : currentBatchEyel,
                                            self.eyer_sam : currentBatchEyer,
                                            self.nose_sam : currentBatchNose,
                                            self.mouth_sam : currentBatchMouth
                                            })
                            savedtest = save_images(currentBatchSamples if WITHOUT_CODEMAP else currentBatchSamples[...,0:3], [100, 100],
                                                    './{}/{:02d}_{:04d}/train{}_'.format(config.sample_dir, epoch, idx, i), suffix='')
                            savedoutput = save_images(samples[5], [100, 100],
                                                      './{}/{:02d}_{:04d}/train{}_'.format(config.sample_dir, epoch, idx, i),suffix='_128')
                            savedoutput = save_images(samples[6], [100, 100],
                                                      './{}/{:02d}_{:04d}/train{}_'.format(config.sample_dir, epoch, idx, i),suffix='_64')
                            savedoutput = save_images(samples[7], [100, 100],
                                                      './{}/{:02d}_{:04d}/train{}_'.format(config.sample_dir, epoch, idx, i),suffix='_32')
                        print("[{} completed{} and saved {}.]".format(config.sample_dir, savedtest*self.sample_run_num, savedoutput*self.sample_run_num))

                        #testing accuracy
                        savedir = 'tem_test'
                        if not os.path.exists(savedir):
                            os.mkdir(savedir)
                        else:
                            #subprocess.call(['rm' '-f' 'tem_test/*'])
                            shutil.rmtree(savedir, ignore_errors=True)
                            os.mkdir(savedir)
                            print("cleaned tem_test!")
                        listfid = open('probef.txt','w')
                        for i in range(self.sample_run_num * RANK_MUL):
                            currentBatchSamples = sample_imagesT[i*self.test_batch_size:(i+1)*self.test_batch_size,...]
                            currentBatchEyel =  sample_eyelT[i*self.test_batch_size:(i+1)*self.test_batch_size,...]
                            currentBatchEyer =  sample_eyerT[i*self.test_batch_size:(i+1)*self.test_batch_size,...]
                            currentBatchNose = sample_noseT[i*self.test_batch_size:(i+1)*self.test_batch_size,...]
                            currentBatchMouth = sample_mouthT[i*self.test_batch_size:(i+1)*self.test_batch_size,...]
                            currentBatchLSamples = sample_labelsT[i*self.test_batch_size:(i+1)*self.test_batch_size,...]
                            currentBatchLEyel =  sample_leyelT[i*self.test_batch_size:(i+1)*self.test_batch_size,...]
                            currentBatchLEyer =  sample_leyerT[i*self.test_batch_size:(i+1)*self.test_batch_size,...]
                            currentBatchLNose = sample_lnoseT[i*self.test_batch_size:(i+1)*self.test_batch_size,...]
                            currentBatchLMouth = sample_lmouthT[i*self.test_batch_size:(i+1)*self.test_batch_size,...]
                            currentBatchIden = sample_idenT[i*self.test_batch_size:(i+1)*self.test_batch_size,...]
                            if i % (4 * RANK_MUL) == 0:
                                self.evaluate(epoch, idx, batch_idxs, start_time, 'test',
                                              currentBatchSamples, currentBatchEyel, currentBatchEyer, currentBatchNose, currentBatchMouth,
                                              currentBatchLSamples, currentBatchLEyel, currentBatchLEyer, currentBatchLNose, currentBatchLMouth, currentBatchIden);
                            samples = self.sess.run(
                                self.sample_generator,
                                feed_dict={ self.sample_images: currentBatchSamples,
                                            self.eyel_sam : currentBatchEyel,
                                            self.eyer_sam : currentBatchEyer,
                                            self.nose_sam : currentBatchNose,
                                            self.mouth_sam : currentBatchMouth
                                            })
                            samplevectors = self.samplevector.eval({self.sample_images_nocode : samples[5]})
                            for k in range(samplevectors.shape[0]):
                                filename = filenamesT[i*self.test_batch_size + k]
                                savefilename = filename.replace('.png','.feat')
                                label = filename[0:3]
                                listfid.write(savedir + '/' + filename +' '+ label + '\n')
                                result = samplevectors[k,:]
                                f = open(savedir +'/'+ savefilename,'wb')
                                f.write(result)
                                f.close()
                        listfid.close()
                        print("[{}  saved {} feats. calling comparision..]".format(savedir, self.test_batch_size * self.sample_run_num))
                        output, err = subprocess.Popen('./evaluation_rank.sh', stdout=subprocess.PIPE, shell=True).communicate()
                        tobePrint = '-------!' + ''.join([rank for rank in output.splitlines() if rank.startswith('Rank-1')]) + '!-------'
                        print(err, tobePrint)
                        self.var_file.write(tobePrint)
                        self.var_file.flush()
                    if np.mod(counter, self.save_interval) == self.save_interval-1:
                        self.save(config.checkpoint_dir, counter)
        else:

            print('test samples reading complete')
            batchnum = sample_images.shape[0] // self.test_batch_size #current test batch size
            savedtest = 0
            savedoutput = 0
            sample_dir = 'testall'
            for i in range(batchnum):
                print('generating test result batch{}'.format(i))
                ind = (i*self.test_batch_size, (i+1)*self.test_batch_size)
                if self.testimg:#Save images
                    samples = self.sess.run(
                        self.sample_generator,
                        feed_dict={ self.sample_images: sample_images[ind[0]:ind[1],:,:,:],
                                    self.eyel_sam : sample_eyel[ind[0]:ind[1],...],
                                    self.eyer_sam : sample_eyer[ind[0]:ind[1],...],
                                    self.nose_sam : sample_nose[ind[0]:ind[1],...],
                                    self.mouth_sam : sample_mouth[ind[0]:ind[1],...]}
                    )
                    colorgt = sample_images[ind[0]:ind[1],:,:,0:3]
                    savedtest += save_images(colorgt, [128, 128],
                                             './{}/'.format(sample_dir),isOutput=False, filelist=filenames[ind[0]:ind[1]])
                    savedoutput += save_images(samples[5], [128, 128],
                                               './{}/'.format(sample_dir),isOutput=True, filelist=filenames[ind[0]:ind[1]])
                    print("[{} completed{} and saved {}.]".format(sample_dir, savedtest, savedoutput))
                else:#save features
                    savedir = 'testall_f'#'gt50_f'
                    samples = self.sess.run(
                        self.sample512,
                        feed_dict={ self.sample_images: sample_images[ind[0]:ind[1],:,:,:],})
                    listfid = open('probef.txt','a')
                    for j in range(self.test_batch_size):
                        filename = filenames[ind[0]+j]
                        savefilename = filename.replace('.png','.feat')
                        label = filename[0:3]
                        listfid.write(savedir + '/' + filename +' '+ label + '\n')
                        result = samples[j,0:448]
                        if not os.path.exists(savedir):
                            os.mkdir(savedir)
                        f = open(savedir +'/'+ savefilename,'wb')
                        f.write(result)
                        f.close()
                    print("saved %d files!" % (self.test_batch_size * (i+1)))
                    listfid.close()


    def processor(self, images, reuse=False):
        #accept 3 channel images, output orginal 3 channels and 3 x 4 gradient map-> 15 channels
        with tf.variable_scope("processor") as scope:
            if reuse:
                scope.reuse_variables()
            input_dim = images.get_shape()[-1]
            gradientKernel = gradientweight()
            output_dim = gradientKernel.shape[-1]
            print("processor:", output_dim)
            k_hw = gradientKernel.shape[0]
            init = tf.constant_initializer(value=gradientKernel, dtype=tf.float32)
            w = tf.get_variable('w', [k_hw, k_hw, input_dim, output_dim],
                                initializer=init)
            conv = tf.nn.conv2d(images, w, strides=[1, 1, 1, 1], padding='SAME')
            #conv = conv * 2
            return tf.concat_v2([images, conv], 3)

    def FeaturePredict(self, featvec, reuse=False):
        with tf.variable_scope("FeaturePredict") as scope:
            if reuse:
                scope.reuse_variables()
            #identity的数目,训练中identity编号不能超过这个
            identitylogits = linear(Dropout(featvec, keep_prob=0.3, is_training= not self.testing), output_size=340, scope='idenLinear', bias_start=0.1, with_w=True)[0]
            
            return None, identitylogits, None

    def discriminatorLocal(self, images, reuse=False):
        with tf.variable_scope("discriminatorLocal") as scope:
            if reuse:
                scope.reuse_variables()

            h0 = lrelu(conv2d(images, self.df_dim, name='d_h0_conv'))
            #64
            h1 = lrelu(batch_norm(conv2d(h0, self.df_dim*2, name='d_h1_conv'), name='d_bn1'))
            #32
            h2 = lrelu(batch_norm(conv2d(h1, self.df_dim*4, name='d_h2_conv'), name='d_bn2'))
            #16
            h3 = lrelu(batch_norm(conv2d(h2, self.df_dim*8, name='d_h3_conv'), name='d_bn3'))
            # #8x8
            h3r1 = resblock(h3, name = "d_h3_conv_res1")
            h4 = lrelu(batch_norm(conv2d(h3r1, self.df_dim*8, name='d_h4_conv'), name='d_bn4'))
            h4r1 = resblock(h4, name = "d_h4_conv_res1")
            h5 = conv2d(h4r1, 1, k_h=1, k_w=1, d_h=1, d_w=1, name='d_h5_conv')
            h6 = tf.reshape(h5, [self.batch_size, -1])
            #fusing 512 feature map to one layer prediction.
            return h6, h6 #tf.nn.sigmoid(h6), h6


    def decoder(self, feat128, feat64, feat32, feat16, feat8, featvec,
                g128_images_with_code, g64_images_with_code, g32_images_with_code,
                eyel, eyer, nose, mouth, c_eyel, c_eyer, c_nose, c_mouth, batch_size = 10, name="decoder", reuse = False):
        sel_feat_capacity = self.gf_dim
        with tf.variable_scope(name) as scope:
            if reuse:
                scope.reuse_variables()
            initial_all = tf.concat_v2([featvec, self.z], 1)
            initial_8 = relu(tf.reshape(linear(initial_all, output_size=8*8*self.gf_dim,scope='initial8', bias_start=0.1, with_w=True)[0],
                                        [batch_size, 8, 8, self.gf_dim]))
            initial_32 = relu(deconv2d(initial_8, [batch_size, 32, 32, self.gf_dim // 2], d_h=4, d_w=4, name="initial32"))
            initial_64 = relu(deconv2d(initial_32, [batch_size, 64, 64, self.gf_dim // 4], name="initial64"))
            initial_128 = relu(deconv2d(initial_64, [batch_size, 128, 128, self.gf_dim // 8], name="initial128"))

            before_select8 = resblock(tf.concat_v2([initial_8, feat8], 3), k_h=2, k_w=2, name = "select8_res_1")
            #selection T module
            reconstruct8 = resblock(resblock(before_select8, k_h=2, k_w=2, name="dec8_res1"), k_h=2, k_w=2, name="dec8_res2")

            #selection F module
            reconstruct16_deconv = relu(batch_norm(deconv2d(reconstruct8, [batch_size, 16, 16, self.gf_dim*8], name="g_deconv16"), name="g_bnd1"))
            before_select16 = resblock(feat16, name = "select16_res_1")
            reconstruct16 = resblock(resblock(tf.concat_v2([reconstruct16_deconv, before_select16], 3), name="dec16_res1"), name="dec16_res2")

            reconstruct32_deconv = relu(batch_norm(deconv2d(reconstruct16, [batch_size, 32, 32, self.gf_dim*4], name="g_deconv32"), name="g_bnd2"))
            before_select32 = resblock(tf.concat_v2([feat32, g32_images_with_code, initial_32], 3), name = "select32_res_1")
            reconstruct32 = resblock(resblock(tf.concat_v2([reconstruct32_deconv, before_select32], 3), name="dec32_res1"), name="dec32_res2")
            img32 = tf.nn.tanh(conv2d(reconstruct32, 3, d_h=1, d_w=1, name="check_img32"))

            reconstruct64_deconv = relu(batch_norm(deconv2d(reconstruct32, [batch_size, 64, 64, self.gf_dim*2], name="g_deconv64"), name="g_bnd3"))
            before_select64 = resblock(tf.concat_v2([feat64, g64_images_with_code, initial_64], 3), k_h=5, k_w=5, name = "select64_res_1")
            reconstruct64 = resblock(resblock(tf.concat_v2([reconstruct64_deconv, before_select64,
                                                            tf.image.resize_bilinear(img32, [64,64])], 3), name="dec64_res1"), name="dec64_res2")
            img64 = tf.nn.tanh(conv2d(reconstruct64, 3, d_h=1, d_w=1, name="check_img64"))

            reconstruct128_deconv = relu(batch_norm(deconv2d(reconstruct64, [batch_size, 128, 128, self.gf_dim], name="g_deconv128"), name="g_bnd4"))
            before_select128 = resblock(tf.concat_v2([feat128, initial_128, g128_images_with_code],3), k_h = 7, k_w = 7, name = "select128_res_1")
            reconstruct128 = resblock(tf.concat_v2([reconstruct128_deconv, before_select128,
                                                    self.partCombiner(eyel, eyer, nose, mouth),
                                                    self.partCombiner(c_eyel, c_eyer, c_nose, c_mouth),
                                                    tf.image.resize_bilinear(img64, [128,128])], 3), k_h=5, k_w=5, name="dec128_res1")
            reconstruct128_1 = lrelu(batch_norm(conv2d(reconstruct128, self.gf_dim, k_h=5, k_w=5, d_h=1, d_w=1, name="recon128_conv"), name="recon128_bnc"))
            reconstruct128_1_r = resblock(reconstruct128_1, name="dec128_res2")
            reconstruct128_2 = lrelu(batch_norm(conv2d(reconstruct128_1_r, self.gf_dim/2, d_h=1, d_w=1, name="recon128_conv2"),name="recon128_bnc2"))
            img128 = tf.nn.tanh(conv2d(reconstruct128_2, 3, d_h=1, d_w=1, name="check_img128"))

            return img128, img64, img32, img32, img32, img128, img64, img32

    def generator(self, images, batch_size, name = "generator", reuse = False):
        with tf.variable_scope(name) as scope:
            if reuse:
                scope.reuse_variables()
        # imgs: input: IMAGE_SIZE x IMAGE_SIZE x CHANNEL
        # return labels: IMAGE_SIZE x IMAGE_SIZE x 3
        # U-Net structure, slightly different from the original on the location of relu/lrelu
            #128x128
            c0 = lrelu(conv2d(images, self.gf_dim, k_h=7, k_w=7, d_h=1, d_w=1, name="g_conv0"))
            c0r = resblock(c0, k_h=7, k_w=7, name="g_conv0_res")
            c1 = lrelu(batch_norm(conv2d(c0r, self.gf_dim, k_h=5, k_w=5, name="g_conv1"),name="g_bnc1"))
            #64x64
            c1r = resblock(c1, k_h=5, k_w=5, name="g_conv1_res")
            c2 = lrelu(batch_norm(conv2d(c1r, self.gf_dim*2, name='g_conv2'),name="g_bnc2"))
            #32x32
            c2r = resblock(c2, name="g_conv2_res")
            c3 = lrelu(batch_norm(conv2d(c2r, self.gf_dim*4, name='g_conv3'),name="g_bnc3"))
            #16x16
            c3r = resblock(c3, name="g_conv3_res")
            c4 = lrelu(batch_norm(conv2d(c3r, self.gf_dim*8, name='g_conv4'),name="g_bnc4"))
            #8x8
            c4r = resblock(c4, name="g_conv4_res")
            # c5 = lrelu(batch_norm(conv2d(c4r, self.gf_dim*8, name='g_conv5'),name="g_bnc5"))
            # #4x4
            # #2x2
            # c6r = resblock(c6,k_h=2, k_w=2, name="g_conv6_res")
            c4r2 = resblock(c4r, name="g_conv4_res2")
            c4r3 = resblock(c4r2, name="g_conv4_res3")
            c4r4 = resblock(c4r3, name="g_conv4_res4")
            c4r4_l = tf.reshape(c4r4,[batch_size, -1])
            c7_l = linear(c4r4_l, output_size=512,scope='feature', bias_start=0.1, with_w=True)[0]
            c7_l_m = tf.maximum(c7_l[:, 0:256], c7_l[:, 256:])

            return c0r, c1r, c2r, c3r, c4r4, c7_l_m

    def partRotator(self, images, name, batch_size=10, reuse=False):
        #HW 40x40, 32x40, 32x48
        with tf.variable_scope(name) as scope:
            if reuse:
                scope.reuse_variables()
            c0 = lrelu(conv2d(images, self.gf_dim, d_h=1, d_w=1, name="p_conv0"))
            c0r = resblock(c0, name="p_conv0_res")
            c1 = lrelu(batch_norm(conv2d(c0r, self.gf_dim*2, name="p_conv1"),name="p_bnc1"))
            #down1
            c1r = resblock(c1, name="p_conv1_res")
            c2 = lrelu(batch_norm(conv2d(c1r, self.gf_dim*4, name='p_conv2'),name="p_bnc2"))
            #down2
            c2r = resblock(c2, name="p_conv2_res")
            c3 = lrelu(batch_norm(conv2d(c2r, self.gf_dim*8, name='p_conv3'),name="p_bnc3"))
            #down3 5x5, 4x5, 4x6
            c3r = resblock(c3, name="p_conv3_res")
            c3r2 = resblock(c3r, name="p_conv3_res2")

            shape = c3r2.get_shape().as_list()
            d1 = lrelu(batch_norm(deconv2d(c3r2, [shape[0], shape[1] * 2, shape[2] * 2, self.gf_dim*4], name="p_deconv1"), name="p_bnd1"))
            #up1
            after_select_d1 = lrelu(batch_norm(conv2d(tf.concat_v2([d1, c2r], axis=3), self.gf_dim*4, d_h=1, d_w=1, name="p_deconv1_s"),name="p_bnd1_s"))
            d1_r = resblock(after_select_d1, name="p_deconv1_res")
            d2 = lrelu(batch_norm(deconv2d(d1_r, [shape[0], shape[1] * 4, shape[2] * 4, self.gf_dim*2], name="p_deconv2"), name="p_bnd2"))
            #up2
            after_select_d2 = lrelu(batch_norm(conv2d(tf.concat_v2([d2, c1r], axis=3), self.gf_dim*2, d_h=1, d_w=1, name="p_deconv2_s"),name="p_bnd2_s"))
            d2_r = resblock(after_select_d2, name="p_deconv2_res")
            d3 = lrelu(batch_norm(deconv2d(d2_r, [shape[0], shape[1] * 8, shape[2] * 8, self.gf_dim], name="p_deconv3"), name="p_bnd3"))
            #up3
            after_select_d3 = lrelu(batch_norm(conv2d(tf.concat_v2([d3, c0r], axis=3), self.gf_dim, d_h=1, d_w=1, name="p_deconv3_s"),name="p_bnd3_s"))
            d3_r = resblock(after_select_d3, name="p_deconv3_res")

            check_part = tf.nn.tanh(conv2d(d3_r, 3, d_h=1, d_w=1, name="p_check"))

        return d3_r, check_part

    def partCombiner(self, eyel, eyer, nose, mouth):
        '''
        x         y
        43.5823   41.0000
        86.4177   41.0000
        64.1165   64.7510
        47.5863   88.8635
        82.5904   89.1124
        this is the mean locaiton of 5 landmarks
        '''
        eyel_p = tf.pad(eyel, [[0,0], [int(41 - EYE_H / 2 - 1), int(IMAGE_SIZE - (41+EYE_H/2 - 1))], [int(44 - EYE_W / 2 - 1), int(IMAGE_SIZE - (44+EYE_W/2-1))], [0,0]])
        eyer_p = tf.pad(eyer, [[0,0], [int(41 - EYE_H / 2 - 1), int(IMAGE_SIZE - (41+EYE_H/2 - 1))], [int(86 - EYE_W / 2 - 1), int(IMAGE_SIZE - (86+EYE_W/2-1))], [0,0]])
        nose_p = tf.pad(nose, [[0,0], [int(65 - NOSE_H / 2 - 1), int(IMAGE_SIZE - (65+NOSE_H/2 - 1))], [int(64 - NOSE_W / 2 - 1), int(IMAGE_SIZE - (64+NOSE_W/2-1))], [0,0]])
        month_p = tf.pad(mouth, [[0,0], [int(89 - MOUTH_H / 2 - 1), int(IMAGE_SIZE - (89+MOUTH_H/2 - 1))], [int(65 - MOUTH_W / 2 - 1), int(IMAGE_SIZE - (65+MOUTH_W/2-1))], [0,0]])
        eyes = tf.maximum(eyel_p, eyer_p)
        eye_nose = tf.maximum(eyes, nose_p)
        return tf.maximum(eye_nose, month_p)
    def evaluate(self,epoch, idx, batch_idxs, start_time, mode,
                 batch_images_with_code,  batch_eyel, batch_eyer, batch_nose, batch_mouth,
                 batch_labels, batch_eyel_label, batch_eyer_label, batch_nose_label, batch_mouth_label, batch_iden):
        errD = self.d_loss.eval({self.images_with_code: batch_images_with_code, self.labels: batch_labels,
                                 self.eyel : batch_eyel, self.eyer : batch_eyer, self.nose: batch_nose, self.mouth: batch_mouth,})

        errG_L = self.weightedErrL1.eval({self.images_with_code: batch_images_with_code, self.labels : batch_labels,
                                          self.eyel : batch_eyel, self.eyer : batch_eyer, self.nose: batch_nose, self.mouth: batch_mouth,})
        errG_L2 = self.weightedErrL2.eval({self.images_with_code: batch_images_with_code, self.labels : batch_labels,
                                           self.eyel : batch_eyel, self.eyer : batch_eyer, self.nose: batch_nose, self.mouth: batch_mouth,})
        errG_L3 = self.weightedErrL3.eval({self.images_with_code: batch_images_with_code, self.labels : batch_labels,
                                           self.eyel : batch_eyel, self.eyer : batch_eyer, self.nose: batch_nose, self.mouth: batch_mouth,})
        errG_adver = self.g_loss_adver.eval({self.images_with_code: batch_images_with_code,
                                             self.eyel : batch_eyel, self.eyer : batch_eyer, self.nose: batch_nose, self.mouth: batch_mouth,})
        errtv = self.tv_loss.eval({self.images_with_code: batch_images_with_code, self.labels : batch_labels,
                                   self.eyel : batch_eyel, self.eyer : batch_eyer, self.nose: batch_nose, self.mouth: batch_mouth,})

        errG_sym = self.symErrL1.eval({self.images_with_code: batch_images_with_code,
                                       self.eyel : batch_eyel, self.eyer : batch_eyer, self.nose: batch_nose, self.mouth: batch_mouth,})
        errG2_sym = self.symErrL2.eval({self.images_with_code: batch_images_with_code,
                                        self.eyel : batch_eyel, self.eyer : batch_eyer, self.nose: batch_nose, self.mouth: batch_mouth,})
        errG3_sym = self.symErrL3.eval({self.images_with_code: batch_images_with_code,
                                        self.eyel : batch_eyel, self.eyer : batch_eyer, self.nose: batch_nose, self.mouth: batch_mouth,})

        errcheck32 = 0#self.weightedErr_check32.eval({self.images_with_code: batch_images_with_code, self.labels : batch_labels, })
        errcheck64 = 0#self.weightedErr_check64.eval({self.images_with_code: batch_images_with_code, self.labels : batch_labels, })
        errcheck128 = 0#self.weightedErr_check128.eval({self.images_with_code: batch_images_with_code, self.labels : batch_labels, })

        erreyel = self.eyel_loss.eval({self.eyel: batch_eyel, self.eyel_label: batch_eyel_label })
        erreyer = self.eyer_loss.eval({self.eyel: batch_eyel, self.eyer: batch_eyer, self.eyer_label: batch_eyer_label})
        errnose = self.nose_loss.eval({self.nose: batch_nose, self.nose_label: batch_nose_label})
        errmouth = self.mouth_loss.eval({self.mouth : batch_mouth, self.mouth_label: batch_mouth_label})
        erriden = self.idenloss.eval({self.images_with_code: batch_images_with_code, self.idenlabels : batch_iden,})


        if 'f' in MODE:
            errDv = self.dv_loss.eval({self.images_with_code: batch_images_with_code, self.labels : batch_labels,
                                       self.eyel : batch_eyel, self.eyer : batch_eyer, self.nose: batch_nose, self.mouth: batch_mouth,})
        else:
            errDv = 0
        err_total_G = L1_1_W * (errG_L + errG_sym * SYM_W) + L1_2_W * (errG_L2 + errG2_sym * SYM_W) + L1_3_W * (errG_L3 + errG3_sym * SYM_W) \
                      + ALPHA_ADVER * errG_adver + errDv * BELTA_FEATURE + IDEN_W*erriden #+ COND_WEIGHT*(errCondL12 + errCondL23) + errtv * TV_WEIGHT
        errfeat_total = errcheck32 + errcheck64+errcheck128 + PART_W * (erreyel + erreyer + errnose + errmouth)
        tobePrint = "%s Epo[%2d][%4d/%4d][t%4.2f] d_l:%.4f" % (MODE + 'T' if mode == 'test' else '', epoch, idx, batch_idxs, time.time() - start_time, errD)
        tobePrint += " fstol:%.0f cg32:%.0f cg64:%.0f cg128:%.0f el:%.0f er:%.0f no:%.0f mo:%.0f\n" % \
                     (errfeat_total, errcheck32, errcheck64,errcheck128, erreyel, erreyer, errnose, errmouth)
        tobePrint += "g_l:%.0f gL1:%.0f(sym:%.0f) gL2:%.0f(sym:%.0f) gL3:%.0f(sym:%.0f) gadv:%.4f dv_l:%.2f iden:%.4f, tv:%.0f " \
                     % (err_total_G, errG_L, errG_sym, errG_L2, errG2_sym, errG_L3, errG3_sym, errG_adver, errDv,  erriden, errtv)
        if 'f' in MODE:
            tobePrint += 'L:{} G:{}'.format(ALPHA_ADVER, BELTA_FEATURE)
        self.f.write(tobePrint+'\n')
        self.f.flush()
        print(tobePrint)
    #DEEPFACE MODEL BEGINS---
    def loadDeepFace(self, DeepFacePath):
        if DeepFacePath is None:
            path = sys.modules[self.__class__.__module__].__file__
            # print path
            path = os.path.abspath(os.path.join(path, os.pardir))
            # print path
            path = os.path.join(path, "DeepFace.pickle")
            DeepFacePath = path
            logging.info("Load npy file from '%s'.", DeepFacePath)
        if not os.path.isfile(DeepFacePath):
            logging.error(("File '%s' not found. "), DeepFacePath)
            sys.exit(1)
        with open(DeepFacePath,'r') as file:
            self.data_dict = pickle.load(file)
        print("Deep Face pickle data file loaded")

    def FeatureExtractDeepFace(self, images, name = "FeatureExtractDeepFace", reuse=False):
        #Preprocessing: from color to gray(reduce_mean)
        with tf.variable_scope(name) as scope:
            if reuse:
                scope.reuse_variables()

            conv1 = self._conv_layer(images, name='conv1')
            print(3, type(3))
            slice1_1, slice1_2 = tf.split(3, 2, conv1)
            eltwise1 = tf.maximum(slice1_1, slice1_2)
            pool1 = tf.nn.max_pool(eltwise1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                                   padding='SAME')
            conv2_1 = self._conv_layer(pool1, name='conv2_1')
            slice2_1_1, slice2_1_2 = tf.split(3, 2, conv2_1)
            eltwise2_1 = tf.maximum(slice2_1_1, slice2_1_2)

            conv2_2 = self._conv_layer(eltwise2_1, name='conv2_2')
            slice2_2_1, slice2_2_2 = tf.split(3, 2, conv2_2)
            eltwise2_2 = tf.maximum(slice2_2_1, slice2_2_2)

            res2_1 = pool1 + eltwise2_2

            conv2a = self._conv_layer(res2_1, name='conv2a')
            slice2a_1, slice2a_2 = tf.split(3, 2, conv2a)
            eltwise2a = tf.maximum(slice2a_1, slice2a_2)

            conv2 = self._conv_layer(eltwise2a, name='conv2')
            slice2_1, slice2_2 = tf.split(3, 2, conv2)
            eltwise2 = tf.maximum(slice2_1, slice2_2)

            pool2 = tf.nn.max_pool(eltwise2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                                   padding='SAME')

            conv3_1 = self._conv_layer(pool2, name='conv3_1')
            slice3_1_1, slice3_1_2 = tf.split(3, 2, conv3_1)
            eltwise3_1 = tf.maximum(slice3_1_1, slice3_1_2)

            conv3_2 = self._conv_layer(eltwise3_1, name='conv3_2')
            slice3_2_1, slice3_2_2 = tf.split(3, 2, conv3_2)
            eltwise3_2 = tf.maximum(slice3_2_1, slice3_2_2)

            res3_1 = pool2 + eltwise3_2

            conv3_3 = self._conv_layer(res3_1, name='conv3_3')
            slice3_3_1, slice3_3_2 = tf.split(3, 2, conv3_3)
            eltwise3_3 = tf.maximum(slice3_3_1, slice3_3_2)

            conv3_4 = self._conv_layer(eltwise3_3, name='conv3_4')
            slice3_4_1, slice3_4_2 = tf.split(3, 2, conv3_4)
            eltwise3_4 = tf.maximum(slice3_4_1, slice3_4_2)

            res3_2 = res3_1 + eltwise3_4

            conv3a = self._conv_layer(res3_2, name='conv3a')
            slice3a_1, slice3a_2 = tf.split(3, 2, conv3a)
            eltwise3a = tf.maximum(slice3a_1, slice3a_2)

            conv3 = self._conv_layer(eltwise3a, name='conv3')
            slice3_1, slice3_2 = tf.split(3, 2, conv3)
            eltwise3 = tf.maximum(slice3_1, slice3_2)

            pool3 = tf.nn.max_pool(eltwise3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                                   padding='SAME')

            conv4_1 = self._conv_layer(pool3, name='conv4_1')
            slice4_1_1, slice4_1_2 = tf.split(3, 2, conv4_1)
            eltwise4_1 = tf.maximum(slice4_1_1, slice4_1_2)

            conv4_2 = self._conv_layer(eltwise4_1, name='conv4_2')
            slice4_2_1, slice4_2_2 = tf.split(3, 2, conv4_2)
            eltwise4_2 = tf.maximum(slice4_2_1, slice4_2_2)

            res4_1 = pool3 + eltwise4_2

            conv4_3 = self._conv_layer(res4_1, name='conv4_3')
            slice4_3_1, slice4_3_2 = tf.split(3, 2, conv4_3)
            eltwise4_3 = tf.maximum(slice4_3_1, slice4_3_2)

            conv4_4 = self._conv_layer(eltwise4_3, name='conv4_4')
            slice4_4_1, slice4_4_2 = tf.split(3, 2, conv4_4)
            eltwise4_4 = tf.maximum(slice4_4_1, slice4_4_2)

            res4_2 = res4_1 + eltwise4_4

            conv4_5 = self._conv_layer(res4_2, name='conv4_5')
            slice4_5_1, slice4_5_2 = tf.split(3, 2, conv4_5)
            eltwise4_5 = tf.maximum(slice4_5_1, slice4_5_2)

            conv4_6 = self._conv_layer(eltwise4_5, name='conv4_6')
            slice4_6_1, slice4_6_2 = tf.split(3, 2, conv4_6)
            eltwise4_6 = tf.maximum(slice4_6_1, slice4_6_2)

            res4_3 = res4_2 + eltwise4_6

            conv4a = self._conv_layer(res4_3, name='conv4a')
            slice4a_1, slice4a_2 = tf.split(3, 2, conv4a)
            eltwise4a = tf.maximum(slice4a_1, slice4a_2)

            conv4 = self._conv_layer(eltwise4a, name='conv4')
            slice4_1, slice4_2 = tf.split(3, 2, conv4)
            eltwise4 = tf.maximum(slice4_1, slice4_2)

            conv5_1 = self._conv_layer(eltwise4, name='conv5_1')
            slice5_1_1, slice5_1_2 = tf.split(3, 2, conv5_1)
            eltwise5_1 = tf.maximum(slice5_1_1, slice5_1_2)

            conv5_2 = self._conv_layer(eltwise5_1, name='conv5_2')
            slice5_2_1, slice5_2_2 = tf.split(3, 2, conv5_2)
            eltwise5_2 = tf.maximum(slice5_2_1, slice5_2_2)

            res5_1 = eltwise4 + eltwise5_2

            conv5_3 = self._conv_layer(res5_1, name='conv5_3')
            slice5_3_1, slice5_3_2 = tf.split(3, 2, conv5_3)
            eltwise5_3 = tf.maximum(slice5_3_1, slice5_3_2)

            conv5_4 = self._conv_layer(eltwise5_3, name='conv5_4')
            slice5_4_1, slice5_4_2 = tf.split(3, 2, conv5_4)
            eltwise5_4 = tf.maximum(slice5_4_1, slice5_4_2)

            res5_2 = res5_1 + eltwise5_4

            conv5_5 = self._conv_layer(res5_2, name='conv5_5')
            slice5_5_1, slice5_5_2 = tf.split(3, 2, conv5_5)
            eltwise5_5 = tf.maximum(slice5_5_1, slice5_5_2)

            conv5_6 = self._conv_layer(eltwise5_5, name='conv5_6')
            slice5_6_1, slice5_6_2 = tf.split(3, 2, conv5_6)
            eltwise5_6 = tf.maximum(slice5_6_1, slice5_6_2)

            res5_3 = res5_2 + eltwise5_6

            conv5_7 = self._conv_layer(res5_3, name='conv5_7')
            slice5_7_1, slice5_7_2 = tf.split(3, 2, conv5_7)
            eltwise5_7 = tf.maximum(slice5_7_1, slice5_7_2)

            conv5_8 = self._conv_layer(eltwise5_7, name='conv5_8')
            slice5_8_1, slice5_8_2 = tf.split(3, 2, conv5_8)
            eltwise5_8 = tf.maximum(slice5_8_1, slice5_8_2)

            res5_4 = res5_3 + eltwise5_8

            conv5a = self._conv_layer(res5_4, name='conv5a')
            slice5a_1, slice5a_2 = tf.split(3, 2, conv5a)
            eltwise5a = tf.maximum(slice5a_1, slice5a_2)

            conv5 = self._conv_layer(eltwise5a, name='conv5')
            slice5_1, slice5_2 = tf.split(3, 2, conv5)
            eltwise5 = tf.maximum(slice5_1, slice5_2)
            pool4 = tf.nn.max_pool(eltwise5, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                                   padding='SAME')
            pool4_transposed = tf.transpose(pool4, perm=[0, 3, 1, 2])
            # pool4_reshaped = tf.reshape(pool4_transposed, shape=[tf.shape(pool4)[0],-1])
            fc1 = self._fc_layer(pool4_transposed, name="fc1")
            slice_fc1_1, slice_fc1_2 = tf.split(1, 2, fc1)
            eltwise_fc1 = tf.maximum(slice_fc1_1, slice_fc1_2)

            return eltwise1, eltwise2, eltwise3, eltwise5, pool4, eltwise_fc1
        #DEEPFACE NET ENDS---

        #DEEPFACE OPS BEGINS---
    def _conv_layer(self, input_, output_dim=96,
                    k_h=3, k_w=3, d_h=1, d_w=1, stddev=0.02,
                    name="conv2d"):
        #Note: currently kernel size and input output channel num are decided by loaded filter weights.
        #only strides are decided by calling param.
        with tf.variable_scope(name) as scope:
            filt = self.get_conv_filter(name)
            conv = tf.nn.conv2d(input_, filt, strides=[1, d_h, d_w, 1], padding='SAME')
            conv_biases = self.get_bias(name)
            return tf.nn.bias_add(conv, conv_biases)
            return conv

    def _fc_layer(self, bottom, name="fc1", num_classes=None):
        with tf.variable_scope(name) as scope:
            #shape = bottom.get_shape().as_list()
            if name == 'fc1':
                filt = self.get_fc_weight(name)
                bias = self.get_bias(name)
            reshaped_bottom = tf.reshape(bottom,[tf.shape(bottom)[0],-1])
            return tf.matmul(reshaped_bottom, filt) + bias

    def get_conv_filter(self, name):
        init = tf.constant_initializer(value=self.data_dict[name][0],
                                       dtype=tf.float32)
        shape = self.data_dict[name][0].shape
        var = tf.get_variable(name="filter", initializer=init, shape=shape)
        return var

    def get_bias(self, name, num_classes=None):
        bias_wights = self.data_dict[name][1]
        shape = self.data_dict[name][1].shape
        init = tf.constant_initializer(value=bias_wights,
                                       dtype=tf.float32)
        var = tf.get_variable(name="biases", initializer=init, shape=shape)
        return var

    def get_fc_weight(self, name):
        init = tf.constant_initializer(value=self.data_dict[name][0],
                                       dtype=tf.float32)
        shape = self.data_dict[name][0].shape
        var = tf.get_variable(name="weights", initializer=init, shape=shape)
        return var

    #DEEPFACE OPS ENDS---

    def save(self, checkpoint_dir, step):
        model_name = "DCGAN.model"
        model_dir = "%s_%s_%s" % (self.dataset_name, self.batch_size, self.output_size)
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        print(" [*] Saving checkpoints...at step " + str(step))
        self.saver.save(self.sess,
                        os.path.join(checkpoint_dir, model_name),
                        global_step=step)

    def load(self, checkpoint_dir):
        print(" [*] Reading checkpoints...")

        model_dir = "%s_%s_%s" % (self.dataset_name, self.batch_size, self.output_size)
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            print(" [*] Success to read {}".format(ckpt_name))
            return True
        else:
            print(" [*] Failed to find a checkpoint")
            return False

def main(_):
    pp.pprint(flags.FLAGS.__flags)

    if not os.path.exists(FLAGS.checkpoint_dir):
        os.makedirs(FLAGS.checkpoint_dir)
    if not os.path.exists(FLAGS.sample_dir):
        os.makedirs(FLAGS.sample_dir)

    with tf.Session() as sess:
        dcgan = DCGAN(sess,
                      image_size=FLAGS.image_size,
                      batch_size=FLAGS.batch_size,
                      output_size=FLAGS.output_size,
                      c_dim=FLAGS.c_dim,
                      dataset_name=FLAGS.dataset,
                      is_crop=FLAGS.is_crop,
                      checkpoint_dir=FLAGS.checkpoint_dir,
                      sample_dir=FLAGS.sample_dir)
        dcgan.train(FLAGS)

if __name__ == '__main__':
    tf.app.run()

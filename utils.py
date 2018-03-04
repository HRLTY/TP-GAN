"""
Some codes from https://github.com/Newmu/dcgan_code
"""
from __future__ import division
import math
import json
import random
import pprint
import scipy.misc
import numpy as np
from time import gmtime, strftime
#from net_input_everything_featparts import MultiPIE
import os

pp = pprint.PrettyPrinter()

get_stddev = lambda x, k_h, k_w: 1/math.sqrt(k_w*k_h*x.get_shape()[-1])

def get_image(image_path, image_size, is_crop=True, resize_w=64, is_grayscale = False):
    return transform(imread(image_path, is_grayscale), image_size, is_crop, resize_w)

def save_images(images, size, image_path, suffix=None, isOutput=False, filelist = None):
    #if isOutput:
    #   images = mirrorLeftToFull(images)
    return imsave(images, size, image_path, suffix, isOutput, filelist)

def mirrorLeftToFull(images):
    width = images.shape[2]
    leftImages = images[:,:,0:width/2,:]
    reversedLeftImages = leftImages[:,:,::-1,:]
    return np.concatenate((leftImages, reversedLeftImages), axis=2)

def imread(path, is_grayscale = False):
    if (is_grayscale):
        return scipy.misc.imread(path, flatten = True).astype(np.float)
    else:
        return scipy.misc.imread(path).astype(np.float)

def merge_images(images, size):
    return inverse_transform(images)

def merge(images, size):
    h, w = images.shape[1], images.shape[2]
    img = np.zeros((h * size[0], w * size[1], 3))
    for idx, image in enumerate(images):
        i = idx % size[1]
        j = idx // size[1]
        img[j*h:j*h+h, i*w:i*w+w, :] = image

    return img

def imsave(images, size, path, suffix=None, isOutput=False, filelist=None):
    num = images.shape[0]
    for i in range(num):
        if filelist is None:
            filename = path+str(i)
        else:
            filename = path+filelist[i][:-4] #discard .png
        if not isOutput:
            filename += '_test'
        if suffix is not None:
            filename += suffix
        filename += '.png'
        dirName = os.path.dirname(filename)
        if not os.path.exists(dirName):
            os.makedirs(dirName)
        if images.shape[-1] == 1:
            scipy.misc.imsave(filename,images[i,:,:,0])
        else:
            scipy.misc.imsave(filename,images[i,:,:,:])
    return num

def center_crop(x, crop_h, crop_w=None, resize_w=64):
    if crop_w is None:
        crop_w = crop_h
    h, w = x.shape[:2]
    j = int(round((h - crop_h)/2.))
    i = int(round((w - crop_w)/2.))
    return scipy.misc.imresize(x[j:j+crop_h, i:i+crop_w],
                               [resize_w, resize_w])

def transform(image, npx=64, is_crop=True, resize_w=64):
    # npx : # of pixels width/height of image
    if is_crop:
        cropped_image = center_crop(image, npx, resize_w=resize_w)
    else:
        cropped_image = image
    return np.array(cropped_image)/127.5 - 1.

def inverse_transform(images):
    return (images+1.)/2.


def to_json(output_path, *layers):
    with open(output_path, "w") as layer_f:
        lines = ""
        for w, b, bn in layers:
            layer_idx = w.name.split('/')[0].split('h')[1]

            B = b.eval()

            if "lin/" in w.name:
                W = w.eval()
                depth = W.shape[1]
            else:
                W = np.rollaxis(w.eval(), 2, 0)
                depth = W.shape[0]

            biases = {"sy": 1, "sx": 1, "depth": depth, "w": ['%.2f' % elem for elem in list(B)]}
            if bn != None:
                gamma = bn.gamma.eval()
                beta = bn.beta.eval()

                gamma = {"sy": 1, "sx": 1, "depth": depth, "w": ['%.2f' % elem for elem in list(gamma)]}
                beta = {"sy": 1, "sx": 1, "depth": depth, "w": ['%.2f' % elem for elem in list(beta)]}
            else:
                gamma = {"sy": 1, "sx": 1, "depth": 0, "w": []}
                beta = {"sy": 1, "sx": 1, "depth": 0, "w": []}

            if "lin/" in w.name:
                fs = []
                for w in W.T:
                    fs.append({"sy": 1, "sx": 1, "depth": W.shape[0], "w": ['%.2f' % elem for elem in list(w)]})

                lines += """
                    var layer_%s = {
                        "layer_type": "fc",
                        "sy": 1, "sx": 1,
                        "out_sx": 1, "out_sy": 1,
                        "stride": 1, "pad": 0,
                        "out_depth": %s, "in_depth": %s,
                        "biases": %s,
                        "gamma": %s,
                        "beta": %s,
                        "filters": %s
                    };""" % (layer_idx.split('_')[0], W.shape[1], W.shape[0], biases, gamma, beta, fs)
            else:
                fs = []
                for w_ in W:
                    fs.append({"sy": 5, "sx": 5, "depth": W.shape[3], "w": ['%.2f' % elem for elem in list(w_.flatten())]})

                lines += """
                    var layer_%s = {
                        "layer_type": "deconv",
                        "sy": 5, "sx": 5,
                        "out_sx": %s, "out_sy": %s,
                        "stride": 2, "pad": 1,
                        "out_depth": %s, "in_depth": %s,
                        "biases": %s,
                        "gamma": %s,
                        "beta": %s,
                        "filters": %s
                    };""" % (layer_idx, 2**(int(layer_idx)+2), 2**(int(layer_idx)+2),
                             W.shape[0], W.shape[3], biases, gamma, beta, fs)
        layer_f.write(" ".join(lines.replace("'","").split()))

def make_gif(images, fname, duration=2, true_image=False):
  import moviepy.editor as mpy

  def make_frame(t):
    try:
      x = images[int(len(images)/duration*t)]
    except:
      x = images[-1]

    if true_image:
      return x.astype(np.uint8)
    else:
      return ((x+1)/2*255).astype(np.uint8)

  clip = mpy.VideoClip(make_frame, duration=duration)
  clip.write_gif(fname, fps = len(images) / duration)

def visualize(sess, dcgan, config, option):
  if option == 0:
    data = MultiPIE(LOAD_60_LABEL=False)
    sample_images, filenames = data.test_batch(9999999999, Random = False)
    print('test samples reading complete')
    batchnum = sample_images.shape[0] // dcgan.test_batch_size #current test batch size
    savedtest = 0
    savedoutput = 0
    sample_dir = 'testall'
    for i in range(batchnum):
        print('generating test result batch{}'.format(i))
        ind = (i*dcgan.test_batch_size, (i+1)*dcgan.test_batch_size)
        samples = sess.run(
            dcgan.sample_generator,
            feed_dict={ dcgan.sample_images: sample_images[ind[0]:ind[1],...]}
        )
        savedtest += save_images(sample_images[ind[0]:ind[1],:,:,0:3], [128, 128],
                    './{}/'.format(sample_dir),isOutput=False, filelist=filenames[ind[0]:ind[1]])
        savedoutput += save_images(samples, [128, 128],
                    './{}/'.format(sample_dir),isOutput=True, filelist=filenames[ind[0]:ind[1]])
        print("[{} completed{} and saved {}.]".format(sample_dir, savedtest, savedoutput))
        #save_images(samples, [100, 100], './samples/test_%s.png' % strftime("%Y-%m-%d %H:%M:%S", gmtime()))
  elif option == 1:
    data = MultiPIE(LOAD_60_LABEL=False)
    sample_images, filenames = data.test_batch(9999999999, Random = False)
    print('test samples reading complete')
    batchnum = sample_images.shape[0] // dcgan.test_batch_size #current test batch size
    savedtest = 0
    savedoutput = 0
    sample_dir = 'testall'
    for i in range(batchnum):
        print('generating test result batch{}'.format(i))
        ind = (i*dcgan.test_batch_size, (i+1)*dcgan.test_batch_size)
        samples = sess.run(
            dcgan.sample_generator,
            feed_dict={ dcgan.sample_images: sample_images[ind[0]:ind[1],:,:,:]}
        )
        colorgt = sample_images[ind[0]:ind[1],:,:,0:3]
        #colorgt.mean(axis=3, keepdims=True)
        savedtest += save_images(colorgt, [128, 128],
                    './{}/'.format(sample_dir),isOutput=False, filelist=filenames[ind[0]:ind[1]])
        #print(samples[5].shape)
        savedoutput += save_images(samples[5], [128, 128],
                    './{}/'.format(sample_dir),isOutput=True, filelist=filenames[ind[0]:ind[1]])
        print("[{} completed{} and saved {}.]".format(sample_dir, savedtest, savedoutput))
        #save_images(samples, [100, 100], './samples/test_%s.png' % strftime("%Y-%m-%d %H:%M:%S", gmtime()))
  elif option == 2:
      #patch version
      data = MultiPIE(LOAD_60_LABEL=False)
      sample_images, filenames,sample_eyel, sample_eyer, sample_nose, sample_mouth = data.test_batch(9999999999, Random = False)
      print('test samples reading complete')
      batchnum = sample_images.shape[0] // dcgan.test_batch_size #current test batch size
      savedtest = 0
      savedoutput = 0
      sample_dir = 'testall'
      for i in range(batchnum):
          print('generating test result batch{}'.format(i))
          ind = (i*dcgan.test_batch_size, (i+1)*dcgan.test_batch_size)
          samples = sess.run(
              dcgan.sample_generator,
              feed_dict={ dcgan.sample_images: sample_images[ind[0]:ind[1],:,:,:],
                          dcgan.eyel_sam : sample_eyel[ind[0]:ind[1],...],
                          dcgan.eyer_sam : sample_eyer[ind[0]:ind[1],...],
                          dcgan.nose_sam : sample_nose[ind[0]:ind[1],...],
                          dcgan.mouth_sam : sample_mouth[ind[0]:ind[1],...]}
          )
          colorgt = sample_images[ind[0]:ind[1],:,:,0:3]
          #colorgt.mean(axis=3, keepdims=True)
          savedtest += save_images(colorgt, [128, 128],
                                   './{}/'.format(sample_dir),isOutput=False, filelist=filenames[ind[0]:ind[1]])
          #print(samples[5].shape)
          savedoutput += save_images(samples[5], [128, 128],
                                     './{}/'.format(sample_dir),isOutput=True, filelist=filenames[ind[0]:ind[1]])
          print("[{} completed{} and saved {}.]".format(sample_dir, savedtest, savedoutput))
          #save_images(samples, [100, 100], './samples/test_%s.png' % strftime("%Y-%m-%d %H:%M:%S", gmtime()))
  elif option == 3:
    values = np.arange(0, 1, 1./config.batch_size)
    for idx in xrange(100):
      print(" [*] %d" % idx)
      z_sample = np.zeros([config.batch_size, dcgan.z_dim])
      for kdx, z in enumerate(z_sample):
        z[idx] = values[kdx]

      samples = sess.run(dcgan.sampler, feed_dict={dcgan.z: z_sample})
      make_gif(samples, './samples/test_gif_%s.gif' % (idx))
  elif option == 4:
    image_set = []
    values = np.arange(0, 1, 1./config.batch_size)

    for idx in xrange(100):
      print(" [*] %d" % idx)
      z_sample = np.zeros([config.batch_size, dcgan.z_dim])
      for kdx, z in enumerate(z_sample): z[idx] = values[kdx]

      image_set.append(sess.run(dcgan.sampler, feed_dict={dcgan.z: z_sample}))
      make_gif(image_set[-1], './samples/test_gif_%s.gif' % (idx))

    new_image_set = [merge(np.array([images[idx] for images in image_set]), [10, 10]) \
        for idx in range(64) + range(63, -1, -1)]
    make_gif(new_image_set, './samples/test_gif_merged.gif', duration=8)

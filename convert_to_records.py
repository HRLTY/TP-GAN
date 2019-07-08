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

"""Converts MNIST data to TFRecords file format with Example protos."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from six.moves import xrange  # pylint: disable=redefined-builtin

import argparse
import os
import sys
from PIL import Image
import random
import re
import tensorflow as tf
import numpy as np
#from tensorflow.contrib.learn.python.learn.datasets import mnist
from dataset import MultiPIE

FLAGS = None
IMSIZE = 128

def _int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def convert_to(filenames, name, multipie):
  """Converts a dataset to tfrecords."""
  
  
  num_examples = len(filenames)
  print('reading set:', name, ' with %d files!' % num_examples)

  filename_record = os.path.join(FLAGS.directory, name + '_test' + '.tfrecords')

  print('Writing', filename_record)
  

  with tf.python_io.TFRecordWriter(filename_record) as writer:
    for index in range(num_examples):
      filename = filenames[index]
      #im = Image.open(os.path.join(FLAGS.data_img_dir, filename))
      #import ipdb;ipdb.set_trace()
      image, eyel, eyer, nose, mouth = multipie.load_image(filename)
      #if index == 79+1:
      
      label, leyel, leyer, lnose, lmouth = multipie.load_label_mask(filename)
      import ipdb;ipdb.set_trace()
      assert len(image) == IMSIZE * IMSIZE * 3
      #image_raw = np.array(im).astype('uint8').tostring()
      

      example = tf.train.Example(
          features=tf.train.Features(
              feature={
                  'image': _bytes_feature(image),
                  'eyel': _bytes_feature(eyel),
                  'eyer': _bytes_feature(eyer),
                  'nose': _bytes_feature(nose),
                  'mouth': _bytes_feature(mouth),
                  'label': _bytes_feature(label),
                  'leyel': _bytes_feature(leyel),
                  'leyer': _bytes_feature(leyer),
                  'lnose': _bytes_feature(lnose),
                  'lmouth': _bytes_feature(lmouth),
              }))
      writer.write(example.SerializeToString())
      if index % 1000 == 0:
        print('wrote %d files.' % index)
        if index == 5000:
          break


def main(unused_argv):
  # Get the data.
  with open(os.path.join(FLAGS.data_list_dir, 'train.csv'), 'r') as f:    
    train_filenames = f.read().splitlines()
  with open(os.path.join(FLAGS.data_list_dir, 'test.csv'), 'r') as f:    
    test_filenames = f.read().splitlines()
  multipie = MultiPIE(FLAGS.data_img_dir, FLAGS.data_keypoint_dir)
  # Convert to Examples and write the result to TFRecords.
  convert_to(train_filenames, 'train', multipie)
  #convert_to(test_filenames, 'test', multipie)

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--directory',
      type=str,
      default='./data',
      help='Directory to download data files and write the converted result'
  )
  parser.add_argument(
      '--data_list_dir',
      type=str,
      default='/Users/huangrui/Developer/singlePIE/s101/s101-no90',
      help='Directory for loading train/test.csv file'
  )
  parser.add_argument(
      '--data_img_dir',
      type=str,
      default='/Users/huangrui/Downloads/drawbox/MTCNNv2/S101_aligned',
      help='Directory for loading train/test.csv file'
  )
  parser.add_argument(
      '--data_keypoint_dir',
      type=str,
      default='/Users/huangrui/Downloads/drawbox/MTCNNv2/FS_t5pt',
      help='Directory for loading train/test.csv file'
  )
  parser.add_argument(
      '--validation_size',
      type=int,
      default=5000,
      help="""\
      Number of examples to separate from the training data for the validation
      set.\
      """
  )
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)

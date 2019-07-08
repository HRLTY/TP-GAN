# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
#You may obtain a copy of the License at
#
#    https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Train Imagenet."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from absl import flags
import config_factory;
import tensorflow as tf
import logging
logging.basicConfig(level = "INFO")
#import model
from dataset import get_multiPIE_batches


tfgan = tf.contrib.gan
gfile = tf.gfile

flags.DEFINE_string('checkpoint_dir', 'checkpoint',
                    'Directory name to save the checkpoints. [checkpoint]')
flags.DEFINE_string(
    # 'data_dir', '/gpu/hz138/Data/imagenet', #'/home/hz138/Data/imagenet',
    'data_dir', 'data/',
    'Directory with Imagenet input data as sharded recordio files of pre-'
    'processed images.')
flags.DEFINE_float('discriminator_learning_rate', 0.0004,
                   'Learning rate of for adam. [0.0004]')
flags.DEFINE_float('generator_learning_rate', 0.0001,
                   'Learning rate of for adam. [0.0004]')
flags.DEFINE_string('loss_type', 'hinge_loss', 'the loss type can be'
                    ' hinge_loss or kl_loss')
flags.DEFINE_integer('batch_size', 64, 'Number of images in input batch. [64]') # ori 16

FLAGS = flags.FLAGS

def main(_, is_test=False):
  #utils.log_versions();
  config_factory.ConfigFactory().flags = FLAGS;

  print('d_learning_rate', FLAGS.discriminator_learning_rate)
  print('g_learning_rate', FLAGS.generator_learning_rate)
  print('data_dir', FLAGS.data_dir)
  print(FLAGS.loss_type, FLAGS.batch_size)

  print('Starting the program..')
  gfile.MakeDirs(FLAGS.checkpoint_dir)

  model_dir = '%s_%s' % ('imagenet', FLAGS.batch_size)
  logdir = os.path.join(FLAGS.checkpoint_dir, model_dir)
  gfile.MakeDirs(logdir)
  image, eyel, eyer, nose, mouth, label, leyel, leyer, lnose, lmouth = \
  						get_multiPIE_batches(FLAGS.data_dir, FLAGS.batch_size)
  sess = tf.Session()
  image_value = sess.run(image)
  import ipdb; ipdb.set_trace()
  # graph = tf.Graph()
  # with graph.as_default():

  #   with tf.device(tf.train.replica_device_setter(FLAGS.ps_tasks)):
  #     # Instantiate global_step.
  #     global_step = tf.train.create_global_step()

  #   # Create model with FLAGS, global_step, and devices.
  #   assigned_gpus = os.environ.get("CUDA_VISIBLE_DEVICES", None);
  #   if( not None is assigned_gpus ):
  #       FLAGS.num_towers = len(assigned_gpus.split(","));
  #   devices = ['/gpu:{}'.format(tower) for tower in range(FLAGS.num_towers)]
  #   n_devices = len(devices);
  #   logging.info("Divide the batch size {} to {} devices.".format(FLAGS.batch_size, n_devices));
  #   FLAGS.batch_size = int(FLAGS.batch_size / n_devices);

  #   # Create noise tensors
  #   zs = utils.make_z_normal(
  #       FLAGS.num_towers, FLAGS.batch_size, FLAGS.z_dim)

  #   print('save_summaries_steps', FLAGS.save_summaries_steps)

  #   dcgan = model.SNGAN(
  #       zs=zs,
  #       config=FLAGS,
  #       global_step=global_step,
  #       devices=devices)

  #   with tf.device(tf.train.replica_device_setter(FLAGS.ps_tasks)):
  #     # Create sync_hooks when needed.
  #     if FLAGS.sync_replicas and FLAGS.num_workers > 1:
  #       print('condition 1')
  #       sync_hooks = [
  #           dcgan.d_opt.make_session_run_hook(FLAGS.task == 0),
  #           dcgan.g_opt.make_session_run_hook(FLAGS.task == 0)
  #       ]
  #     else:
  #       print('condition 2')
  #       sync_hooks = []

  #   train_ops = tfgan.GANTrainOps(
  #       generator_train_op=dcgan.g_optim,
  #       discriminator_train_op=dcgan.d_optim,
  #       global_step_inc_op=dcgan.increment_global_step)


  #   # We set allow_soft_placement to be True because Saver for the DCGAN model
  #   # gets misplaced on the GPU.
  #   session_config = tf.ConfigProto(
  #       allow_soft_placement=True, log_device_placement=False)
  #   session_config.gpu_options.allow_growth = True
  #   if is_test:
  #     return graph

  #   print("G step: ", FLAGS.g_step)
  #   print("D_step: ", FLAGS.d_step)
  #   train_steps = tfgan.GANTrainSteps(FLAGS.g_step, FLAGS.d_step)

  #   tfgan.gan_train(
  #       train_ops,
  #       get_hooks_fn=tfgan.get_sequential_train_hooks(
  #           train_steps=train_steps),
  #       hooks=([tf.train.StopAtStepHook(num_steps=2000000)] + sync_hooks),
  #       logdir=logdir,
  #       # master=FLAGS.master,
  #       # scaffold=scaffold, # load from google checkpoint
  #       is_chief=(FLAGS.task == 0),
  #       save_summaries_steps=FLAGS.save_summaries_steps,
  #       save_checkpoint_secs=FLAGS.save_checkpoint_secs,
  #       config=session_config)


if __name__ == '__main__':
  tf.app.run()
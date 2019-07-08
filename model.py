"""Convolutional Neural Network Model for TP-GAN, built with tf.layers."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

BATCH_NORM_DECAY = 0.9
BATCH_NORM_EPSILON = 1e-5
################################################################################
# Convenience functions for building the ResNet model.
################################################################################
def batch_norm_relu(inputs, is_training, relu=True, init_zero=False,
                    data_format='channels_last'):
  """Performs a batch normalization followed by a ReLU.
  Args:
    inputs: `Tensor` of shape `[batch, channels, ...]`.
    is_training: `bool` for whether the model is training.
    relu: `bool` if False, omits the ReLU operation.
    init_zero: `bool` if True, initializes scale parameter of batch
        normalization with 0 instead of 1 (default).
    data_format: `str` either "channels_first" for `[batch, channels, height,
        width]` or "channels_last for `[batch, height, width, channels]`.
  Returns:
    A normalized `Tensor` with the same `data_format`.
  """
  if init_zero:
    gamma_initializer = tf.zeros_initializer()
  else:
    gamma_initializer = tf.ones_initializer()

  if data_format == 'channels_first':
    axis = 1
  else:
    axis = 3

  inputs = tf.layers.batch_normalization(
      inputs=inputs,
      axis=axis,
      momentum=BATCH_NORM_DECAY,
      epsilon=BATCH_NORM_EPSILON,
      center=True,
      scale=True,
      training=is_training,
      fused=True,
      gamma_initializer=gamma_initializer)

  if relu:
    inputs = tf.nn.leaky_relu(inputs, alpha=0.2)
  return inputs

def fixed_padding(inputs, kernel_size, data_format='channels_last'):
  """Pads the input along the spatial dimensions independently of input size.
  Args:
    inputs: `Tensor` of size `[batch, channels, height, width]` or
        `[batch, height, width, channels]` depending on `data_format`.
    kernel_size: `int` kernel size to be used for `conv2d` or max_pool2d`
        operations. Should be a positive integer.
    data_format: `str` either "channels_first" for `[batch, channels, height,
        width]` or "channels_last for `[batch, height, width, channels]`.
  Returns:
    A padded `Tensor` of the same `data_format` with size either intact
    (if `kernel_size == 1`) or padded (if `kernel_size > 1`).
  """
  pad_total = kernel_size - 1
  pad_beg = pad_total // 2
  pad_end = pad_total - pad_beg
  if data_format == 'channels_first':
    padded_inputs = tf.pad(inputs, [[0, 0], [0, 0],
                                    [pad_beg, pad_end], [pad_beg, pad_end]])
  else:
    padded_inputs = tf.pad(inputs, [[0, 0], [pad_beg, pad_end],
                                    [pad_beg, pad_end], [0, 0]])

  return padded_inputs


def conv2d_fixed_padding(inputs, filters, kernel_size, strides,
                         data_format='channels_last'):
  """Strided 2-D convolution with explicit padding.
  The padding is consistent and is based only on `kernel_size`, not on the
  dimensions of `inputs` (as opposed to using `tf.layers.conv2d` alone).
  Args:
    inputs: `Tensor` of size `[batch, channels, height_in, width_in]`.
    filters: `int` number of filters in the convolution.
    kernel_size: `int` size of the kernel to be used in the convolution.
    strides: `int` strides of the convolution.
    data_format: `str` either "channels_first" for `[batch, channels, height,
        width]` or "channels_last for `[batch, height, width, channels]`.
  Returns:
    A `Tensor` of shape `[batch, filters, height_out, width_out]`.
  """
  if strides > 1:
    inputs = fixed_padding(inputs, kernel_size, data_format=data_format)

  return tf.layers.conv2d(
      inputs=inputs, filters=filters, kernel_size=kernel_size, strides=strides,
      padding=('SAME' if strides == 1 else 'VALID'), use_bias=False,
      kernel_initializer=tf.variance_scaling_initializer(),
      data_format=data_format)



def residual_block(inputs, filters, is_training, kernel_size=3, strides=1,
                   use_projection=False, data_format='channels_last',
                   dropblock_keep_prob=None, dropblock_size=None):
  """Standard building block for residual networks with BN after convolutions.
  Args:
    inputs: `Tensor` of size `[batch, channels, height, width]`.
    filters: `int` number of filters for the first two convolutions. Note that
        the third and final convolution will use 4 times as many filters.
    is_training: `bool` for whether the model is in training.
    strides: `int` block stride. If greater than 1, this block will ultimately
        downsample the input.
    use_projection: `bool` for whether this block should use a projection
        shortcut (versus the default identity shortcut). This is usually `True`
        for the first block of a block group, which may change the number of
        filters and the resolution.
    data_format: `str` either "channels_first" for `[batch, channels, height,
        width]` or "channels_last for `[batch, height, width, channels]`.
    dropblock_keep_prob: unused; needed to give method same signature as other
      blocks
    dropblock_size: unused; needed to give method same signature as other
      blocks
  Returns:
    The output `Tensor` of the block.
  """
  del dropblock_keep_prob
  del dropblock_size
  shortcut = inputs
  if use_projection:
    # Projection shortcut in first layer to match filters and strides
    shortcut = conv2d_fixed_padding(
        inputs=inputs, filters=filters, kernel_size=1, strides=strides,
        data_format=data_format)
    shortcut = batch_norm_relu(shortcut, is_training, relu=False,
                               data_format=data_format)

  inputs = conv2d_fixed_padding(
      inputs=inputs, filters=filters, kernel_size=kernel_size, strides=strides,
      data_format=data_format)
  inputs = batch_norm_relu(inputs, is_training, data_format=data_format)

  inputs = conv2d_fixed_padding(
      inputs=inputs, filters=filters, kernel_size=kernel_size, strides=1,
      data_format=data_format)
  inputs = batch_norm_relu(inputs, is_training, relu=False, init_zero=True,
                           data_format=data_format)

  return tf.nn.relu(inputs + shortcut)


def deconv2d_fixed_padding(inputs, filters, kernel_size, strides,
                         data_format='channels_last'):
  if strides > 1:
    inputs = fixed_padding(inputs, kernel_size, data_format=data_format)

  return tf.layers.conv2d_transpose(
      inputs=inputs, filters=filters, kernel_size=kernel_size, strides=strides,
      padding=('SAME' if strides == 1 else 'VALID'), use_bias=True,
      kernel_initializer=tf.variance_scaling_initializer(),
      data_format=data_format)


def encoder(inputs, is_training, data_format='channels_last'):
	gf_dim = 64
	inputs = conv2d_fixed_padding(inputs, filters=gf_dim, kernel_size=7, strides=1)
	inputs = tf.identity(inputs, 'g_conv0')
	inputs = residual_block(inputs, filters=gf_dim, is_training=is_training, kernel_size=7, strides=1)
	inputs = tf.identity(inputs, 'g_conv0_res') #128x128
	c0r = inputs

	inputs = conv2d_fixed_padding(inputs, filters=gf_dim, kernel_size=5, strides=2)
	inputs = batch_norm_relu(inputs, is_training, data_format=data_format)
	inputs = tf.identity(inputs, 'g_conv1') #64x64
	inputs = residual_block(inputs, filters=gf_dim, is_training=is_training, kernel_size=5, strides=1)
	inputs = tf.identity(inputs, 'g_conv1_res')
	c1r = inputs

	inputs = conv2d_fixed_padding(inputs, filters=gf_dim * 2, kernel_size=3, strides=2)
	inputs = batch_norm_relu(inputs, is_training, data_format=data_format)
	inputs = tf.identity(inputs, 'g_conv2') #32x32
	inputs = residual_block(inputs, filters=gf_dim * 2, is_training=is_training)
	inputs = tf.identity(inputs, 'g_conv2_res')
	c2r = inputs

	inputs = conv2d_fixed_padding(inputs, filters=gf_dim * 4, kernel_size=3, strides=2)
	inputs = batch_norm_relu(inputs, is_training, data_format=data_format)
	inputs = tf.identity(inputs, 'g_conv3') #16x16
	inputs = residual_block(inputs, filters=gf_dim * 4, is_training=is_training)
	inputs = tf.identity(inputs, 'g_conv3_res')
	c3r = inputs

	inputs = conv2d_fixed_padding(inputs, filters=gf_dim * 8, kernel_size=3, strides=2)
	inputs = batch_norm_relu(inputs, is_training, data_format=data_format)
	inputs = tf.identity(inputs, 'g_conv4') #8x8
	inputs = residual_block(inputs, filters=gf_dim * 8, is_training=is_training)
	inputs = tf.identity(inputs, 'g_conv4_res')
	inputs = residual_block(inputs, filters=gf_dim * 8, is_training=is_training)
	inputs = tf.identity(inputs, 'g_conv4_res2')
	inputs = residual_block(inputs, filters=gf_dim * 8, is_training=is_training)
	inputs = tf.identity(inputs, 'g_conv4_res3')
	inputs = residual_block(inputs, filters=gf_dim * 8, is_training=is_training)
	inputs = tf.identity(inputs, 'g_conv4_res4')
	c4r4 = inputs
	inputs = tf.reshape(inputs, [c4r4.get_shape().as_list()[0], -1])
	inputs = tf.layers.dense(inputs, 256)
	inputs = tf.identity(inputs, 'g_linear')

	return c0r, c1r, c2r, c3r, c4r4, inputs


def partRotator(inputs, is_training, data_format='channels_last'):

	gf_dim = 64
	inputs = conv2d_fixed_padding(inputs, filters=gf_dim, kernel_size=3, strides=1)	
	inputs = tf.nn.leaky_relu(inputs, alpha=0.2)
	inputs = tf.identity(inputs, 'p_conv0')

	inputs = residual_block(inputs, filters=gf_dim, is_training=is_training, kernel_size=7, strides=1)
	inputs = tf.identity(inputs, 'p_conv0_res')
	c0r = inputs

	inputs = conv2d_fixed_padding(inputs, filters=gf_dim * 2, kernel_size=3, strides=2)
	inputs = batch_norm_relu(inputs, is_training, data_format=data_format)
	inputs = tf.identity(inputs, 'p_conv1') #down 1
	inputs = residual_block(inputs, filters=gf_dim, is_training=is_training, kernel_size=3, strides=1)
	inputs = tf.identity(inputs, 'p_conv1_res')
	c1r = inputs

	inputs = conv2d_fixed_padding(inputs, filters=gf_dim * 4, kernel_size=3, strides=2)
	inputs = batch_norm_relu(inputs, is_training, data_format=data_format)
	inputs = tf.identity(inputs, 'p_conv2') #down2
	inputs = residual_block(inputs, filters=gf_dim * 4, is_training=is_training)
	inputs = tf.identity(inputs, 'p_conv2_res')
	c2r = inputs

	inputs = conv2d_fixed_padding(inputs, filters=gf_dim * 8, kernel_size=3, strides=2)
	inputs = batch_norm_relu(inputs, is_training, data_format=data_format)
	inputs = tf.identity(inputs, 'p_conv3') #down3
	inputs = residual_block(inputs, filters=gf_dim * 8, is_training=is_training)
	inputs = tf.identity(inputs, 'p_conv3_res')
	inputs = residual_block(inputs, filters=gf_dim * 8, is_training=is_training)
	inputs = tf.identity(inputs, 'p_conv3_res2')

	inputs = deconv2d_fixed_padding(inputs, filters=gf_dim * 4, kernel_size=3, strides=2)
	inputs = batch_norm_relu(inputs, is_training, data_format=data_format)
	inputs = tf.identity(inputs, 'p_dconv1') #up1
	d1 = inputs

	inputs = conv2d_fixed_padding(tf.concat([d1, c2r], axis=3), filters=gf_dim * 4, kernel_size=3, strides=1)
	inputs = batch_norm_relu(inputs, is_training, data_format=data_format)
	inputs = tf.identity(inputs, 'p_dconv1_s')
	inputs = residual_block(inputs, filters=gf_dim * 4, is_training=is_training)
	inputs = tf.identity(inputs, 'p_dconv1_res')

	inputs = deconv2d_fixed_padding(inputs, filters=gf_dim * 2, kernel_size=3, strides=2)
	inputs = batch_norm_relu(inputs, is_training, data_format=data_format)
	inputs = tf.identity(inputs, 'p_dconv2') #up2
	d2 = inputs

	inputs = conv2d_fixed_padding(tf.concat([d2, c1r], axis=3), filters=gf_dim * 2, kernel_size=3, strides=1)
	inputs = batch_norm_relu(inputs, is_training, data_format=data_format)
	inputs = tf.identity(inputs, 'p_dconv2_s')
	inputs = residual_block(inputs, filters=gf_dim * 2, is_training=is_training)
	inputs = tf.identity(inputs, 'p_dconv2_res')

	inputs = deconv2d_fixed_padding(inputs, filters=gf_dim, kernel_size=3, strides=2)
	inputs = batch_norm_relu(inputs, is_training, data_format=data_format)
	inputs = tf.identity(inputs, 'p_dconv3') #up3
	d3 = inputs

	inputs = conv2d_fixed_padding(tf.concat([d3, c0r], axis=3), filters=gf_dim, kernel_size=3, strides=1)
	inputs = batch_norm_relu(inputs, is_training, data_format=data_format)
	inputs = tf.identity(inputs, 'p_dconv3_s')
	inputs = residual_block(inputs, filters=gf_dim, is_training=is_training)
	inputs = tf.identity(inputs, 'p_dconv3_res')
	d3_r = inputs

	inputs = tf.nn.tanh(conv2d_fixed_padding(d3_r, filters=3, kernel_size=3, strides=1))
	inputs = tf.identity(inputs, 'p_check')

	return d3_r, inputs

def decoder(noise, 
			c0r, c1r, c2r, c3r, c4r, fea_linear,
			eyel, eyel_img,
			eyer, eyel_img,
			nose, nose_img,
			mouth, mouth_img):
	
	gf_dim = 64
	initial_all = tf.concat([fea_linear, noise], axis=1)
	batch_size = noise.get_shape().as_list()[0]
	initial_8 = tf.layers.dense(initial_all, 8 * 8 * gf_dim)
	initial_8 =tf.nn.leaky_relu(tf.reshape(initial_8, [batch_size, 8, 8, gf_dim]), alpha=0.2)

	initial_32 = tf.nn.leaky_relu(deconv2d_fixed_padding(initial_8, filters=gf_dim // 2, kernel_size=3, strides=4), alpha=0.2)
	initial_64 = tf.nn.leaky_relu(deconv2d_fixed_padding(initial_32, filters=gf_dim // 4, kernel_size=3, strides=2), alpha=0.2)
	initial_128 = tf.nn.leaky_relu(deconv2d_fixed_padding(initial_64, filters=gf_dim // 8, kernel_size=3, strides=2), alpha=0.2) 

	inputs = tf.concat([initial_8, c4r], axis=3)
	filters = inputs.get_shape().as_list()[3]
	inputs = residual_block(inputs, filters=filters, is_training=is_training)
	inputs = residual_block(inputs, filters=filters, is_training=is_training)
	inputs = residual_block(inputs, filters=filters, is_training=is_training)
	inputs = tf.identity(inputs, 'dec8_res2')

	








#discard keras version
# def resblock(input, kernel_size=3, stride=1):
# 	""" res block with 2 conv layers..."""
# 	l = tf.keras.layers
# 	out = tf.keras.Sequential()
# 	out_dim = input.get_shape()[-1]
# 	out.add(l.Conv2D(filters=out_dim, kernel_size=kernel_size,
# 					strides=stride, padding="same", use_bias=False))
# 	out.add(l.BatchNormalization, axis=3, momentum=0.9, epsilon=1e-5,
# 		center=True, scale=False, trainable=True)
# 	out.add(l.LeakyReLU(0.2))
# 	out.add(l.Conv2D(filters=out_dim, kernel_size=kernel_size,
# 					strides=stride, padding="same", use_bias=False))
# 	out.add(l.BatchNormalization, axis=3, momentum=0.9, epsilon=1e-5,
# 		center=True, scale=False, trainable=True)





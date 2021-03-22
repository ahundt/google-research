# coding=utf-8
# Copyright 2021 The Google Research Authors.
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

#!/usr/bin/env python
"""Attention module."""

import numpy as np
from ravens import utils
from ravens.models.resnet import ResNet36_4s
from ravens.models.resnet import ResNet43_8s
from ravens.models.transformer import ViT
import tensorflow as tf
import tensorflow_addons as tfa

from ravens.models.supernet_macro import build_supernet
from ravens.models import nas_utils
from ravens.models.efficientnet import CONV_KERNEL_INITIALIZER
from nas_train_final.models import build_model as nas_final_model #to build searched architecture using indicators

class Attention:
  """Attention module."""

  def __init__(self, in_shape, n_rotations, preprocess, lite=False, model_name='resnet'):
    self.n_rotations = n_rotations
    self.preprocess = preprocess

    max_dim = np.max(in_shape[:2])

    self.padding = np.zeros((3, 2), dtype=int)
    pad = (max_dim - np.array(in_shape[:2])) / 2
    self.padding[:2] = pad.reshape(2, 1)

    in_shape = np.array(in_shape)
    in_shape += np.sum(self.padding, axis=1)
    in_shape = tuple(in_shape)

    if model_name == 'resnet':
      # Initialize fully convolutional Residual Network with 43 layers and
      # 8-stride (3 2x2 max pools and 3 2x bilinear upsampling)
      if lite:
        d_in, d_out = ResNet36_4s(in_shape, 1)
      else:
        d_in, d_out = ResNet43_8s(in_shape, 1)
      self.model = tf.keras.models.Model(inputs=[d_in], outputs=[d_out])

    elif model_name=='supernet':
      # global_step = tf.train.get_global_step()
      # warmup_steps = 6255
      # dropout_rate = nas_utils.build_dropout_rate(global_step, warmup_steps)
      # todo fix dropout rate, for time being const. dropout being used
      dropout_rate = 0.2
      is_training = True # TODO check if its true or false
      in0 = tf.keras.layers.Input(shape=in_shape)
      out0, runtime_val, indicators = build_supernet(
        in0,
        model_name='single-path-search', # default option, add flags for other search space: ref @single-path-nas search_main.py
        training=is_training,
        # override_params=override_params, 
        dropout_rate=dropout_rate)
      # print("attention supernet shapes", in0.shape, out0.shape)
      y0 = tf.keras.layers.UpSampling2D(
          size=(2, 2), interpolation='bilinear', name='upsample_4')(
              out0)
      bn_axis = 3 if tf.keras.backend.image_data_format() == 'channels_last' else 1
      name = 'z0'
      z0 = tf.keras.layers.Conv2D(
          1,
          1,
          padding='same',
          use_bias=False,
          kernel_initializer=CONV_KERNEL_INITIALIZER,
          name=name + 'out_conv')(y0)
      print("z0.shape",z0.shape)
      self.model = tf.keras.models.Model(inputs=[in0], outputs=[z0])

    elif model_name == 'supernet_train_final':
    	print("Training final NAS architecture (attention) --------- ")
    	in0=tf.keras.layers.Input(shape=in_shape)
    	out0, _ = nas_final_model(
    		in0,
    		model_name='single-path',
    		training=True,
    		# override_params=override_params,
    		parse_search_dir='nas_model')
    	out0 = tf.keras.layers.Reshape((160,160,5), name='predictions')(out0)
    	out0 = tf.keras.layers.Conv2D(
    		1,
    		1,
    		padding='same',
    		use_bias=False,
    		kernel_initializer=CONV_KERNEL_INITIALIZER,
    		name='out_conv1')(out0)
    	out0 = tf.keras.layers.UpSampling2D(size=(2, 2), interpolation='bilinear', name='upsample_attn_1')(out0)
    	print("attention out0.shape",out0.shape)
    	self.model = tf.keras.models.Model(inputs=[in0], outputs=[out0])

    elif model_name == 'vit':
      self.model = ViT(image_size=in_shape, num_classes=1)
    else:
      raise NotImplementedError('model_name not implemented: ' + str(model_name))

    self.optim = tf.keras.optimizers.Adam(learning_rate=1e-4)
    self.metric = tf.keras.metrics.Mean(name='loss_attention')

  def forward(self, in_img, softmax=True):
    """Forward pass."""
    in_data = np.pad(in_img, self.padding, mode='constant')
    in_data = self.preprocess(in_data)
    in_shape = (1,) + in_data.shape
    in_data = in_data.reshape(in_shape)
    in_tens = tf.convert_to_tensor(in_data, dtype=tf.float32)

    # Rotate input.
    pivot = np.array(in_data.shape[1:3]) / 2
    rvecs = self.get_se2(self.n_rotations, pivot)
    in_tens = tf.repeat(in_tens, repeats=self.n_rotations, axis=0)
    in_tens = tfa.image.transform(in_tens, rvecs, interpolation='NEAREST')

    # Forward pass.
    in_tens = tf.split(in_tens, self.n_rotations)
    logits = ()
    for x in in_tens:
      logits += (self.model(x),)
    logits = tf.concat(logits, axis=0)

    # Rotate back output.
    rvecs = self.get_se2(self.n_rotations, pivot, reverse=True)
    logits = tfa.image.transform(logits, rvecs, interpolation='NEAREST')
    c0 = self.padding[:2, 0]
    c1 = c0 + in_img.shape[:2]
    logits = logits[:, c0[0]:c1[0], c0[1]:c1[1], :]

    logits = tf.transpose(logits, [3, 1, 2, 0])
    output = tf.reshape(logits, (1, np.prod(logits.shape)))
    if softmax:
      output = tf.nn.softmax(output)
      output = np.float32(output).reshape(logits.shape[1:])
    return output

  def train(self, in_img, p, theta, backprop=True):
    """Train."""
    self.metric.reset_states()
    with tf.GradientTape() as tape:
      output = self.forward(in_img, softmax=False)

      # Get label.
      theta_i = theta / (2 * np.pi / self.n_rotations)
      theta_i = np.int32(np.round(theta_i)) % self.n_rotations
      label_size = in_img.shape[:2] + (self.n_rotations,)
      label = np.zeros(label_size)
      label[p[0], p[1], theta_i] = 1
      label = label.reshape(1, np.prod(label.shape))
      label = tf.convert_to_tensor(label, dtype=tf.float32)

      # Get loss.
      loss = tf.nn.softmax_cross_entropy_with_logits(label, output)
      loss = tf.reduce_mean(loss)

    # Backpropagate
    if backprop:
      grad = tape.gradient(loss, self.model.trainable_variables)
      self.optim.apply_gradients(zip(grad, self.model.trainable_variables))
      self.metric(loss)

    return np.float32(loss)

  def load(self, path):
    self.model.load_weights(path)

  def save(self, filename):
    # self.model.save(filename)
    self.model.save_weights(filename)

  def get_se2(self, n_rotations, pivot, reverse=False):
    """Get SE2 rotations discretized into n_rotations angles counter-clockwise."""
    rvecs = []
    for i in range(n_rotations):
      theta = i * 2 * np.pi / n_rotations
      theta = -theta if reverse else theta
      rmat = utils.get_image_transform(theta, (0, 0), pivot)
      rvec = rmat.reshape(-1)[:-1]
      rvecs.append(rvec)
    return np.array(rvecs, dtype=np.float32)

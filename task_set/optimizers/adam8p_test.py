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

# python3
"""Tests for task_set.optimizers.adam8p."""
import numpy as np
from task_set.optimizers import adam8p
import tensorflow.compat.v1 as tf


class Adam8pTest(tf.test.TestCase):

  def test_adam8p_dense(self):
    opt = adam8p.Adam8POptimizer(learning_rate=1e-2)
    v = tf.get_variable(
        name='weights',
        shape=[10],
        dtype=tf.float32,
        initializer=tf.initializers.random_normal())
    loss = tf.reduce_mean(v**2)
    global_step = tf.train.get_or_create_global_step()
    train_op = opt.minimize(loss, var_list=[v], global_step=global_step)
    with self.cached_session() as sess:
      sess.run(tf.initializers.global_variables())
      initial_loss = sess.run(loss)
      for _ in range(100):
        sess.run(train_op)
      final_loss = sess.run(loss)
      self.assertLess(final_loss, 0.20804106)
      self.assertGreater(initial_loss, 0.50804106)

  def test_adam8p_sparse(self):
    opt = adam8p.Adam8POptimizer(learning_rate=1e-2)
    dense_var = tf.get_variable(
        name='weights',
        shape=[20, 20],
        dtype=tf.float32,
        initializer=tf.initializers.random_normal())
    loss = tf.reduce_mean(
        tf.nn.embedding_lookup(dense_var, np.arange(5, dtype=np.int32))**2)
    global_step = tf.train.get_or_create_global_step()
    train_op = opt.minimize(loss, var_list=[dense_var], global_step=global_step)
    with self.cached_session() as sess:
      sess.run(tf.initializers.global_variables())
      initial_loss = sess.run(loss)
      for _ in range(100):
        sess.run(train_op)
      final_loss = sess.run(loss)
      print(final_loss, initial_loss)
      self.assertLess(final_loss, 0.4)
      self.assertGreater(initial_loss, 1.0)


if __name__ == '__main__':
  tf.test.main()

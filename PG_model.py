from __future__ import division

import numpy as np
import random
import tensorflow as tf
import tensorflow.contrib.slim as slim
import imageio
import scipy.misc
import os
from PIL import ImageFile
import time
import traceback

# Actions x y in domain of
ACTION_RANGE = range(0,91)
ImageFile.LOAD_TRUNCATED_IMAGES = True


## Paste in for trainning#
class Policy_Gradient():

    def __init__(self, out_size):
        self.ep_obs, self.ep_as, self.ep_rs = [], [], []
        self.imageIn = tf.placeholder(shape=[None, 84, 84, 3], dtype=tf.float32, name="X")
        self.rewards = tf.placeholder(dtype=tf.float32, shape=[None, 1], name="rewards")
        self.actions = tf.placeholder(dtype=tf.int32, shape=[None, 90], name="actions")
        self.imageIn = tf.reshape(self.imageIn, shape=[-1, 84, 84, 3])
        self.conv1 = slim.conv2d( inputs=self.imageIn,
                                  num_outputs=32,
                                  kernel_size=[8, 8],
                                  stride=[4, 4],
                                  padding='VALID',
                                  biases_initializer=None)
        self.conv2 = slim.conv2d(inputs=self.conv1,
                                 num_outputs=64,
                                 kernel_size=[4, 4],
                                 stride=[2, 2],
                                 padding='VALID',
                                 biases_initializer=None)
        self.conv3 = slim.conv2d(inputs=self.conv2,
                                 num_outputs=64,
                                 kernel_size=[3, 3],
                                 stride=[1, 1],
                                 padding='VALID',
                                 biases_initializer=None)
        self.conv4 = slim.conv2d(inputs=self.conv3,
                                 num_outputs=out_size,
                                 kernel_size=[7, 7],
                                 stride=[1, 1],
                                 padding='VALID',
                                 biases_initializer=None)

        self.policy_head = slim.flatten(self.conv4)

        self.fc1 = tf.layers.dense(
            inputs=self.policy_head,
            units=200,
            activation=None,
            kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.3),
            bias_initializer=tf.constant_initializer(0.1),
            name='fc1')

        self.fc2 = tf.layers.dense(
            inputs=self.fc1,
            units=90,
            activation=None,
            kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.3),
            bias_initializer=tf.constant_initializer(0.1),
            name='fc1')

        self.prob = tf.nn.softmax(self.fc2, name='act_prob')

        self.neg_log_prob = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.prob)   # this is negative log of chosen action
        # or in this way:
        # neg_log_prob = tf.reduce_sum(-tf.log(self.all_act_prob)*tf.one_hot(self.tf_acts, self.n_actions), axis=1)
        self.loss = tf.reduce_mean(self.neg_log_prob * self.rewards)  # reward guided loss
        self.optimized = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(loss)

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())


    def choose_action(self, observation):
        prob_weights = self.sess.run(self.prob, feed_dict={self.imageIn: observation[np.newaxis, :]})
        action = np.random.choice(range(prob_weights.shape[1]), p=prob_weights.ravel())  # select action w.r.t the actions prob
        return action

    def store_transition(self, s, a, r):
        self.ep_obs.append(s)
        self.ep_as.append(a)
        self.ep_rs.append(r)

    def learn(self):
        # discount and normalize episode reward
        discounted_ep_rs_norm = self._discount_and_norm_rewards()

        # train on episode
        self.sess.run(self.optimized, feed_dict={
            self.imageIn: np.vstack(self.ep_obs),  # shape=[None, 84,84,3]
            self.actions: np.array(self.ep_as),  # shape=[None, 90]
            self.rewards: discounted_ep_rs_norm,  # shape=[None, 1]
        })

        self.ep_obs, self.ep_as, self.ep_rs = [], [], []    # empty episode data
        return discounted_ep_rs_norm

    def _discount_and_norm_rewards(self):
        # discount episode rewards
        discounted_ep_rs = np.zeros_like(self.ep_rs)
        running_add = 0
        for t in reversed(range(0, len(self.ep_rs))):
            running_add = running_add * 0.9 + self.ep_rs[t] #gamma = 0.9
            discounted_ep_rs[t] = running_add

        # normalize episode rewards
        discounted_ep_rs -= np.mean(discounted_ep_rs)
        discounted_ep_rs /= np.std(discounted_ep_rs)
        return discounted_ep_rs

    def enforce_KL_divergence(self):
        pass

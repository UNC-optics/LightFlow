#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 29 11:30:19 2022

@author: hoss
"""
import tensorflow as tf


def accuracy(y_true, y_pred):
    denom = tf.sqrt(tf.reduce_sum(tf.square(y_pred), axis=[1, 2, 3])*tf.reduce_sum(tf.square(y_true), axis=[1, 2, 3]))
    return 1-tf.reduce_mean((tf.reduce_sum(y_pred * y_true, axis=[1, 2, 3])+1)/(denom+1), axis = 0)
        
        
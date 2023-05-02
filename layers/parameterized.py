#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 19 14:02:43 2022

@author: hoss
"""

import tensorflow as tf
import numpy as np

#%
class SLM(tf.keras.layers.Layer):
    def __init__(
        self,
        device="LCoS",
        passthrough=True,
        num_frames=0,
        quantization=8,
        name="SLM",
        toggle_dmd=False,
    ):
        """
        A parameterized model for spatial light modulators.

        Parameters
        ----------
        device : str, optional
            DESCRIPTION. The default is 'LCoS'.
        passthrough : bool, optional
            DESCRIPTION. The default is True.
        num_frames : int, optional
            DESCRIPTION. The default is 0.
        quantization : int, optional
            DESCRIPTION. The default is 0.
        name : str, optional
            DESCRIPTION. The default is 'SLM'.
        toggle_dmd : TYPE, optional
            DESCRIPTION. The default is False.

        Returns
        -------
        None.

        """
        super(SLM, self).__init__(name=name)
        self.device = device
        self.quantization = 2**quantization
        self.toggle_dmd = toggle_dmd

        if num_frames:
            self.bs_nf = (1, num_frames)
        else:
            self.bs_nf = (1, 1)
        self.passthrough = passthrough

    @tf.custom_gradient
    def quantize(self, x):
        quantized = tf.math.round((x + np.pi) * (self.quantization - 1) / (2 * np.pi))
        quantized /= self.quantization - 1
        quantized *= 2 * np.pi
        quantized -= np.pi

        def grad(dy):
            return dy

        return quantized, grad

    @tf.custom_gradient
    def binarize(self, x):
        alpha = 1
        thr = tf.constant([[[[[0.5]]]]], dtype=tf.float32)
        bnrz = tf.cast(x > thr, tf.float32)

        def grad(dy):
            sig = 1 / (1 + tf.exp((-alpha * x) + thr * alpha))
            return dy * (alpha / 2.0) * (1 - sig**2 - (1 - sig) ** 2)

        return bnrz, grad

    def nomralize2pi(self, x):
        return tf.math.angle(tf.exp(tf.complex(0.0, x)))

    def build(self, input_shape):
        shape = (
            self.bs_nf + input_shape[0][2:-1] + (1,)
        )  # we have to ensure the last dim is 1

        if self.device == "LCoS":
            self.phase = self.add_weight(
                "Phase", shape=shape, dtype=tf.float32, trainable=True
            )
            self.amplitude = tf.constant(tf.ones(shape, dtype=tf.float32))

        else:
            self.phase = tf.constant(tf.zeros(shape, dtype=tf.float32))
            self.amplitude = self.add_weight(
                "Amp",
                shape=shape,
                dtype=tf.float32,
                trainable=True,
                initializer=tf.keras.initializers.RandomUniform(minval=0, maxval=1),
            )

    def call(self, inputs):
        # If it's a DMD binarize the amplitude
        if self.device == "DMD":
            phase = self.phase
            amplitude = self.binarize(self.amplitude)
        # If it's an LCoS normalize the range of the
        else:
            if self.quantization:
                phase = self.nomralize2pi(self.phase)
                phase = self.quantize(phase)
                # phase += np.pi
            else:
                phase = self.phase

            amplitude = self.amplitude

        if self.passthrough:
            a = tf.reduce_sum(
                inputs[0] - inputs[0] + inputs[1] - inputs[1], axis=-1, keepdims=True
            )  # we have to ensure the last dim is 1. TODO not ideal!
            amp = amplitude + tf.stop_gradient(a)
            phi = phase + tf.stop_gradient(a)
        else:
            amp = amplitude * inputs[0]
            phi = phase + inputs[1]
        return amp, phi


class RandomDiffuser(tf.keras.layers.Layer):
    def __init__(self, name="SLM"):
        """
        A parameterized model for spatial light modulators.

        Parameters
        ----------
        device : str, optional
            DESCRIPTION. The default is 'LCoS'.
        passthrough : bool, optional
            DESCRIPTION. The default is True.
        num_frames : int, optional
            DESCRIPTION. The default is 0.
        quantization : int, optional
            DESCRIPTION. The default is 0.
        name : str, optional
            DESCRIPTION. The default is 'SLM'.
        toggle_dmd : TYPE, optional
            DESCRIPTION. The default is False.

        Returns
        -------
        None.

        """
        super(RandomDiffuser, self).__init__(name=name)

    def nomralize2pi(self, x):
        return tf.math.angle(tf.exp(tf.complex(0.0, x)))

    def build(self, input_shape):
        shape = (1, 1) + input_shape[0][2:4] + (1,)
        phi = tf.random.uniform(shape, minval=0, maxval=2 * np.pi)
        amp = tf.random.uniform(shape, minval=0.01, maxval=1)
        self.diffuser_phi = phi
        self.diffuser_amp = amp  # tf.ones_like(phi)

    def call(self, inputs):
        return inputs[0] * self.diffuser_amp, inputs[1] + self.diffuser_phi

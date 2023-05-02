#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 29 11:37:03 2022

@author: hoss
"""

import tensorflow as tf
import numpy as np


class Intensity(tf.keras.layers.Layer):
    def __init__(self, time_average=True, name="Intensity"):
        super(Intensity, self).__init__(name=name)
        self.time_average = time_average

    def build(self, input_shape):
        pass

    def call(self, inputs):
        ints = tf.square(tf.abs(inputs[0]))
        if self.time_average:
            return tf.reduce_sum(ints, axis=1, keepdims=False)
        else:
            return ints


class Amplitude(tf.keras.layers.Layer):
    def __init__(self, sum_ints=False, name="Amplitude"):
        super(Amplitude, self).__init__(name=name)
        self.sum_ints = sum_ints

    def build(self, input_shape):
        pass

    def call(self, inputs):
        if self.sum_ints:
            return tf.sqrt(
                tf.reduce_sum(tf.square(tf.abs(inputs)), axis=1, keepdims=False)
            )
        else:
            return tf.abs(inputs)


class Phase(tf.keras.layers.Layer):
    def __init__(self, name="Phase"):
        super(Phase, self).__init__(name=name)

    def build(self, input_shape):
        pass

    def call(self, inputs):
        return tf.math.angle(inputs)


class NormalizePhase(tf.keras.layers.Layer):
    def __init__(self, zero_one=False, name="NormalizePhase"):
        super(NormalizePhase, self).__init__(name=name)
        self.zero_one = zero_one

    def build(self, input_shape):
        pass

    def call(self, inputs):
        xx = tf.math.angle(tf.math.exp(tf.complex(0.0, inputs))) + np.pi
        if self.zero_one:
            return xx / (2 * np.pi)
        else:
            return xx


class NormalizeEnergy(tf.keras.layers.Layer):
    def __init__(self, name="NormalizePhase"):
        super(NormalizeEnergy, self).__init__(name=name)

    def build(self, input_shape):
        pass

    def call(self, inputs):
        energy = tf.complex(
            tf.cast(
                tf.sqrt(
                    tf.reduce_sum(tf.abs(tf.square(inputs)), axis=[2, 3], keepdims=True)
                ),
                tf.float32,
            ),
            0.0,
        )
        return inputs / tf.stop_gradient(energy)


class LightCompatible(tf.keras.layers.Layer):
    def __init__(self, list_compat=None, dims_compat=-1, name="GOCompatible"):
        """
        Make tensors compatible with GradientOptics layers.

        Parameters
        ----------
        list_compat : TYPE, optional
            DESCRIPTION. The default is None.
        dims_compat : TYPE, optional
            DESCRIPTION. The default is -1.
        name : TYPE, optional
            DESCRIPTION. The default is 'GOCompatible'.

        Returns
        -------
        None.

        """
        assert list_compat in [
            "amplitude",
            "phase",
            None,
        ], "list compatibility not correct.\
            The tensor is either phase or amplitude, provided: {}".format(
            list_compat
        )
        assert dims_compat > -2 and isinstance(
            dims_compat, int
        ), "invalid dimension\
            provided: {}".format(
            dims_compat
        )
        super(LightCompatible, self).__init__(name=name)
        self.list_compat = list_compat
        self.dims_compat = dims_compat

    def build(self, input_shape):
        self.is_list = isinstance(input_shape, list)

        pass

    def call(self, inputs):
        if self.dims_compat != -1:
            if not self.is_list:
                out_ = tf.expand_dims(inputs, axis=self.dims_compat)
            else:
                out_ = [tf.expand_dims(inp, axis=self.dims_compat) for inp in inputs]
        else:
            out_ = inputs

        if (self.list_compat is not None) and (not self.is_list):
            if self.list_compat == "phase":
                out = [tf.ones_like(out_), out_]
            else:
                out = [out_, tf.zeros_like(out_)]
        else:
            out = out_
        return out

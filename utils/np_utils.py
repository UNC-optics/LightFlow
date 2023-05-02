#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  6 11:43:14 2022

@author: hoss
"""
import numpy as np


def normalize_energy(x):
    """
    Normalizes the enregy of the input to be equal to one for each image, frame,
    and depth plane. For tensors of shape [bs, fr, h, w, d] the normalization
    will be performed for each individual d in each fr for individual elements
    in the bs.

    Parameters
    ----------
    x : keras tensor
        has shape [bs, fr, h, w, d].

    Returns
    -------
    keras tensor
        of same shape and dtype.

    """
    assert (
        x.ndim == 5
    ), "Dimensionalities don't seem to match. This function\
        only works for GradientOptics based numpy tensors."
    return x / np.sqrt(np.sum(np.abs(np.square(x)), axis=(2, 3), keepdims=True))


def zeropad_to_1080p(image):
    """
    zero pad images of GO format

    Parameters
    ----------
    image : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    target_shape = np.array([1080, 1920])
    shape = np.array([image.shape[2], image.shape[3]])
    pad_a = np.ceil((target_shape - shape) / 2.0).astype(np.int)
    pad_b = np.floor((target_shape - shape) / 2.0).astype(np.int)

    return np.pad(
        image,
        pad_width=((0, 0), (0, 0), (pad_b[0], pad_a[0]), (pad_b[1], pad_a[1]), (0, 0)),
        constant_values=0,
    )


def phase2bmp(x):
    return np.round((x + np.pi) * 255 / (2 * np.pi)).astype(np.uint8)

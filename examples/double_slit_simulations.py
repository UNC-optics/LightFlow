#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  8 23:20:50 2023

@author: hoss
"""

# Import necessary libraries
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Input
from lightflow.layers.misc import LightCompatible, Intensity
from lightflow.layers.parameterized import SLM
from lightflow.layers.propagation import Propagation
import imageio
# Set up the optical system
shape = (None, 256, 256, 1)
device = "LCoS"
quantization = 256
physics={"wavelength": 660e-9, "pixel_size": 4.5e-6}
z = list(np.linspace(0, 1e-2, 500))  # Propagation distances in meters
method = "asm"
unfiltered = False

# Create the double-slit pattern
def create_double_slit(size, width, separation, height_fraction=0.9):
    """
    Create a double-slit pattern.

    Args:
    size: int, size of the square array
    width: int, width of each slit
    separation: int, separation between the centers of the slits
    height_fraction: float, height of the slits as a fraction of the array size

    Returns:
    pattern: 2D numpy array, double-slit pattern
    """
    pattern = np.zeros((size, size))
    slit_height = int(size * height_fraction)
    
    center = size // 2
    half_sep = separation // 2
    half_width = width // 2
    
    y_start = (size - slit_height) // 2
    y_end = y_start + slit_height
    
    x_start_left = center - half_sep - half_width
    x_end_left = center - half_sep + half_width
    
    x_start_right = center + half_sep - half_width
    x_end_right = center + half_sep + half_width
    
    pattern[y_start:y_end, x_start_left:x_end_left] = 1
    pattern[y_start:y_end, x_start_right:x_end_right] = 1
    
    return pattern

double_slit = create_double_slit(width=2, separation=20, size=256, height_fraction=0.999)

plt.imshow(double_slit, cmap='gray')
plt.title('The modeled aperture: double slit')
plt.show()

# Define the input and layers
target = Input(shape=shape[1:], name="xy")
target_comp = LightCompatible(list_compat="phase", dims_compat=1)(target)
cf = Propagation(physics=physics, z=z, method=method, unfiltered=unfiltered, name="SLM2Image")(target_comp)
out_intensity = Intensity()(cf)

# Create a TensorFlow model and simulate the double-slit experiment
model = tf.keras.Model(inputs=target, outputs=out_intensity)
input_pattern = np.expand_dims(double_slit, axis=(0, -1))
output_patterns = model.predict(input_pattern)

#%%
def plot_output(distance_idx):
    plt.figure(figsize=(10, 10))
    plt.imshow(output_patterns[0, :, :, distance_idx], cmap='gray')
    plt.axis('off')
    plt.show()
    
for i in range(len(z)):
    plot_output(i)

#%%
from ipywidgets import interact, IntSlider
interact(plot_output, distance_idx=IntSlider(min=0, max=len(z)-1, step=1, value=0, description="Propagation distance index"))

#%%
def create_gif(images, output_path, duration):
    # Normalize the images for display
    images = (images - images.min()) / (images.max() - images.min())

    # Convert the normalized images to 8-bit
    images_8bit = (images * 255).astype(np.uint8)

    # Convert grayscale images to RGB
    images_rgb = np.stack([images_8bit] * 3, axis=-1)
    
    print(images_rgb.shape)
    
    # Create a GIF
    with imageio.get_writer(output_path, mode='I', duration=duration) as writer:
        for i in range(images_rgb.shape[2]):
            img = images_rgb[:, :, i, :]
            writer.append_data(img)


output_path = 'animation.gif'
duration = 0.05  # Duration of each frame in seconds

create_gif(output_patterns.squeeze(), output_path, duration)

#%%
plt.figure(figsize=(20, 10))
plt.imshow(output_patterns.squeeze()[128, :, :], cmap='gray')
plt.axis('off')
plt.xlabel('z')
plt.ylabel('x')
plt.show()































































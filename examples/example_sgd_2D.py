#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 29 11:17:46 2022
@author: hoss
"""
from lightflow.optical_setup import cgh
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from time import time
import tensorflow as tf

#%% load target image
images = np.mean(np.array(Image.open("samples/USAF_Resolution_Chart.jpg")), axis=-1)[
    np.newaxis, ..., np.newaxis
].astype(np.float32)

#%% specify setup
num_frames = 1
device = "DMD"
physics = {"wavelength": 660e-9, "pixel_size": [4.5e-6] * 2}
sgd = cgh.CGH_SGDSolver(
    physics=physics,
    device=device,
    shape=(num_frames,) + images.shape[1:],
    quantization=4,
    z=[130e-3],
    method="asm",
    unfiltered=0,
)

#%%
tf.keras.utils.plot_model(
    sgd.model, to_file="model.png", show_shapes=True, show_dtype=True
)

#%% solve and time
t0 = time()
moduls, sims = sgd.solve(images, images, learning_rate=1, iters=300)
print("Elapsed time is", time() - t0)

#%% show
plt.figure(figsize=(20, 20))
plt.imshow(np.squeeze(images[0, ..., 0]), cmap="gray")
plt.show()
plt.figure(figsize=(20, 20))
plt.imshow(sims, cmap="gray")  # , vmax=np.percentile(sims, 90))
plt.show()

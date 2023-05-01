#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  9 16:12:13 2022

@author: hoss
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 29 11:17:46 2022

@author: hoss
"""

from models.SGD import SGD
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from time import time
from utils.np_utils import normalize_energy
import tensorflow as tf
from utils.np_utils import zeropad_to_1080p

#% load target image
multiplane = True

images = np.stack([np.mean(np.array(Image.open('samples/{}.jpg'.format(i))), axis = -1) for i in ['dog', 'bird']] , axis=-1)[None, ...]

#%% specify setup
num_frames = 32
sgd = SGD(device = 'DMD',
          setup = 'Nearfield',
          shape = (num_frames,) + images.shape[1:],
          quantization = 16,
          z = [200e-3, 220e-3],
          wavelength = 660e-9,
          ps = 7.56e-6,
          method = 'asm',
          unfiltered = 0)

#%%
tf.keras.utils.plot_model(
    sgd.model,
    to_file='model.png',
    show_shapes=True,
    show_dtype=True)

#%% solve and time
t0 = time()
moduls, sims = sgd.solve(inp = images,
                         outp = images,
                         learning_rate = 1,
                         iters = 100)
print('Elapsed time is', time() - t0)

#%% show
for i in range(images.shape[-1]):
    plt.figure(figsize=(10,5))
    plt.imshow(np.squeeze(images[..., i]), cmap='gray')
    plt.show()
    plt.figure(figsize=(10,5))
    plt.imshow(sims[..., i], cmap='gray')
    plt.show()
































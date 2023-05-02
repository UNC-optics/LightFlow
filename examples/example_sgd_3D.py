from lightflow.optical_setup import cgh
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from time import time
import tensorflow as tf

#%% Load data
images = np.stack(
    [
        np.mean(np.array(Image.open("samples/{}.jpg".format(i))), axis=-1)
        for i in ["dog", "bird"]
    ],
    axis=-1,
)[None, ...]

#%% specify setup
num_frames = 32
device = "DMD"
physics = {"wavelength": 660e-9, "pixel_size": [4.5e-6] * 2}
sgd = cgh.CGH_SGDSolver(
    physics=physics,
    device=device,
    shape=(num_frames,) + images.shape[1:],
    quantization=4,
    z=[130e-3, 150e-3],
    method="asm",
    unfiltered=0,
)

#%%
tf.keras.utils.plot_model(
    sgd.model, to_file="model.png", show_shapes=True, show_dtype=True
)

#%% solve and time
t0 = time()
moduls, sims = sgd.solve(inp=images, outp=images, learning_rate=1, iters=100)
print("Elapsed time is", time() - t0)

#%% show
for i in range(images.shape[-1]):
    plt.figure(figsize=(10, 5))
    plt.imshow(np.squeeze(images[..., i]), cmap="gray")
    plt.show()
    plt.figure(figsize=(10, 5))
    plt.imshow(sims[..., i], cmap="gray")
    plt.show()

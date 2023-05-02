from time import time
from ..layers.parameterized import SLM, RandomDiffuser
from ..layers.propagation import Propagation, Lens
from ..layers.misc import Intensity, LightCompatible
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Lambda
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from ..ml.losses import accuracy
from PIL import Image

#%%
class CGH_SGDSolver:
    def __init__(
        self,
        physics={"wavelength": 660e-9, "pixel_size": 4.5e-6},
        device="DMD",
        shape=(1, 1080, 1920, 1),
        quantization=8,
        z=[149e-3],
        method="asm",
        unfiltered=0,
    ):

        self.device = device

        assert device in [
            "LCoS",
            "DMD",
        ], "Device name not supported. Either LCoS or DMD"
        assert isinstance(z, list), "z must be a list"
        assert len(z) == shape[-1], "Not enough z values proviced for multiple planes"

        target = Input(shape=shape[1:], name="xy")

        target_comp = LightCompatible(list_compat="phase", dims_compat=1)(target)

        lcos = SLM(
            device=device,
            passthrough=True,
            num_frames=shape[0],
            quantization=quantization,
            name="SLM",
        )(target_comp)

        cf = Propagation(
            physics=physics, z=z, method=method, unfiltered=unfiltered, name="SLM2Image"
        )(lcos)

        out_intensity = Intensity()(cf)

        self.model = Model(target, out_intensity)

    def solve(self, inp, outp, loss=accuracy, learning_rate=2, iters=100):
        self.model.compile(optimizer="adam", loss=loss)
        self.model.optimizer.lr = learning_rate
        self.model.fit(inp, outp, batch_size=1, epochs=iters, verbose=1)

        simulation = np.squeeze(self.model.predict(inp))

        if self.device == "DMD":
            modulation = np.squeeze(self.model.get_weights()[0] > 0.5) * 1
        else:
            modulation = np.angle(np.exp(1j * np.squeeze(self.model.get_weights()[0])))

        return modulation, simulation

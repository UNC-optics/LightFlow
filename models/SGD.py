from time import time
from layers.parameterized import SLM, RandomDiffuser
from layers.propagation import NearField, MLA, Lens
from layers.misc import Intensity
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Lambda
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from losses import accuracy
from PIL import Image
from layers.misc import LightCompatible

#%%
class SGD:
    def __init__(self,
             device = 'DMD',
             setup = 'Nearfield',
             shape = (1, 1080, 1920, 1),
             quantization = 8,
             z = 149e-3,
             wavelength = 535e-9,
             ps = 7.56e-6,
             method = 'asm',
             unfiltered = 0,
             focal_length = 200,
             mla_f = 4.2e-3,
             ml_resolution = (11, 11),
             random_diffuser = False):
        
        self.device = device
        
        assert device in ['LCoS', 'DMD'], 'Device name not supported. Either LCoS or DMD'
        assert setup in ['Fourier', 'Nearfield']
        
        target = Input(shape = shape[1:], name='xy')
        
        target_comp = LightCompatible(list_compat = 'phase', dims_compat = 1)(target)
        
        lcos = SLM(device = device, passthrough = True, num_frames = shape[0], quantization = quantization, name = 'SLM')(target_comp)
        
        if setup == 'Nearfield':
            cf = NearField(shape = shape[1:],
                           z = [z[0], z[1] + mla_f] if setup == 'LF' else z, # because we wanna optimize for speed
                           wavelength = wavelength,
                           ps = ps,
                           method = method,
                           unfiltered = unfiltered,
                           name = 'SLM2{}'.format('MLA' if setup == 'LF' else 'Image'))(lcos)
        else:
            cf = Lens(name = 'SLM2{}'.format('MLA' if setup == 'LF' else 'Image'))(lcos)
        
        out_intensity = Intensity()(cf)
        
        self.model = Model(target, out_intensity)
        
    def solve(self,
              inp,
              outp,
              loss = accuracy,
              learning_rate = 2,
              iters = 100):
        # TODO check dimensions
        self.model.compile(optimizer = 'adam',
                      loss = loss)
        self.model.optimizer.lr = learning_rate
        self.model.fit(inp,
                       outp,
                       batch_size = 1,
                       epochs = iters,
                       verbose = 1)
        
        simulation = np.squeeze(self.model.predict(inp))
        
        if self.device == 'DMD':
            modulation = np.squeeze(self.model.get_weights()[0] > 0.5) * 1
        else:
            modulation = np.angle(np.exp(1j*np.squeeze(self.model.get_weights()[0])))
        
        return modulation, simulation
        




































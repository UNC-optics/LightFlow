# -*- coding: utf-8 -*-
"""
Created on Fri May 20 13:18:31 2022

@author: Admin
"""
import tensorflow as tf
import numpy as np

class NearField(tf.keras.layers.Layer):
    def __init__(self, shape, z, wavelength, ps, method, unfiltered = 0, name = 'Propagation'):
        '''
        Keras layer for complex wave propagation from a single plane to (multiple)
        plane(s) at distance(s) specified in z. This layer implements fresnel
        and Angular spectrum method propagation models. The inptut is a list of
        two keras layers, representing the amplitude (first list item)
        and phase (second list item) of the complex field that will be propagated
        in free space. The dimensions of the inputs are the same and in the form
        of [batch, frames, h, w, 1]. The last dimension has to be equal to 1 for
        Keras compatibility. The output is also a list of amplitude and phase
        corresponding to the complex field at the imag plane.
        
        Parameters
        ----------
        shape : list or tuple
            Describes the shape of the input to the layer. Last dimension must
            be 1, aka shape must be (h, w, 1).
        z : list or int
            The distance(s) to whih th elayer will propagate the complex field.
        wavelength : float
            Wavelength of the light, in meters.
        ps : float
            pixel size in meters.
        method : str
            String determining the propagation mdoel to be used. Either 'asm',
            or 'fresnel'.
        unfiltered : int
            Determines whether unfiltered ASM will be used or not. The number
            determines how many hgher orders will be used.
        name : str, optional
            Name of this layer.
            
        Returns
        -------
        list containing the 
        
        '''
        super(NearField, self).__init__(name = name)
        
        self.method = method
        # this is for 3D compatibility in the future
        if isinstance(z, (list, tuple)):
            self.Z = z
        else:
            self.Z = [z]
        
        self.wavelength = wavelength
        self.ps = ps
        
        self.shape = shape
        
        self.unfiltered = unfiltered
        
        self.__get_freqMeshGrid()
        self.__get_functions()
        

    def __get_functions(self):
        if self.unfiltered:
            self.mask = (np.sinc(np.pi*self.ps*self.big_fxx)*np.sinc(np.pi*self.ps*self.big_fyy)).astype(np.complex64)
            # Tile the FFT of CF to create sampling effect
            @tf.function
            def tiling(x):
                return tf.tile(x, [1, 1, 1, self.unfiltered, self.unfiltered])
            self.SpectrumMapping = tiling
            # Downsample to get final image size
            @tf.function
            def pooling(x):
                return tf.nn.avg_pool3d(x[..., tf.newaxis],
                                        (1, self.unfiltered, self.unfiltered),
                                        (1, self.unfiltered, self.unfiltered),
                                        'VALID',
                                        data_format='NHWC')
            
            self.downsample = lambda x: tf.complex(pooling(tf.abs(x)), 0.)*tf.exp(tf.complex(0., pooling(tf.math.angle(x))))
            
            
        else:
            self.mask = 1
            self.SpectrumMapping = lambda x: x
            self.downsample = lambda x: x


    def __get_freqMeshGrid(self):
        if self.unfiltered:
            fx = np.fft.fftshift(np.fft.fftfreq(self.shape[1]*self.unfiltered, d = self.ps))
            fy = np.fft.fftshift(np.fft.fftfreq(self.shape[0]*self.unfiltered, d = self.ps))
            self.big_fxx, self.big_fyy = np.meshgrid(fx, fy)
        
        fx = np.fft.fftshift(np.fft.fftfreq(self.shape[1], d = self.ps))
        fy = np.fft.fftshift(np.fft.fftfreq(self.shape[0], d = self.ps))
        self.short_fxx, self.short_fyy = np.meshgrid(fx, fy)


    def get_ASM(self):
        H = []
        
        argument = (2 * np.pi)**2 * ((1. / self.wavelength) ** 2 - self.short_fxx ** 2 - self.short_fyy ** 2)
        
        #Calculate the propagating and the evanescent (complex) modes
        tmp = np.sqrt(np.abs(argument))
        kz = np.where(argument >= 0, tmp, 1j*tmp)
        
        for z in self.Z:
            H.append(np.exp(1j * kz * z).astype(np.complex64))
        return H


    def get_Fresnel(self):
        H = []
        
        fx = self.short_fxx/self.ps/self.shape[1]
        fy = self.short_fyy/self.ps/self.shape[0]
        argument = -1j * np.pi * self.wavelength * (fx**2 + fy**2)
        for z in self.Z:
            H.append(np.exp(argument * z).astype(np.complex64))
        return H


    def build(self, input_shape):
        if self.method == 'fresnel':
            self.H = self.get_Fresnel()
        elif self.method == 'asm':
            self.H = self.get_ASM()
        
        if self.unfiltered:
            self.H = [np.pad(h, [[self.shape[0]]*int((self.unfiltered-1)/2),
                                 [self.shape[1]]*int((self.unfiltered-1)/2)]) * self.mask for h in self.H]
        # first two dimensions are batch and frames
        self.H = [tf.constant(h[np.newaxis, np.newaxis, ...]) for h in self.H]
        
        
    def call(self, inputs):
        # Having the squeeze ensures that we propagate a single plane
        amp = tf.complex(tf.squeeze(inputs[0], axis=-1), 0.) #
        phi = tf.complex(0., tf.squeeze(inputs[1], axis=-1))
        cf = tf.cast(amp * tf.exp(phi), tf.complex64)
        c = tf.signal.fftshift(tf.signal.fft2d(cf), axes=[2, 3])
        
        mapped = self.SpectrumMapping(c)
        
        amplitudes = []
        phases = []
        
        for h in self.H:
            a = mapped * h
            mul = tf.signal.ifftshift(a, axes=[2, 3])
            ift = self.downsample(tf.signal.ifft2d(mul))
            amplitudes.append(tf.abs(ift)[..., tf.newaxis])
            phases.append(tf.math.angle(ift)[..., tf.newaxis])
        
        return tf.concat(amplitudes, axis=-1), tf.concat(phases, axis=-1)


class Lens(tf.keras.layers.Layer):
    def __init__(self, name = 'Propagation'):
        super(Lens, self).__init__(name = name)

    def build(self, input_shape):
        pass


    def call(self, inputs):
        amp = tf.complex(tf.squeeze(inputs[0], axis=-1), 0.) #
        phi = tf.complex(0., tf.squeeze(inputs[1], axis=-1))
        cf = tf.cast(amp * tf.exp(phi), tf.complex64)
        
        fft = tf.signal.ifftshift(tf.signal.fft2d(tf.signal.fftshift(cf, axes = [2, 3])), axes = [2, 3])[..., tf.newaxis]
        return [tf.cast(tf.abs(fft), dtype=tf.float32), tf.cast(tf.math.angle(fft), dtype=tf.float32)]


class MLA(tf.keras.layers.Layer):
    def __init__(self, mla_f, ml_resolution, wavelength, ps, name = 'MLA'):
        
        super(MLA, self).__init__(name = name)
        
        self.wavelength = wavelength
        self.ps = ps
        self.f = mla_f
        self.ml_resolution = ml_resolution#tf.TensorShape(ml_resolution)


    def __get_mlacf(self, f, shape, ps, ml_resolution, wavelength):
        shape_ = (1078, 1914)
        ml_resolution_ = (11, 11)
        UY = ps * (tf.cast(tf.math.floormod(tf.range(1, shape_[0]+1), ml_resolution_[0]), tf.float32) - ml_resolution_[0]/2)
        UX = ps * (tf.cast(tf.math.floormod(tf.range(1, shape_[1]+1), ml_resolution_[1]), tf.float32) - ml_resolution_[1]/2)
        XX, YY = tf.meshgrid(UX, UY)
        phase = tf.cast(-(2 * np.pi / wavelength) * (XX**2 + YY**2) / (2 * f), tf.float32)
        return tf.constant(tf.math.exp(tf.complex(0., phase))[np.newaxis, np.newaxis, ..., np.newaxis], dtype=tf.complex64)


    def build(self, input_shape):
        mla = self.__get_mlacf(self.f, input_shape, self.ps, self.ml_resolution, self.wavelength)
        self.mla = [tf.abs(mla), tf.math.angle(mla)]


    def call(self, inputs):
        return [inputs[0] * self.mla[0], inputs[1] * self.mla[1]]


#%%
# import numpy as np
# import tensorflow as tf
# import matplotlib.pyplot as plt
# shape = (960, 960, 1)
# ps = 4.25e-6
# wavelength = 660e-9

# fx = np.fft.fftshift(np.fft.fftfreq(shape[1], d = ps))
# fy = np.fft.fftshift(np.fft.fftfreq(shape[0], d = ps))
# fxx, fyy = np.meshgrid(fx, fy)

# argument = (2 * np.pi)**2 * ((1. / wavelength) ** 2 - fxx ** 2 - fyy ** 2)

# tmp = np.sqrt(np.abs(argument))
# exclusive_exp = 1j * np.where(argument >= 0, tmp, 1j*tmp).astype(np.complex64)[tf.newaxis, ...]

# #%%
# N = 10
# zs = np.random.randint(20, 100, (N, 1, 1)) * 1e-3
# print(exclusive_exp.shape)
# print(zs.shape)
# h = tf.cast(tf.exp(tf.constant(exclusive_exp) * tf.complex(tf.cast(zs, tf.float32), 0.)), tf.complex64)
# print(h.shape)

# #%%
# for hh in h:
#     plt.imshow(np.angle(hh))
#     plt.show()

# #%%
# plt.imshow(np.angle(hh))
# plt.show()

#%%
# # %%
# resolution = (1080, 1920, 1)
# ps = 4.25e-6
# wavelength = 660e-9
# z = 32e-3
# input_fn = lambda x: [x[0], x[1], tf.complex(x[2], 0.)]
# # input_fn = lambda x: [tf.ones_like(x), x, z]
# output_fn = lambda x: tf.square(tf.abs(x[0]))
# prop = PropagateZ(resolution, wavelength, ps, input_fn = input_fn, output_fn = output_fn, name = 'Propagation')
# prop.build(resolution)

# phase = (np.mean(np.array(Image.open('solutions/660nm/Calibration_22.0mm.bmp')).astype(np.float32)*2*np.pi/255., axis=-1, keepdims = True))[np.newaxis, ...]

# #%%
# for i in range(20):
#     sim = prop([np.ones_like(phase), phase, 22e-3+((i-10)/(30))*1e-3])
    
#     plt.figure(figsize=(20,10))
#     plt.imshow(sim[0], cmap='gray')
#     plt.show()

# #%%
# prop = Propagate(resolution, z, wavelength, ps, 'asm', unfiltered = 0, input_fn = None, output_fn = None, name = 'Propagation')

# #%%
# prop.build((1080, 1920, 1))

# #%%
# image_layer = np.squeeze((prop([np.ones_like(phase), phase])[0].numpy())**2)

# #%%
# plt.figure(figsize=(20,10))
# plt.imshow(image_layer, cmap='gray')
# plt.axis('off')
# plt.show()

# #%%
# from SGD_KerasLayer import SGD

# sgd = SGD(resolution[:-1], ps, wavelength, [z], prop_model='hod')

# int_sgd = sgd.simulation(phase[0])

# plt.figure(figsize=(20,10))
# plt.imshow(int_sgd, cmap='gray')
# plt.axis('off')
# plt.show()

#%%


#%%


#%% Test the propagation module:
# from PIL import Image
# import numpy as np
# import matplotlib.pyplot as plt
# phase = ((np.mean(np.array(Image.open('solutions/660nm/Science_22.0mm.bmp')), axis=-1) / 255.)[np.newaxis, ..., np.newaxis] * 2 * np.pi).astype(np.float32)

# #%%
# prop = Propagate((1080, 1920), 22e-3, 660e-9, 4.25e-6, model='asm', mode='prop', name='image')
# prop.build(phase.shape)

# #%%
# sim = prop([np.ones_like(phase), phase]).numpy()

# plt.imshow(np.squeeze(sim), cmap='gray')
# plt.show()

#%%







































































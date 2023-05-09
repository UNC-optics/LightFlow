# -*- coding: utf-8 -*-
"""
Created on Fri May 20 13:18:31 2022

@author: Admin
"""
import tensorflow as tf
import numpy as np

def fft_frequency(n, d=1.0):
    val = 1.0 / (n * d)
    results = tf.range(0, n, dtype=tf.float32)
    results = tf.where(results <= n // 2, results, results - n)
    return results * val

class Propagation(tf.keras.layers.Layer):
    def __init__(self, physics, z, method, unfiltered=0, name="Propagation"):
        """
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

        """
        super(Propagation, self).__init__(name=name)

        self.method = method.lower()
        assert self.method in ['asm','fresnel']
        # this is for 3D compatibility in the future
        if isinstance(z, (list, tuple)):
            self.Z = z
        else:
            self.Z = [z]
        
        if isinstance(physics["pixel_size"], (list, tuple)):
            self.ps = physics["pixel_size"]
        else:
            self.ps = [physics["pixel_size"]] * 2

        self.physics = physics
        self.wavelength = physics["wavelength"]

        self.unfiltered = unfiltered

    def __get_functions(self):
        if self.unfiltered:
            self.mask = (
                np.sinc(np.pi * self.ps[0] * self.big_fxx)
                * np.sinc(np.pi * self.ps[1] * self.big_fyy)
            ).astype(np.complex64)
            # Tile the FFT of CF to create sampling effect
            @tf.function
            def tiling(x):
                return tf.tile(x, [1, 1, 1, self.unfiltered, self.unfiltered])

            self.SpectrumMapping = tiling
            # Downsample to get final image size
            @tf.function
            def pooling(x):
                return tf.nn.avg_pool3d(
                    x[..., tf.newaxis],
                    (1, self.unfiltered, self.unfiltered),
                    (1, self.unfiltered, self.unfiltered),
                    "VALID",
                    data_format="NHWC",
                )

            self.downsample = lambda x: tf.complex(pooling(tf.abs(x)), 0.0) * tf.exp(
                tf.complex(0.0, pooling(tf.math.angle(x)))
            )

        else:
            self.mask = 1
            self.SpectrumMapping = lambda x: x
            self.downsample = lambda x: x

    def __get_freqMeshGrid(self):
        if self.unfiltered:
            fx = tf.signal.fftshift(
                fft_frequency(self.shape[-2] * self.unfiltered, d=self.ps[0])
            )
            fy = tf.signal.fftshift(
                fft_frequency(self.shape[-3] * self.unfiltered, d=self.ps[1])
            )
            self.big_fxx, self.big_fyy = tf.meshgrid(fx, fy)
        print(type(self.ps))
        print(type(self.shape))
        fx = tf.signal.fftshift(fft_frequency(self.shape[-2], d=self.ps[0]))
        fy = tf.signal.fftshift(fft_frequency(self.shape[-3], d=self.ps[1]))
        self.short_fxx, self.short_fyy = tf.meshgrid(fx, fy)

    def get_ASM(self):
        H = []

        argument = (2 * np.pi) ** 2 * (
            (1.0 / self.wavelength) ** 2 - self.short_fxx**2 - self.short_fyy**2
        )

        # Calculate the propagating and the evanescent (complex) modes
        tmp = tf.sqrt(tf.abs(argument))
        # kz = tf.where(argument >= 0, tmp, 1j * tmp)
        kz = tf.where(argument >= 0, tf.complex(tmp, 0.), tf.complex(0., tmp))


        for z in self.Z:
            H.append(tf.cast(tf.exp(1j * kz * z), tf.complex64))
        return H

    def get_Fresnel(self):
        H = []

        fx = self.short_fxx / self.ps[0] / self.shape[-2]
        fy = self.short_fyy / self.ps[1] / self.shape[-3]
        argument = tf.complex(0., -1 * np.pi * self.wavelength * (fx**2 + fy**2))
        for z in self.Z:
            H.append(tf.cast(tf.exp(argument * z), tf.complex64))
        return H

    def build(self, input_shape):
        assert (
            input_shape[0] == input_shape[1]
        ), "Phase and Amplitude dimensions mismatch."
        self.shape = input_shape[0]
        self.__get_freqMeshGrid()
        self.__get_functions()
        if self.method == "fresnel":
            self.H = self.get_Fresnel()
        elif self.method == "asm":
            self.H = self.get_ASM()

        if self.unfiltered:
            self.H = [
                np.pad(
                    h,
                    [
                        [self.shape[-3]] * int((self.unfiltered - 1) / 2),
                        [self.shape[-2]] * int((self.unfiltered - 1) / 2),
                    ],
                )
                * self.mask
                for h in self.H
            ]
        # first two dimensions are batch and frames
        self.H = [tf.constant(h[np.newaxis, np.newaxis, ...]) for h in self.H]

    def call(self, inputs):
        # Having the squeeze ensures that we propagate a single plane
        amp = tf.complex(tf.squeeze(inputs[0], axis=-1), 0.0)  #
        phi = tf.complex(0.0, tf.squeeze(inputs[1], axis=-1))
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
    def __init__(self, physics, f, name="Propagation"):
        super(Lens, self).__init__(name=name)
        self.physics = physics

    def build(self, input_shape):
        shape = input_shape[0]
        psx = np.abs(
            self.f
            * self.physics["wavelength"]
            / (shape[-2] * self.physics["pixel_size"][0])
        )
        psy = np.abs(
            self.f
            * self.physics["wavelength"]
            / (shape[-3] * self.physics["pixel_size"][1])
        )
        self.physics["pixel_size"] = [psx, psy]
        pass

    def call(self, inputs):
        amp = tf.complex(tf.squeeze(inputs[0], axis=-1), 0.0)  #
        phi = tf.complex(0.0, tf.squeeze(inputs[1], axis=-1))
        cf = tf.cast(amp * tf.exp(phi), tf.complex64)

        fft = tf.signal.ifftshift(
            tf.signal.fft2d(tf.signal.fftshift(cf, axes=[2, 3])), axes=[2, 3]
        )[..., tf.newaxis]
        return [
            tf.cast(tf.abs(fft), dtype=tf.float32),
            tf.cast(tf.math.angle(fft), dtype=tf.float32),
        ]

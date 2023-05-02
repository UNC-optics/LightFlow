#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  2 01:48:36 2023

@author: hoss
"""
from setuptools import setup, find_packages

with open('requirements.txt') as f:
    install_requires = f.read().splitlines()

setup(
    name='lightflow',
    version='0.1',
    description='Wave optics simulation in Tensorflow-Keras',
    author='M. Hossein Eybposh',
    author_email='hosseybposh@gmail.com',
    url='https://github.com/UNC-optics/LightFlow',
    packages=find_packages(),
    install_requires=install_requires,
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
)

import lightflow as lf

a = lf.layers.parameterized.SLM()
#!/usr/bin/env python

from distutils.core import setup
from setuptools import find_packages

setup(
    name='rebar',
    version='0.1',
    description='My reinforcement learning toolkit',
    author='Andy Jones',
    author_email='andyjones.ed@gmail.com',
    url='http://www.github.com/andyljones/rebar',
    packages=find_packages(include=['rebar*']),
    python_requires='>=3.6',
    install_requires=[
        'numpy>=1.18',
        'scipy>=1.4',
        'torch>=1.7',
        'torchvision>=0.6',
        'matplotlib>=3'],
    package_data={'': ['data/*']}
)
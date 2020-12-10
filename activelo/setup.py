#!/usr/bin/env python

from distutils.core import setup
from setuptools import find_packages

setup(
    name='activelo',
    version='0.1',
    description='Actively learns Elo rankings',
    author='Andy Jones',
    author_email='andyjones.ed@gmail.com',
    url='http://www.github.com/andyljones/activelo',
    packages=find_packages(include=['activelo*']),
    python_requires='>=3.6',
    install_requires=[
        'numpy>=1.18',
        'scipy>=1.5',
        'torch>=1.7',
        'torchvision>=0.6',
        'geotorch>=0.1',
        'matplotlib>=3',
        'rebar @ git+https://github.com/andyljones/rebar.git@master#egg=rebar'],
    package_data={'': ['data/*']}
)
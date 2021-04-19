#!/usr/bin/env python

from distutils.core import setup
from setuptools import find_packages

setup(
    name='boardlaw',
    version='0.1',
    description='Codebase for Scaling Scaling Laws',
    author='Andy Jones',
    author_email='me@andyljones.com',
    url='http://andyljones.com/boardlaw',
    packages=find_packages(include=['boardlaw*', 'pavlov*', 'rebar*']),
    python_requires='>=3.6',
    install_requires=[
        'cloudpickle>=1.6',
        'numpy>=1.20',
        'torch>=1.7.1',
        'pandas>=1.2',
        'portalocker>=2.2',
        'pytest>=6.2',
        'aljpy==0.7',
        'requests>=2.24',
        'bokeh>=2.2',
        'matplotlib>=3.3',
        'scipy>=1.6',
        'av>=8',
        'loky>=1.6',
        'sqlalchemy>=1.4',
        'plotnine>=0.7',
        'geotorch@git+https://github.com/Lezcano/geotorch#egg=geotorch'],
    extras_require={},
    package_data={'boardlaw.arena': ['data/*'], 'boardlaw.arena.live': ['data/*']})

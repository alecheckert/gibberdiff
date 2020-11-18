#!/usr/bin/env python
"""
setup.py

"""
import setuptools

setuptools.setup(
    name="gibberdiff",
    version="1.0",
    packages=setuptools.find_packages(),
    author="Alec Heckert",
    author_email="alecheckert@gmail.com",
    description="Gibbs sampler for finite-state mixtures of regular Brownian motions"
)

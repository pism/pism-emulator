#!/usr/bin/env python
# -*- coding: utf-8 -*-

# pylint: skip-file

import os
import re
import sys
from pathlib import Path

from pkg_resources import parse_version
from setuptools import find_packages, setup

# load version form _version.py

with open(str(Path(".", "VERSION").absolute())) as version_file:
    version = version_file.read().strip()

# Dependencies of pism-emulator
requirements = [
    "numpy>=1.10.0",
    "scipy>=0.18.0",
    "sklearn",
    "PyDOE",
    "SALib",
    "pandas>=0.18.1",
    "matplotlib>=2.2.2",
    "GPy>=1.9.0",
    "seaborn>=0.8",
    "xarray",
]

packages = find_packages(".")


setup(
    name="pismemulator",
    version=version,
    author="Andy Aschwanden, Rachel S Chen",
    author_email="andy.aschwanden@gmail.com",
    description=("Gaussian process emulators for PISM"),
    license="GPL 3.0",
    keywords="machine-learning gaussian-processes PISM",
    url="https://github.com/pism/pism-emulator",
    project_urls={
        "Bug Tracker": "https://github.com/pism/pism-emulator/issues",
        "Documentation": "https://github.com/pism/pism-emulator",
        "Source Code": "https://github.com/pism/pism-emulator",
    },
    packages=packages,
    install_requires=requirements,
    include_package_data=True,
    classifiers=[
        "License :: OSI Approved :: Apache Software License",
        "Natural Language :: English",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: Microsoft :: Windows",
        "Operating System :: POSIX :: Linux",
        "Programming Language :: Python :: 3.6",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)

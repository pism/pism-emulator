#!/usr/bin/env python
# -*- coding: utf-8 -*-

# pylint: skip-file

from pathlib import Path
from setuptools import setup

with open(str(Path(".", "VERSION").absolute())) as version_file:
    version = version_file.read().strip()

packages = ["pismemulator"]

# Dependencies of pism-emulator
requirements = [
    "numpy>=1.10.0",
    "scipy>=0.18.0",
    "sklearn",
    "PyDOE",
    "SALib",
    "pandas>=0.18.1",
    "pathlib",
    "matplotlib>=2.2.2",
    "openpyxl>=3.0.0",
    "seaborn>=0.8",
    "xarray",
    "torch",
    "pytorch-lightning",
]

setup(
    name="pismemulator",
    version=version,
    author="Andy Aschwanden, Douglas C Brinkerhoff",
    author_email="andy.aschwanden@gmail.com",
    description=("Emulators for PISM"),
    license="GPL 3.0",
    keywords="machine-learning pytorch PISM",
    url="https://github.com/pism/pism-emulator",
    project_urls={
        "Bug Tracker": "https://github.com/pism/pism-emulator/issues",
        "Documentation": "https://github.com/pism/pism-emulator",
        "Source Code": "https://github.com/pism/pism-emulator",
    },
    packages=packages,
    install_requires=requirements,
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

"""Installation script for the 'vr_mjlab' python package."""

from setuptools import setup

# Minimum dependencies required prior to installation
INSTALL_REQUIRES = [
    "mjlab==1.4.0",
    "mujoco-warp==3.8.1",
    "mujoco==3.8.1",
    "warp-lang==1.13.0",
]

# Installation operation
setup(
    name="vr_mjlab",
    packages=["src"],
    version="0.0.1",
    author="VinRobotics",
    license="Apache-2.0",
    install_requires=INSTALL_REQUIRES,
)

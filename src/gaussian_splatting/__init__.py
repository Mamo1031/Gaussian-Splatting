"""Gaussian Splatting package

A Python package for generating 3D models from multiple images using Gaussian Splatting.
"""

from .colmap_utils import (
    run_colmap,
    load_colmap_data,
    read_cameras_binary,
    read_images_binary,
    read_points3D_binary,
)
from .gaussian_model import GaussianModel
from .renderer import render, Camera
from .trainer import Trainer

__all__ = [
    "run_colmap",
    "load_colmap_data",
    "read_cameras_binary",
    "read_images_binary",
    "read_points3D_binary",
    "GaussianModel",
    "render",
    "Camera",
    "Trainer",
]

__version__ = "0.1.0"


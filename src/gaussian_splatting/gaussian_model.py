"""Gaussian Splatting model definition

Provides functionality for managing, initializing, and saving/loading Gaussian parameters.
"""

import torch
import numpy as np
from typing import List
from plyfile import PlyData, PlyElement


class GaussianModel:
    """Gaussian Splatting model

    Manages position, color, opacity, scale, and rotation for each Gaussian.
    """

    def __init__(self):
        """Initialize Gaussian model."""
        self._xyz = None  # Position [N, 3]
        self._features_dc = None  # Spherical harmonics DC component (color) [N, 3]
        self._features_rest = None  # Remaining spherical harmonics components [N, K, 3]
        self._opacity = None  # Opacity [N, 1]
        self._scaling = None  # Scale [N, 3]
        self._rotation = None  # Rotation (quaternion) [N, 4]

    @property
    def get_xyz(self):
        """Get Gaussian positions."""
        return self._xyz

    @property
    def get_features(self):
        """Get Gaussian features (color)."""
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)

    @property
    def get_opacity(self):
        """Get Gaussian opacity."""
        return self._opacity

    @property
    def get_scaling(self):
        """Get Gaussian scale."""
        return self._scaling

    @property
    def get_rotation(self):
        """Get Gaussian rotation."""
        return self._rotation

    def create_from_pcd(
        self,
        pcd: np.ndarray,
        spatial_lr_scale: float = 0.01
    ):
        """Initialize Gaussian model from point cloud.

        Args:
            pcd: Point cloud data [N, 3] (xyz) or [N, 6] (xyz + rgb)
            spatial_lr_scale: Spatial learning rate scale
        """
        if pcd.shape[1] == 3:
            xyz = pcd
            rgb = np.ones((pcd.shape[0], 3)) * 0.5  # Default to gray
        else:
            xyz = pcd[:, :3]
            rgb = pcd[:, 3:6] / 255.0  # Normalize from 0-255 to 0-1

        self._xyz = torch.nn.Parameter(torch.tensor(xyz, dtype=torch.float32).cuda())

        # Convert color to spherical harmonics DC component
        rgb_tensor = torch.tensor(rgb, dtype=torch.float32).cuda()
        self._features_dc = torch.nn.Parameter(rgb_tensor.unsqueeze(1))  # [N, 1, 3]
        # Initialize remaining components to zero
        self._features_rest = torch.nn.Parameter(
            torch.zeros((xyz.shape[0], 15, 3), dtype=torch.float32).cuda()
        )

        # Initialize opacity (pre-sigmoid value)
        opacities = torch.ones((xyz.shape[0], 1), dtype=torch.float32) * 0.1
        self._opacity = torch.nn.Parameter(opacities.cuda())

        # Initialize scale
        scales = torch.ones((xyz.shape[0], 3), dtype=torch.float32) * np.log(spatial_lr_scale)
        self._scaling = torch.nn.Parameter(scales.cuda())

        # Initialize rotation (unit quaternion)
        rots = torch.zeros((xyz.shape[0], 4), dtype=torch.float32)
        rots[:, 0] = 1.0  # Set w component to 1
        self._rotation = torch.nn.Parameter(rots.cuda())

    def create_from_colmap_points(
        self,
        points3D: dict,
        spatial_lr_scale: float = 0.01
    ):
        """Initialize Gaussian model from COLMAP 3D point cloud.

        Args:
            points3D: COLMAP points3D dictionary
            spatial_lr_scale: Spatial learning rate scale
        """
        xyz_list = []
        rgb_list = []

        for point_id, point_data in points3D.items():
            xyz_list.append(point_data["xyz"])
            rgb_list.append(point_data["rgb"] / 255.0)

        xyz = np.array(xyz_list)
        rgb = np.array(rgb_list)

        pcd = np.concatenate([xyz, rgb], axis=1)
        self.create_from_pcd(pcd, spatial_lr_scale)

    def get_parameters(self) -> List[torch.nn.Parameter]:
        """Get list of trainable parameters.

        Returns:
            List of parameters
        """
        return [
            self._xyz,
            self._features_dc,
            self._features_rest,
            self._opacity,
            self._scaling,
            self._rotation,
        ]

    def save_ply(self, path: str):
        """Save Gaussian model as .ply file.

        Args:
            path: Output path
        """
        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)

        f_dc = self._features_dc.detach().cpu().numpy()

        # Convert from spherical harmonics to RGB (simplified: DC component only)
        opacities = torch.sigmoid(self._opacity).detach().cpu().numpy()
        scales = torch.exp(self._scaling).detach().cpu().numpy()
        rotations = self._rotation.detach().cpu().numpy()

        # Calculate RGB from DC component
        rgb = (f_dc[:, 0, :] + 0.28209479177387814) / 1.0  # SH coefficient normalization
        rgb = np.clip(rgb, 0, 1)

        dtype_full = [(attribute, 'f4') for attribute in ['x', 'y', 'z', 'nx', 'ny', 'nz']]
        dtype_full += [(attribute, 'f4') for attribute in ['f_dc_0', 'f_dc_1', 'f_dc_2']]
        dtype_full += [(attribute, 'f4') for attribute in ['opacity']]
        dtype_full += [(attribute, 'f4') for attribute in ['scale_0', 'scale_1', 'scale_2']]
        dtype_full += [(attribute, 'f4') for attribute in ['rot_0', 'rot_1', 'rot_2', 'rot_3']]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, f_dc[:, 0, :], opacities, scales, rotations), axis=1)
        elements[:] = list(map(tuple, attributes))

        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

    @classmethod
    def load_ply(cls, path: str) -> 'GaussianModel':
        """Load Gaussian model from PLY file.

        Args:
            path: Path to PLY file

        Returns:
            Loaded GaussianModel instance
        """
        plydata = PlyData.read(path)
        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])), axis=1)

        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]
        scales = np.asarray([plydata.elements[0]["scale_0"],
                             plydata.elements[0]["scale_1"],
                             plydata.elements[0]["scale_2"]]).transpose(1, 0)
        rotations = np.asarray([plydata.elements[0]["rot_0"],
                                plydata.elements[0]["rot_1"],
                                plydata.elements[0]["rot_2"],
                                plydata.elements[0]["rot_3"]]).transpose(1, 0)

        f_dc = np.stack((np.asarray(plydata.elements[0]["f_dc_0"]),
                          np.asarray(plydata.elements[0]["f_dc_1"]),
                          np.asarray(plydata.elements[0]["f_dc_2"])), axis=1)[:, np.newaxis, :]

        model = cls()
        model._xyz = torch.nn.Parameter(torch.tensor(xyz, dtype=torch.float32).cuda())
        model._features_dc = torch.nn.Parameter(torch.tensor(f_dc, dtype=torch.float32).cuda())
        model._features_rest = torch.nn.Parameter(
            torch.zeros((xyz.shape[0], 15, 3), dtype=torch.float32).cuda()
        )
        model._opacity = torch.nn.Parameter(torch.tensor(opacities, dtype=torch.float32).cuda())
        model._scaling = torch.nn.Parameter(torch.tensor(np.log(scales), dtype=torch.float32).cuda())
        model._rotation = torch.nn.Parameter(torch.tensor(rotations, dtype=torch.float32).cuda())

        return model


"""Rendering functionality

Implements Gaussian Splatting rendering using diff-gaussian-rasterization.
"""

import torch
import numpy as np
from typing import Tuple, Optional, TYPE_CHECKING
try:
    from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
except ImportError:
    print("Warning: diff-gaussian-rasterization not found. Please install it.")
    GaussianRasterizationSettings = None
    GaussianRasterizer = None

if TYPE_CHECKING:
    from .gaussian_model import GaussianModel


def rasterize_gaussians(
    means3D: torch.Tensor,
    means2D: torch.Tensor,
    sh: torch.Tensor,
    colors_precomp: Optional[torch.Tensor],
    opacities: torch.Tensor,
    scales: torch.Tensor,
    rotations: torch.Tensor,
    cov3D_precomp: Optional[torch.Tensor],
    raster_settings: GaussianRasterizationSettings,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Rasterize Gaussians.

    Args:
        means3D: 3D positions [N, 3]
        means2D: 2D positions [N, 2]
        sh: Spherical harmonics coefficients [N, M, 3]
        colors_precomp: Precomputed colors [N, 3] (optional)
        opacities: Opacities [N, 1]
        scales: Scales [N, 3]
        rotations: Rotations (quaternions) [N, 4]
        cov3D_precomp: Precomputed 3D covariance [N, 3, 3] (optional)
        raster_settings: Rasterization settings

    Returns:
        (Rendered image, depth, radii, other information)
    """
    if GaussianRasterizer is None:
        raise ImportError("diff-gaussian-rasterization is not installed")

    return GaussianRasterizer(
        raster_settings=raster_settings
    )(
        means3D=means3D,
        means2D=means2D,
        sh=sh,
        colors_precomp=colors_precomp,
        opacities=opacities,
        scales=scales,
        rotations=rotations,
        cov3D_precomp=cov3D_precomp,
    )


def render(
    viewpoint_camera,
    pc,  # GaussianModel
    pipe,
    bg_color: torch.Tensor,
    scaling_modifier: float = 1.0,
    override_color: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Render Gaussian Splatting.

    Args:
        viewpoint_camera: Camera information (position, rotation, intrinsics, etc.)
        pc: GaussianModel instance
        pipe: Pipeline settings
        bg_color: Background color [3]
        scaling_modifier: Scale modification factor
        override_color: Color override [N, 3] (optional)

    Returns:
        (Rendered image [C, H, W], Depth map [H, W])
    """
    if GaussianRasterizationSettings is None:
        raise ImportError("diff-gaussian-rasterization is not installed")

    # Get camera parameters
    if hasattr(viewpoint_camera, 'world_view_transform'):
        world_view_transform = viewpoint_camera.world_view_transform
    else:
        # Build transformation matrix from camera matrices
        R = viewpoint_camera.R
        t = viewpoint_camera.T
        world_view_transform = torch.eye(4, dtype=torch.float32).cuda()
        world_view_transform[:3, :3] = R.T
        world_view_transform[:3, 3] = -R.T @ t

    if hasattr(viewpoint_camera, 'full_proj_transform'):
        full_proj_transform = viewpoint_camera.full_proj_transform
    else:
        # Build projection matrix
        K = viewpoint_camera.K
        width = viewpoint_camera.image_width
        height = viewpoint_camera.image_height
        znear = 0.01
        zfar = 100.0

        # OpenGL-style projection matrix
        proj = torch.zeros((4, 4), dtype=torch.float32).cuda()
        fx = K[0, 0]
        fy = K[1, 1]
        cx = K[0, 2]
        cy = K[1, 2]

        proj[0, 0] = 2.0 * fx / width
        proj[1, 1] = 2.0 * fy / height
        proj[0, 2] = 1.0 - 2.0 * cx / width
        proj[1, 2] = 2.0 * cy / height - 1.0
        proj[2, 2] = (zfar + znear) / (znear - zfar)
        proj[2, 3] = 2.0 * zfar * znear / (znear - zfar)
        proj[3, 2] = -1.0

        full_proj_transform = (world_view_transform.unsqueeze(0).bmm(proj.unsqueeze(0))).squeeze(0)

    # Get Gaussian parameters
    means3D = pc.get_xyz
    opacity = pc.get_opacity
    scales = pc.get_scaling
    rotations = pc.get_rotation
    shs = pc.get_features

    # Apply scale
    scales = torch.exp(scales) * scaling_modifier

    # Convert opacity with sigmoid
    opacities = torch.sigmoid(opacity)

    # Calculate color
    if override_color is None:
        # Calculate color from spherical harmonics (simplified: DC component only)
        colors = shs[:, 0, :] + 0.28209479177387814
        colors = torch.clamp(colors, 0.0, 1.0)
    else:
        colors = override_color

    # Transform to camera coordinate system
    means3D_cam = (world_view_transform[:3, :3] @ means3D.T + world_view_transform[:3, 3:4]).T

    # 2D projection
    means2D = (full_proj_transform[:3, :3] @ means3D_cam.T + full_proj_transform[:3, 3:4]).T
    means2D = means2D[:, :2] / means2D[:, 2:3]

    # Rasterization settings
    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=float(np.tan(viewpoint_camera.FoVx * 0.5)),
        tanfovy=float(np.tan(viewpoint_camera.FoVy * 0.5)),
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=world_view_transform.cpu().numpy(),
        projmatrix=full_proj_transform.cpu().numpy(),
        sh_degree=0,  # Use DC component only
        campos=means3D_cam.mean(dim=0).cpu().numpy(),
        prefiltered=False,
        debug=False,
    )

    # Rasterize
    rendered_image, radii, rendered_depth, rendered_alpha = rasterize_gaussians(
        means3D=means3D,
        means2D=means2D,
        sh=shs,
        colors_precomp=colors,
        opacities=opacities,
        scales=scales,
        rotations=rotations,
        cov3D_precomp=None,
        raster_settings=raster_settings,
    )

    # Composite background
    rendered_image = rendered_image + bg_color.unsqueeze(-1).unsqueeze(-1) * (1.0 - rendered_alpha)

    return rendered_image, rendered_depth


class Camera:
    """Class to hold camera information"""

    def __init__(
        self,
        R: torch.Tensor,
        T: torch.Tensor,
        K: torch.Tensor,
        image_width: int,
        image_height: int,
        FoVx: float,
        FoVy: float,
        world_view_transform: Optional[torch.Tensor] = None,
        full_proj_transform: Optional[torch.Tensor] = None,
    ):
        """Initialize camera.

        Args:
            R: Rotation matrix [3, 3]
            T: Translation vector [3]
            K: Intrinsic parameter matrix [3, 3]
            image_width: Image width
            image_height: Image height
            FoVx: Horizontal field of view (radians)
            FoVy: Vertical field of view (radians)
            world_view_transform: World to view transformation matrix [4, 4] (optional)
            full_proj_transform: Projection transformation matrix [4, 4] (optional)
        """
        self.R = R
        self.T = T
        self.K = K
        self.image_width = image_width
        self.image_height = image_height
        self.FoVx = FoVx
        self.FoVy = FoVy
        self.world_view_transform = world_view_transform
        self.full_proj_transform = full_proj_transform


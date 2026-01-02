"""Main execution script

Parses command line arguments and integrates the workflow.
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import torch
import cv2
from typing import List

from gaussian_splatting import (
    run_colmap,
    load_colmap_data,
    read_points3D_binary,
    GaussianModel,
    Trainer,
    Camera,
)


def load_images(image_dir: str) -> List[np.ndarray]:
    """Load images.

    Args:
        image_dir: Path to image directory

    Returns:
        List of images
    """
    image_path = Path(image_dir)
    image_files = sorted([
        f for f in image_path.iterdir()
        if f.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']
    ])

    images = []
    for img_file in image_files:
        img = cv2.imread(str(img_file))
        if img is not None:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            images.append(img)
        else:
            print(f"Warning: Could not load {img_file}")

    return images


def create_cameras(
    cam_centers: List[np.ndarray],
    cam_rotations: List[np.ndarray],
    cam_intrinsics: List[np.ndarray],
    image_names: List[str],
    images: List[np.ndarray],
) -> List[Camera]:
    """Create list of camera objects.

    Args:
        cam_centers: List of camera centers
        cam_rotations: List of camera rotation matrices
        cam_intrinsics: List of camera intrinsics
        image_names: List of image names
        images: List of images

    Returns:
        List of camera objects
    """
    cameras = []
    for i, (center, R, K, img) in enumerate(zip(cam_centers, cam_rotations, cam_intrinsics, images)):
        # Transform to camera coordinate system
        R_cam = R.T  # World -> Camera
        T_cam = -R.T @ center.reshape(3, 1)
        T_cam = T_cam.flatten()

        # Calculate field of view
        height, width = img.shape[:2]
        fx = K[0, 0]
        fy = K[1, 1]
        FoVx = 2 * np.arctan(width / (2 * fx))
        FoVy = 2 * np.arctan(height / (2 * fy))

        camera = Camera(
            R=torch.tensor(R_cam, dtype=torch.float32).cuda(),
            T=torch.tensor(T_cam, dtype=torch.float32).cuda(),
            K=torch.tensor(K, dtype=torch.float32).cuda(),
            image_width=width,
            image_height=height,
            FoVx=FoVx,
            FoVy=FoVy,
        )
        cameras.append(camera)

    return cameras


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Gaussian Splatting: Generate 3D model from images")
    parser.add_argument(
        "--input_images",
        type=str,
        required=True,
        help="Path to directory containing input images"
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to output directory"
    )
    parser.add_argument(
        "--colmap_dir",
        type=str,
        default=None,
        help="COLMAP output directory (if already executed)"
    )
    parser.add_argument(
        "--run_colmap",
        action="store_true",
        help="Whether to run COLMAP"
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=30000,
        help="Number of training iterations (default: 30000)"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.01,
        help="Learning rate (default: 0.01)"
    )
    parser.add_argument(
        "--colmap_path",
        type=str,
        default="colmap",
        help="Path to COLMAP executable (default: 'colmap')"
    )

    args = parser.parse_args()

    # Create output directory
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load images
    print("Loading images...")
    images = load_images(args.input_images)
    if len(images) == 0:
        print("Error: No images found in the input directory")
        sys.exit(1)
    print(f"Loaded {len(images)} images")

    # Run or load COLMAP
    if args.run_colmap or args.colmap_dir is None:
        print("Running COLMAP...")
        colmap_output_dir = output_path / "colmap_output"
        success = run_colmap(
            image_dir=args.input_images,
            output_dir=str(colmap_output_dir),
            colmap_path=args.colmap_path,
        )
        if not success:
            print("Error: COLMAP execution failed")
            sys.exit(1)
        colmap_dir = str(colmap_output_dir)
    else:
        colmap_dir = args.colmap_dir

    # Load COLMAP data
    print("Loading COLMAP data...")
    try:
        cam_centers, cam_rotations, cam_intrinsics, image_names, scale = load_colmap_data(
            colmap_dir
        )
    except Exception as e:
        print(f"Error loading COLMAP data: {e}")
        sys.exit(1)

    print(f"Loaded {len(cam_centers)} camera poses")

    # Check correspondence between images and cameras
    if len(images) != len(cam_centers):
        print(f"Warning: Number of images ({len(images)}) doesn't match number of cameras ({len(cam_centers)})")
        # Match to minimum number
        min_len = min(len(images), len(cam_centers))
        images = images[:min_len]
        cam_centers = cam_centers[:min_len]
        cam_rotations = cam_rotations[:min_len]
        cam_intrinsics = cam_intrinsics[:min_len]
        image_names = image_names[:min_len]

    # Create camera objects
    print("Creating camera objects...")
    cameras = create_cameras(
        cam_centers, cam_rotations, cam_intrinsics, image_names, images
    )

    # Load 3D points
    print("Loading 3D points...")
    try:
        sparse_dir = Path(colmap_dir) / "sparse" / "0"
        if not sparse_dir.exists():
            # Try alternative path
            sparse_dirs = [d for d in Path(colmap_dir).iterdir() if d.is_dir() and d.name.startswith("sparse")]
            if sparse_dirs:
                sparse_dir = sparse_dirs[0] / "0"
        
        points3D = read_points3D_binary(str(sparse_dir))
        print(f"Loaded {len(points3D)} 3D points")
    except Exception as e:
        print(f"Warning: Could not load 3D points: {e}")
        print("Initializing from random points...")
        points3D = None

    # Initialize Gaussian model
    print("Initializing Gaussian model...")
    gaussian_model = GaussianModel()
    if points3D is not None:
        gaussian_model.create_from_colmap_points(points3D, spatial_lr_scale=scale * 0.01)
    else:
        # Fallback: initialize from random point cloud
        num_points = 10000
        xyz = np.random.randn(num_points, 3) * scale * 0.1
        rgb = np.random.rand(num_points, 3)
        pcd = np.concatenate([xyz, rgb], axis=1)
        gaussian_model.create_from_pcd(pcd, spatial_lr_scale=scale * 0.01)

    print(f"Initialized {gaussian_model._xyz.shape[0]} Gaussians")

    # Run training
    print("Starting training...")
    trainer = Trainer(
        gaussian_model=gaussian_model,
        cameras=cameras,
        images=images,
        output_dir=str(output_path / "training_output"),
        iterations=args.iterations,
        lr=args.lr,
    )
    trainer.train()

    print("Done!")


if __name__ == "__main__":
    main()

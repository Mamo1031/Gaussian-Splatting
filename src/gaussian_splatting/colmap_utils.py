"""COLMAP integration utilities

Provides functionality to estimate and load camera parameters using COLMAP.
"""

import subprocess
import os
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import json


def run_colmap(
    image_dir: str,
    output_dir: str,
    colmap_path: str = "colmap"
) -> bool:
    """Run COLMAP to estimate camera parameters.

    Args:
        image_dir: Directory containing input images
        output_dir: Directory to save COLMAP output
        colmap_path: Path to COLMAP executable (default: "colmap")

    Returns:
        Whether COLMAP execution was successful
    """
    # Check if COLMAP is available
    import shutil
    if not shutil.which(colmap_path):
        print(f"Error: COLMAP not found at '{colmap_path}'")
        print("Please install COLMAP:")
        print("  Ubuntu/Debian: sudo apt-get install colmap")
        print("  macOS: brew install colmap")
        print("  Or specify the path with --colmap_path option")
        return False

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Set environment variables for headless operation (WSL2/GUI-less environments)
    env = os.environ.copy()
    env["QT_QPA_PLATFORM"] = "offscreen"  # Use offscreen platform for headless operation
    env["DISPLAY"] = ""  # Clear DISPLAY to prevent X11 issues

    # Create database
    database_path = output_path / "database.db"
    cmd = [
        colmap_path, "feature_extractor",
        "--database_path", str(database_path),
        "--image_path", str(image_dir),
        "--ImageReader.camera_model", "PINHOLE",
        "--SiftExtraction.use_gpu", "0",  # Disable GPU for WSL2/headless environments
        "--SiftExtraction.max_num_features", "8192",  # Increase feature points for better matching
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, env=env)
    except PermissionError:
        print(f"Error: Permission denied when executing '{colmap_path}'")
        print("Please check if COLMAP is installed and has execute permissions")
        return False
    except FileNotFoundError:
        print(f"Error: COLMAP executable not found at '{colmap_path}'")
        print("Please install COLMAP or specify the correct path with --colmap_path")
        return False
    
    if result.returncode != 0:
        print(f"Feature extraction failed: {result.stderr}")
        return False

    # Feature matching
    cmd = [
        colmap_path, "exhaustive_matcher",
        "--database_path", str(database_path),
        "--SiftMatching.use_gpu", "0",  # Disable GPU for WSL2/headless environments
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, env=env)
    if result.returncode != 0:
        print(f"Feature matching failed: {result.stderr}")
        return False

    # Sparse reconstruction
    sparse_dir = output_path / "sparse"
    sparse_dir.mkdir(exist_ok=True)
    cmd = [
        colmap_path, "mapper",
        "--database_path", str(database_path),
        "--image_path", str(image_dir),
        "--output_path", str(sparse_dir),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, env=env)
    if result.returncode != 0:
        print(f"Sparse reconstruction failed:")
        print(f"STDERR: {result.stderr}")
        
        # Check for common error patterns
        if "no good initial image pair" in result.stdout.lower() or "no good initial image pair" in result.stderr.lower():
            print("\n" + "="*60)
            print("ERROR: No good initial image pair found")
            print("="*60)
            print("\nPossible causes:")
            print("1. Images are too similar (same viewpoint)")
            print("2. Insufficient overlap between images")
            print("3. Images lack distinctive features (texture, edges)")
            print("4. Too few images (need at least 5-10 with different viewpoints)")
            print("\nRecommendations:")
            print("- Use 10-20 images taken from different angles around the object/scene")
            print("- Ensure images have good overlap (30-60% overlap between adjacent images)")
            print("- Use images with rich texture and distinctive features")
            print("- Avoid images that are too similar or from the same viewpoint")
            print("="*60)
        elif "insufficient" in result.stderr.lower() or "not enough" in result.stderr.lower():
            print("\nNote: COLMAP typically requires at least 5-10 images for successful reconstruction.")
            print("Please provide more images with different viewpoints.")
        else:
            # Show full output for other errors
            print(f"STDOUT: {result.stdout}")
        return False

    return True


def read_cameras_binary(path_to_model_file: str) -> Dict:
    """Read COLMAP cameras.bin file.

    Args:
        path_to_model_file: Path to COLMAP model directory

    Returns:
        Dictionary of camera information
    """
    cameras = {}
    with open(os.path.join(path_to_model_file, "cameras.bin"), "rb") as fid:
        num_cameras = np.fromfile(fid, dtype=np.int64, count=1)[0]
        for _ in range(num_cameras):
            camera_properties = np.fromfile(fid, dtype=np.float64, count=4)
            camera_id = np.fromfile(fid, dtype=np.int32, count=1)[0]
            model_id = np.fromfile(fid, dtype=np.int32, count=1)[0]
            width = np.fromfile(fid, dtype=np.int64, count=1)[0]
            height = np.fromfile(fid, dtype=np.int64, count=1)[0]
            params = np.fromfile(fid, dtype=np.float64, count=-1)
            cameras[camera_id] = {
                "model_id": model_id,
                "width": width,
                "height": height,
                "params": params,
            }
    return cameras


def read_images_binary(path_to_model_file: str) -> Dict:
    """Read COLMAP images.bin file.

    Args:
        path_to_model_file: Path to COLMAP model directory

    Returns:
        Dictionary of image information
    """
    images = {}
    with open(os.path.join(path_to_model_file, "images.bin"), "rb") as fid:
        num_reg_images = np.fromfile(fid, dtype=np.int64, count=1)[0]
        for _ in range(num_reg_images):
            binary_image_properties = np.fromfile(fid, dtype=np.float64, count=7)
            image_id = np.fromfile(fid, dtype=np.int32, count=1)[0]
            qvec = binary_image_properties[0:4]
            tvec = binary_image_properties[4:7]
            camera_id = np.fromfile(fid, dtype=np.int32, count=1)[0]
            image_name = ""
            current_char = np.fromfile(fid, dtype=np.uint8, count=1)[0]
            while current_char != 0:
                image_name += chr(current_char)
                current_char = np.fromfile(fid, dtype=np.uint8, count=1)[0]
            num_points2D = np.fromfile(fid, dtype=np.int64, count=1)[0]
            x_y_id_s = np.fromfile(fid, dtype=np.float64, count=2 * num_points2D)
            xys = np.reshape(x_y_id_s, (num_points2D, 2))
            point3D_ids = np.fromfile(fid, dtype=np.int64, count=num_points2D)
            images[image_id] = {
                "qvec": qvec,
                "tvec": tvec,
                "camera_id": camera_id,
                "name": image_name,
                "xys": xys,
                "point3D_ids": point3D_ids,
            }
    return images


def read_points3D_binary(path_to_model_file: str) -> Dict:
    """Read COLMAP points3D.bin file.

    Args:
        path_to_model_file: Path to COLMAP model directory

    Returns:
        Dictionary of 3D point cloud information
    """
    points3D = {}
    with open(os.path.join(path_to_model_file, "points3D.bin"), "rb") as fid:
        num_points = np.fromfile(fid, dtype=np.int64, count=1)[0]
        for _ in range(num_points):
            binary_point_line_properties = np.fromfile(fid, dtype=np.float64, count=3)
            point3D_id = np.fromfile(fid, dtype=np.int64, count=1)[0]
            xyz = binary_point_line_properties[0:3]
            rgb = np.fromfile(fid, dtype=np.uint8, count=3)
            error = np.fromfile(fid, dtype=np.float64, count=1)[0]
            track_length = np.fromfile(fid, dtype=np.int64, count=2)
            image_ids = np.fromfile(fid, dtype=np.int32, count=track_length[0])
            point2D_idxs = np.fromfile(fid, dtype=np.int32, count=track_length[0])
            points3D[point3D_id] = {
                "xyz": xyz,
                "rgb": rgb,
                "error": error,
            }
    return points3D


def qvec2rotmat(qvec: np.ndarray) -> np.ndarray:
    """Convert quaternion to rotation matrix.

    Args:
        qvec: Quaternion [w, x, y, z]

    Returns:
        3x3 rotation matrix
    """
    return np.array([
        [1 - 2 * qvec[2]**2 - 2 * qvec[3]**2,
         2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
         2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2]],
        [2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
         1 - 2 * qvec[1]**2 - 2 * qvec[3]**2,
         2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1]],
        [2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
         2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
         1 - 2 * qvec[1]**2 - 2 * qvec[2]**2]
    ])


def load_colmap_data(
    colmap_dir: str,
    images_ext: str = ".jpg"
) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray], List[str], float]:
    """Load COLMAP data and return camera parameters.

    Args:
        colmap_dir: Path to COLMAP model directory
        images_ext: Image file extension

    Returns:
        (Camera positions, camera rotations, camera intrinsics, image name list, scale)
    """
    # Find sparse reconstruction directory
    sparse_dirs = [d for d in os.listdir(colmap_dir) if os.path.isdir(os.path.join(colmap_dir, d)) and d.startswith("sparse")]
    if not sparse_dirs:
        # If colmap_dir is directly the model directory
        model_dir = colmap_dir
    else:
        # Use the first sparse directory
        model_dir = os.path.join(colmap_dir, sparse_dirs[0], "0")

    cameras = read_cameras_binary(model_dir)
    images = read_images_binary(model_dir)
    points3D = read_points3D_binary(model_dir)

    # Extract camera parameters
    cam_centers = []
    cam_rotations = []
    cam_intrinsics = []
    image_names = []

    for image_id, image_data in images.items():
        qvec = image_data["qvec"]
        tvec = image_data["tvec"]
        camera_id = image_data["camera_id"]
        image_name = image_data["name"]

        # Calculate rotation matrix and camera center
        R = qvec2rotmat(qvec)
        t = tvec.reshape(3, 1)
        cam_center = -R.T @ t
        cam_center = cam_center.flatten()

        # Camera intrinsics
        camera = cameras[camera_id]
        width = camera["width"]
        height = camera["height"]
        params = camera["params"]
        fx = params[0]
        fy = params[1]
        cx = params[2]
        cy = params[3]

        intrinsics = np.array([
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1]
        ])

        cam_centers.append(cam_center)
        cam_rotations.append(R)
        cam_intrinsics.append(intrinsics)
        image_names.append(image_name)

    # Calculate scale from point cloud range
    if points3D:
        all_points = np.array([p["xyz"] for p in points3D.values()])
        scale = np.max(np.abs(all_points))
    else:
        scale = 1.0

    return cam_centers, cam_rotations, cam_intrinsics, image_names, scale


#!/usr/bin/env python3
"""
Undistort DSLR images for photogrammetry using COLMAP camera/image data.

Reads:
  - cameras.txt: Camera intrinsics and distortion parameters
  - images.txt: Image list with poses and camera associations

Output:
  - Undistorted images in the specified output directory
  - Updated cameras.txt with zero distortion
  - Updated images.txt (copied, as poses don't change)
"""

import argparse
import os
import re
import sys
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Optional

import cv2
import numpy as np


@dataclass
class Camera:
    """Camera model with intrinsics and distortion parameters."""
    camera_id: int
    model: str
    width: int
    height: int
    params: np.ndarray

    def get_intrinsic_matrix(self) -> np.ndarray:
        """Get 3x3 camera intrinsic matrix K."""
        if self.model in ['SIMPLE_PINHOLE', 'SIMPLE_RADIAL', 'RADIAL']:
            f, cx, cy = self.params[0], self.params[1], self.params[2]
            fx, fy = f, f
        elif self.model in ['PINHOLE', 'OPENCV', 'OPENCV_FISHEYE', 'FULL_OPENCV']:
            fx, fy, cx, cy = self.params[0], self.params[1], self.params[2], self.params[3]
        else:
            raise ValueError(f"Unsupported camera model: {self.model}")

        return np.array([
            [fx, 0, cx],
            [0, fy, cy],
            [0,  0,  1]
        ], dtype=np.float64)

    def get_distortion_coeffs(self) -> np.ndarray:
        """Get distortion coefficients for OpenCV undistort."""
        if self.model == 'SIMPLE_PINHOLE':
            return np.zeros(4)
        elif self.model == 'PINHOLE':
            return np.zeros(4)
        elif self.model == 'SIMPLE_RADIAL':
            # k1 only
            k1 = self.params[3]
            return np.array([k1, 0, 0, 0])
        elif self.model == 'RADIAL':
            # k1, k2
            k1, k2 = self.params[3], self.params[4]
            return np.array([k1, k2, 0, 0])
        elif self.model == 'OPENCV':
            # k1, k2, p1, p2
            k1, k2, p1, p2 = self.params[4], self.params[5], self.params[6], self.params[7]
            return np.array([k1, k2, p1, p2])
        elif self.model == 'FULL_OPENCV':
            # k1, k2, p1, p2, k3, k4, k5, k6
            return self.params[4:12]
        elif self.model == 'OPENCV_FISHEYE':
            # k1, k2, k3, k4 for fisheye
            return self.params[4:8]
        else:
            raise ValueError(f"Unsupported camera model: {self.model}")

    def is_fisheye(self) -> bool:
        return self.model == 'OPENCV_FISHEYE'

    def to_undistorted_line(self) -> str:
        """Return camera line with zero distortion for output."""
        K = self.get_intrinsic_matrix()
        fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]

        if self.model in ['SIMPLE_PINHOLE', 'SIMPLE_RADIAL', 'RADIAL']:
            # Convert to SIMPLE_PINHOLE (no distortion)
            return f"{self.camera_id} SIMPLE_PINHOLE {self.width} {self.height} {fx} {cx} {cy}"
        else:
            # Convert to PINHOLE (no distortion)
            return f"{self.camera_id} PINHOLE {self.width} {self.height} {fx} {fy} {cx} {cy}"


@dataclass
class Image:
    """Image with pose and camera reference."""
    image_id: int
    qw: float
    qx: float
    qy: float
    qz: float
    tx: float
    ty: float
    tz: float
    camera_id: int
    name: str
    points2d_line: str  # Keep the second line as-is


def parse_cameras(cameras_path: Path) -> Dict[int, Camera]:
    """Parse COLMAP cameras.txt file."""
    cameras = {}

    with open(cameras_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue

            parts = line.split()
            camera_id = int(parts[0])
            model = parts[1]
            width = int(parts[2])
            height = int(parts[3])
            params = np.array([float(p) for p in parts[4:]])

            cameras[camera_id] = Camera(
                camera_id=camera_id,
                model=model,
                width=width,
                height=height,
                params=params
            )

    return cameras


def parse_images(images_path: Path) -> Dict[str, Image]:
    """Parse COLMAP images.txt file."""
    images = {}

    with open(images_path, 'r') as f:
        lines = f.readlines()

    i = 0
    while i < len(lines):
        line = lines[i].strip()

        # Skip comments and empty lines
        if not line or line.startswith('#'):
            i += 1
            continue

        # Parse image line
        parts = line.split()
        if len(parts) < 10:
            i += 1
            continue

        image_id = int(parts[0])
        qw, qx, qy, qz = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
        tx, ty, tz = float(parts[5]), float(parts[6]), float(parts[7])
        camera_id = int(parts[8])
        name = parts[9]

        # Next line contains 2D points
        points2d_line = ""
        if i + 1 < len(lines):
            points2d_line = lines[i + 1].strip()
            i += 1

        images[name] = Image(
            image_id=image_id,
            qw=qw, qx=qx, qy=qy, qz=qz,
            tx=tx, ty=ty, tz=tz,
            camera_id=camera_id,
            name=name,
            points2d_line=points2d_line
        )

        i += 1

    return images


def undistort_image(
    img: np.ndarray,
    camera: Camera,
    alpha: float = 0
) -> np.ndarray:
    """
    Undistort an image using camera parameters.

    Args:
        img: Input image
        camera: Camera with intrinsics and distortion
        alpha: Free scaling parameter (0=no black borders, 1=keep all pixels)

    Returns:
        Undistorted image
    """
    K = camera.get_intrinsic_matrix()
    dist_coeffs = camera.get_distortion_coeffs()

    h, w = img.shape[:2]

    if camera.is_fisheye():
        # Fisheye undistortion
        new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
            K, dist_coeffs, (w, h), np.eye(3), balance=alpha
        )
        map1, map2 = cv2.fisheye.initUndistortRectifyMap(
            K, dist_coeffs, np.eye(3), new_K, (w, h), cv2.CV_16SC2
        )
    else:
        # Standard lens undistortion
        new_K, roi = cv2.getOptimalNewCameraMatrix(
            K, dist_coeffs, (w, h), alpha, (w, h)
        )
        map1, map2 = cv2.initUndistortRectifyMap(
            K, dist_coeffs, None, new_K, (w, h), cv2.CV_16SC2
        )

    undistorted = cv2.remap(img, map1, map2, cv2.INTER_LANCZOS4)

    return undistorted


def undistort_points2d(
    points2d_line: str,
    camera: Camera
) -> str:
    """
    Undistort 2D point observations.

    Args:
        points2d_line: Line containing X, Y, POINT3D_ID triplets
        camera: Camera parameters

    Returns:
        Updated points2d line with undistorted coordinates
    """
    if not points2d_line.strip():
        return points2d_line

    parts = points2d_line.split()
    if len(parts) < 3:
        return points2d_line

    K = camera.get_intrinsic_matrix()
    dist_coeffs = camera.get_distortion_coeffs()

    # Parse points
    points = []
    point3d_ids = []
    for i in range(0, len(parts), 3):
        if i + 2 < len(parts):
            x, y = float(parts[i]), float(parts[i + 1])
            point3d_id = parts[i + 2]
            points.append([x, y])
            point3d_ids.append(point3d_id)

    if not points:
        return points2d_line

    # Undistort points
    points_array = np.array(points, dtype=np.float64).reshape(-1, 1, 2)

    if camera.is_fisheye():
        undistorted_points = cv2.fisheye.undistortPoints(
            points_array, K, dist_coeffs, P=K
        )
    else:
        undistorted_points = cv2.undistortPoints(
            points_array, K, dist_coeffs, P=K
        )

    undistorted_points = undistorted_points.reshape(-1, 2)

    # Rebuild line
    new_parts = []
    for i, (pt, pt3d_id) in enumerate(zip(undistorted_points, point3d_ids)):
        new_parts.extend([f"{pt[0]:.6f}", f"{pt[1]:.6f}", pt3d_id])

    return " ".join(new_parts)


def main():
    parser = argparse.ArgumentParser(
        description="Undistort DSLR images for photogrammetry using COLMAP data"
    )
    parser.add_argument(
        "--input-dir", "-i",
        type=Path,
        required=True,
        help="Directory containing images.txt and cameras.txt"
    )
    parser.add_argument(
        "--images-dir",
        type=Path,
        default=None,
        help="Directory containing source images (default: input-dir/images)"
    )
    parser.add_argument(
        "--output-dir", "-o",
        type=Path,
        required=True,
        help="Output directory for undistorted images and updated files"
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0,
        help="Scaling parameter: 0=crop black borders, 1=keep all pixels (default: 0)"
    )
    parser.add_argument(
        "--update-points",
        action="store_true",
        help="Also undistort 2D point observations in images.txt"
    )
    parser.add_argument(
        "--format",
        choices=["png", "jpg", "tiff"],
        default="png",
        help="Output image format (default: png)"
    )
    parser.add_argument(
        "--quality",
        type=int,
        default=95,
        help="JPEG quality if using jpg format (default: 95)"
    )

    args = parser.parse_args()

    # Setup paths
    cameras_path = args.input_dir / "cameras.txt"
    images_txt_path = args.input_dir / "images.txt"
    images_dir = args.images_dir or (args.input_dir / "images")

    # Validate inputs
    if not cameras_path.exists():
        print(f"Error: cameras.txt not found at {cameras_path}")
        sys.exit(1)
    if not images_txt_path.exists():
        print(f"Error: images.txt not found at {images_txt_path}")
        sys.exit(1)
    if not images_dir.exists():
        print(f"Error: Images directory not found at {images_dir}")
        sys.exit(1)

    # Create output directories
    output_images_dir = args.output_dir / "images"
    output_images_dir.mkdir(parents=True, exist_ok=True)

    # Parse COLMAP files
    print(f"Reading cameras from {cameras_path}")
    cameras = parse_cameras(cameras_path)
    print(f"  Found {len(cameras)} camera(s)")

    for cam_id, cam in cameras.items():
        print(f"    Camera {cam_id}: {cam.model} ({cam.width}x{cam.height})")

    print(f"\nReading images from {images_txt_path}")
    images = parse_images(images_txt_path)
    print(f"  Found {len(images)} image(s)")

    # Process images
    print(f"\nUndistorting images...")

    processed = 0
    failed = 0

    for name, img_data in images.items():
        # Find source image
        src_path = images_dir / name
        if not src_path.exists():
            # Try common extensions
            for ext in ['.jpg', '.jpeg', '.png', '.tiff', '.tif', '.JPG', '.JPEG', '.PNG']:
                alt_path = images_dir / (Path(name).stem + ext)
                if alt_path.exists():
                    src_path = alt_path
                    break

        if not src_path.exists():
            print(f"  Warning: Image not found: {name}")
            failed += 1
            continue

        # Get camera
        camera = cameras.get(img_data.camera_id)
        if camera is None:
            print(f"  Warning: Camera {img_data.camera_id} not found for image {name}")
            failed += 1
            continue

        # Read and undistort
        img = cv2.imread(str(src_path), cv2.IMREAD_UNCHANGED)
        if img is None:
            print(f"  Warning: Failed to read image: {src_path}")
            failed += 1
            continue

        undistorted = undistort_image(img, camera, args.alpha)

        # Save undistorted image
        out_name = Path(name).stem + f".{args.format}"
        out_path = output_images_dir / out_name

        if args.format == "jpg":
            cv2.imwrite(str(out_path), undistorted, [cv2.IMWRITE_JPEG_QUALITY, args.quality])
        elif args.format == "png":
            cv2.imwrite(str(out_path), undistorted, [cv2.IMWRITE_PNG_COMPRESSION, 3])
        else:
            cv2.imwrite(str(out_path), undistorted)

        processed += 1
        print(f"  [{processed}/{len(images)}] {name} -> {out_name}")

        # Update image name in data for output
        img_data.name = out_name

        # Undistort 2D points if requested
        if args.update_points:
            img_data.points2d_line = undistort_points2d(img_data.points2d_line, camera)

    # Write updated cameras.txt (with zero distortion)
    cameras_out_path = args.output_dir / "cameras.txt"
    print(f"\nWriting undistorted cameras to {cameras_out_path}")
    with open(cameras_out_path, 'w') as f:
        f.write("# Camera list with one line of data per camera:\n")
        f.write("#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
        f.write(f"# Number of cameras: {len(cameras)}\n")
        for camera in cameras.values():
            f.write(camera.to_undistorted_line() + "\n")

    # Write updated images.txt
    images_out_path = args.output_dir / "images.txt"
    print(f"Writing updated images to {images_out_path}")
    with open(images_out_path, 'w') as f:
        f.write("# Image list with two lines of data per image:\n")
        f.write("#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
        f.write("#   POINTS2D[] as (X, Y, POINT3D_ID)\n")
        f.write(f"# Number of images: {len(images)}\n")
        for img_data in images.values():
            f.write(f"{img_data.image_id} {img_data.qw} {img_data.qx} {img_data.qy} {img_data.qz} ")
            f.write(f"{img_data.tx} {img_data.ty} {img_data.tz} {img_data.camera_id} {img_data.name}\n")
            f.write(f"{img_data.points2d_line}\n")

    # Summary
    print(f"\nDone!")
    print(f"  Processed: {processed} images")
    print(f"  Failed: {failed} images")
    print(f"  Output directory: {args.output_dir}")


if __name__ == "__main__":
    main()

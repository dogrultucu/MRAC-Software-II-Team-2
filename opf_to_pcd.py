"""
OPF/glTF to Point Cloud Converter
=================================
Converts Pix4D/OpenPipeline (OPF) glTF point cloud format to standard PLY/PCD.

This script reads the binary glbin files from Pix4D output and converts
them to Open3D compatible formats.

Usage:
    python opf_to_pcd.py --input "./olympic_flame/olympic_flame" --output "./output"
    python opf_to_pcd.py --input "./olympic_flame/olympic_flame" --visualize

Requirements:
    pip install open3d numpy
"""

import argparse
import json
import struct
import sys
from pathlib import Path
from datetime import datetime

import numpy as np

try:
    import open3d as o3d
except ImportError:
    print("Error: Open3D required. Install with: pip install open3d")
    sys.exit(1)


def read_glbin_float32(filepath: Path, count: int, components: int = 3) -> np.ndarray:
    """Read float32 binary data from glbin file"""
    with open(filepath, 'rb') as f:
        data = f.read()

    # float32 = 4 bytes per value
    expected_size = count * components * 4
    if len(data) < expected_size:
        print(f"  Warning: File size mismatch. Expected {expected_size}, got {len(data)}")

    # Unpack as float32
    values = np.frombuffer(data[:expected_size], dtype=np.float32)
    return values.reshape(count, components)


def read_glbin_uint8(filepath: Path, count: int, components: int = 4) -> np.ndarray:
    """Read uint8 binary data from glbin file (for colors)"""
    with open(filepath, 'rb') as f:
        data = f.read()

    expected_size = count * components
    if len(data) < expected_size:
        print(f"  Warning: File size mismatch. Expected {expected_size}, got {len(data)}")

    values = np.frombuffer(data[:expected_size], dtype=np.uint8)
    return values.reshape(count, components)


def load_opf_point_cloud(project_dir: Path) -> o3d.geometry.PointCloud:
    """Load point cloud from OPF/Pix4D project directory"""

    dense_dir = project_dir / "dense"
    gltf_path = dense_dir / "pcl.gltf"

    if not gltf_path.exists():
        print(f"Error: pcl.gltf not found in {dense_dir}")
        return None

    print(f"\nLoading OPF point cloud from: {project_dir}")

    # Parse glTF to get metadata
    with open(gltf_path, 'r') as f:
        gltf = json.load(f)

    # Get point count from accessors
    accessors = gltf.get('accessors', [])
    if not accessors:
        print("Error: No accessors found in glTF")
        return None

    # Position accessor (index 0)
    pos_accessor = accessors[0]
    point_count = pos_accessor['count']

    print(f"  Point count: {point_count:,}")

    # Get transformation matrix (Y-up to Z-up conversion)
    transform = np.eye(4)
    nodes = gltf.get('nodes', [])
    if nodes and 'matrix' in nodes[0]:
        matrix = np.array(nodes[0]['matrix']).reshape(4, 4).T  # Column-major to row-major
        transform = matrix

    # Load positions
    positions_path = dense_dir / "positions.glbin"
    if not positions_path.exists():
        print(f"Error: positions.glbin not found")
        return None

    print(f"  Loading positions from: positions.glbin")
    positions = read_glbin_float32(positions_path, point_count, 3)
    print(f"    Shape: {positions.shape}")
    print(f"    Range X: [{positions[:, 0].min():.3f}, {positions[:, 0].max():.3f}]")
    print(f"    Range Y: [{positions[:, 1].min():.3f}, {positions[:, 1].max():.3f}]")
    print(f"    Range Z: [{positions[:, 2].min():.3f}, {positions[:, 2].max():.3f}]")

    # Apply transformation (glTF Y-up to Z-up)
    positions_h = np.hstack([positions, np.ones((point_count, 1))])
    positions_transformed = (transform @ positions_h.T).T[:, :3]

    # Load normals
    normals = None
    normals_path = dense_dir / "normals.glbin"
    if normals_path.exists():
        print(f"  Loading normals from: normals.glbin")
        normals = read_glbin_float32(normals_path, point_count, 3)
        # Transform normals (rotation only)
        normals_transformed = (transform[:3, :3] @ normals.T).T
        # Normalize
        norms = np.linalg.norm(normals_transformed, axis=1, keepdims=True)
        norms[norms == 0] = 1
        normals = normals_transformed / norms
        print(f"    Shape: {normals.shape}")

    # Load colors
    colors = None
    colors_path = dense_dir / "colors.glbin"
    if colors_path.exists():
        print(f"  Loading colors from: colors.glbin")
        colors_raw = read_glbin_uint8(colors_path, point_count, 4)
        # Convert from RGBA uint8 to RGB float [0,1]
        colors = colors_raw[:, :3].astype(np.float64) / 255.0
        print(f"    Shape: {colors.shape}")

    # Create Open3D point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(positions_transformed.astype(np.float64))

    if colors is not None:
        pcd.colors = o3d.utility.Vector3dVector(colors)

    if normals is not None:
        pcd.normals = o3d.utility.Vector3dVector(normals.astype(np.float64))

    print(f"\n  Successfully loaded point cloud:")
    print(f"    Points: {len(pcd.points):,}")
    print(f"    Has colors: {pcd.has_colors()}")
    print(f"    Has normals: {pcd.has_normals()}")

    return pcd


def load_calibrated_cameras(project_dir: Path) -> list:
    """Load camera poses from calibrated_cameras.json"""
    cameras_path = project_dir / "calibrated_cameras.json"

    if not cameras_path.exists():
        return []

    with open(cameras_path, 'r') as f:
        data = json.load(f)

    cameras = data.get('cameras', [])
    print(f"  Found {len(cameras)} calibrated cameras")

    return cameras


def create_camera_frustums(cameras: list, scale: float = 0.1) -> list:
    """Create camera frustum visualizations"""
    frustums = []

    for cam in cameras:
        pos = np.array(cam['position'])

        # Create a small coordinate frame at camera position
        frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=scale)
        frame.translate(pos)

        # Simple pyramid frustum
        # We'd need full rotation matrix for proper orientation
        # For now, just show camera positions
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=scale * 0.5)
        sphere.translate(pos)
        sphere.paint_uniform_color([1.0, 0.5, 0.0])  # Orange

        frustums.append(sphere)

    return frustums


def save_point_cloud(pcd: o3d.geometry.PointCloud, output_dir: Path, name: str = "olympic_flame"):
    """Save point cloud to various formats"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    print(f"\n" + "=" * 60)
    print("SAVING POINT CLOUD")
    print("=" * 60)

    # PLY format
    ply_path = output_dir / f"{name}_{timestamp}.ply"
    o3d.io.write_point_cloud(str(ply_path), pcd)
    print(f"  Saved: {ply_path}")

    # PCD format
    pcd_path = output_dir / f"{name}_{timestamp}.pcd"
    o3d.io.write_point_cloud(str(pcd_path), pcd)
    print(f"  Saved: {pcd_path}")

    # XYZ with colors
    xyz_path = output_dir / f"{name}_{timestamp}.xyz"
    points = np.asarray(pcd.points)
    if pcd.has_colors():
        colors = (np.asarray(pcd.colors) * 255).astype(int)
        data = np.hstack([points, colors])
        np.savetxt(str(xyz_path), data, fmt='%.6f %.6f %.6f %d %d %d')
    else:
        np.savetxt(str(xyz_path), points, fmt='%.6f')
    print(f"  Saved: {xyz_path}")

    # LAS format (if laspy available)
    try:
        import laspy
        las_path = output_dir / f"{name}_{timestamp}.las"

        # Create LAS file
        las = laspy.create(file_version="1.4", point_format=7)
        las.x = points[:, 0]
        las.y = points[:, 1]
        las.z = points[:, 2]

        if pcd.has_colors():
            colors_16bit = (np.asarray(pcd.colors) * 65535).astype(np.uint16)
            las.red = colors_16bit[:, 0]
            las.green = colors_16bit[:, 1]
            las.blue = colors_16bit[:, 2]

        las.write(str(las_path))
        print(f"  Saved: {las_path}")
    except ImportError:
        pass  # laspy not available

    return ply_path, pcd_path


def visualize(pcd: o3d.geometry.PointCloud, cameras: list = None, show_cameras: bool = True):
    """Visualize the point cloud"""
    print("\n" + "=" * 60)
    print("VISUALIZATION")
    print("=" * 60)
    print("Controls:")
    print("  Mouse drag: Rotate")
    print("  Scroll: Zoom")
    print("  +/-: Point size")
    print("  Q: Quit")
    print("=" * 60)

    geometries = [pcd]

    # Add coordinate frame
    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5)
    geometries.append(coord_frame)

    # Add camera positions if available
    if show_cameras and cameras:
        print(f"  Showing {len(cameras)} camera positions (orange spheres)")
        frustums = create_camera_frustums(cameras, scale=0.05)
        geometries.extend(frustums)

    o3d.visualization.draw_geometries(
        geometries,
        window_name="Olympic Flame Point Cloud",
        width=1280,
        height=720,
        point_show_normal=False
    )


def print_project_info(project_dir: Path):
    """Print information about the OPF project"""
    print("\n" + "=" * 60)
    print("OPF PROJECT INFORMATION")
    print("=" * 60)

    # Check project.opf
    opf_path = project_dir / "project.opf"
    if opf_path.exists():
        print(f"  Project file: {opf_path.name}")

    # Count images
    images_dir = project_dir / "images"
    if images_dir.exists():
        images = list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.JPG"))
        print(f"  Source images: {len(images)}")

    # Check dense folder
    dense_dir = project_dir / "dense"
    if dense_dir.exists():
        gltf_path = dense_dir / "pcl.gltf"
        if gltf_path.exists():
            with open(gltf_path, 'r') as f:
                gltf = json.load(f)

            # Get point count
            accessors = gltf.get('accessors', [])
            if accessors:
                point_count = accessors[0].get('count', 0)
                print(f"  Dense point cloud: {point_count:,} points")

            # Get bounding box
            if accessors:
                bbox_min = accessors[0].get('min', [])
                bbox_max = accessors[0].get('max', [])
                if bbox_min and bbox_max:
                    print(f"  Bounding box:")
                    print(f"    Min: [{bbox_min[0]:.2f}, {bbox_min[1]:.2f}, {bbox_min[2]:.2f}]")
                    print(f"    Max: [{bbox_max[0]:.2f}, {bbox_max[1]:.2f}, {bbox_max[2]:.2f}]")

            # Generator
            asset = gltf.get('asset', {})
            generator = asset.get('generator', 'Unknown')
            print(f"  Generated by: {generator}")

    # Check sparse folder
    sparse_dir = project_dir / "sparse"
    if sparse_dir.exists():
        print(f"  Sparse reconstruction: Present")

    # Calibration
    calib_path = project_dir / "calibrated_cameras.json"
    if calib_path.exists():
        with open(calib_path, 'r') as f:
            calib = json.load(f)
        cam_count = len(calib.get('cameras', []))
        print(f"  Calibrated cameras: {cam_count}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert OPF/Pix4D point cloud to PLY/PCD",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python opf_to_pcd.py --input "./olympic_flame/olympic_flame"
  python opf_to_pcd.py --input "./olympic_flame/olympic_flame" --output "./output" --visualize
  python opf_to_pcd.py --input "./olympic_flame/olympic_flame" --info
        """
    )

    parser.add_argument("--input", "-i", type=str, required=True,
                       help="Path to OPF project directory (containing dense/, images/, etc.)")
    parser.add_argument("--output", "-o", type=str, default="./output_opf",
                       help="Output directory (default: ./output_opf)")
    parser.add_argument("--name", "-n", type=str, default="olympic_flame",
                       help="Output file name prefix (default: olympic_flame)")
    parser.add_argument("--visualize", "-v", action="store_true",
                       help="Visualize the point cloud")
    parser.add_argument("--show-cameras", action="store_true",
                       help="Show camera positions in visualization")
    parser.add_argument("--info", action="store_true",
                       help="Print project information only")
    parser.add_argument("--downsample", "-d", type=float, default=0,
                       help="Voxel downsample size (0 = no downsampling)")

    args = parser.parse_args()

    project_dir = Path(args.input)
    if not project_dir.exists():
        print(f"Error: Project directory not found: {project_dir}")
        sys.exit(1)

    # Print project info
    print_project_info(project_dir)

    if args.info:
        sys.exit(0)

    # Load point cloud
    pcd = load_opf_point_cloud(project_dir)
    if pcd is None:
        sys.exit(1)

    # Load cameras
    cameras = load_calibrated_cameras(project_dir)

    # Downsample if requested
    if args.downsample > 0:
        print(f"\nDownsampling with voxel size: {args.downsample}")
        original_count = len(pcd.points)
        pcd = pcd.voxel_down_sample(args.downsample)
        print(f"  {original_count:,} -> {len(pcd.points):,} points")

    # Save
    save_point_cloud(pcd, args.output, args.name)

    # Summary
    print("\n" + "=" * 60)
    print("CONVERSION COMPLETE")
    print("=" * 60)
    print(f"  Points: {len(pcd.points):,}")
    print(f"  Output: {args.output}")

    # Visualize if requested
    if args.visualize:
        visualize(pcd, cameras, args.show_cameras)


if __name__ == "__main__":
    main()

"""
PCD/PLY Point Cloud Viewer with Heatmap Support
================================================
Standalone viewer for point cloud files with crack detection heatmap visualization.

Usage:
    # View a point cloud
    python pcd_viewer.py --pcd ./pointcloud.pcd

    # View with crack detection JSON (generates heatmap from crack locations)
    python pcd_viewer.py --pcd ./pointcloud.pcd --cracks ./crack_detection_report.json

    # View pre-generated heatmap file
    python pcd_viewer.py --pcd ./annotated_cracks_heatmap_*.ply

    # Adjust heatmap distance
    python pcd_viewer.py --pcd ./pointcloud.pcd --cracks ./cracks.json --heatmap-distance 1.0

Controls:
    Mouse drag: Rotate view
    Scroll: Zoom
    H: Toggle heatmap view (Blue=far, Red=on crack)
    R: Reset to original colors
    +/-: Increase/decrease point size
    Q: Quit
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

try:
    import open3d as o3d
except ImportError:
    print("Error: Open3D is required. Install with: pip install open3d")
    sys.exit(1)


def load_point_cloud(filepath: str):
    """Load point cloud from PCD, PLY, or other supported formats"""
    filepath = Path(filepath)
    if not filepath.exists():
        print(f"Error: File not found: {filepath}")
        return None

    print(f"Loading point cloud: {filepath}")
    pcd = o3d.io.read_point_cloud(str(filepath))

    if len(pcd.points) == 0:
        print("Error: Point cloud is empty")
        return None

    print(f"  Points: {len(pcd.points):,}")
    print(f"  Has colors: {pcd.has_colors()}")
    print(f"  Has normals: {pcd.has_normals()}")

    return pcd


def load_crack_points_from_json(json_path: str) -> np.ndarray:
    """Load crack points from detection report JSON"""
    json_path = Path(json_path)
    if not json_path.exists():
        print(f"Error: JSON file not found: {json_path}")
        return None

    print(f"Loading crack data: {json_path}")

    with open(json_path, 'r') as f:
        data = json.load(f)

    crack_points = []

    # Try different JSON structures
    if "cracks" in data:
        for crack in data["cracks"]:
            if "points_3d" in crack:
                crack_points.extend(crack["points_3d"])
            elif "centroid" in crack:
                crack_points.append(crack["centroid"])
    elif "detections" in data:
        for det in data["detections"]:
            if "points_3d" in det:
                crack_points.extend(det["points_3d"])
            elif "centroid" in det:
                crack_points.append(det["centroid"])

    if len(crack_points) == 0:
        print("  Warning: No crack points found in JSON")
        return None

    crack_points = np.array(crack_points)
    print(f"  Loaded {len(crack_points):,} crack points")
    return crack_points


def generate_heatmap_colors(pcd, crack_points: np.ndarray, max_distance: float = 0.5) -> np.ndarray:
    """Generate heatmap colors based on distance to crack points"""
    from scipy.spatial import cKDTree

    print(f"Generating heatmap (max distance: {max_distance}m)...")

    crack_tree = cKDTree(crack_points)
    points = np.asarray(pcd.points)
    distances, _ = crack_tree.query(points, k=1)

    # Normalize distances
    normalized = np.clip(distances / max_distance, 0, 1)

    # Generate colors: Red (on crack) -> Orange -> Yellow -> Green -> Cyan -> Blue (far)
    colors = np.zeros((len(normalized), 3))

    for i, d in enumerate(normalized):
        if d <= 0.0:
            # On crack - red
            colors[i] = [1.0, 0.0, 0.0]
        elif d <= 0.15:
            # Red to orange
            t = d / 0.15
            colors[i] = [1.0, 0.5 * t, 0.0]
        elif d <= 0.3:
            # Orange to yellow
            t = (d - 0.15) / 0.15
            colors[i] = [1.0, 0.5 + 0.5 * t, 0.0]
        elif d <= 0.5:
            # Yellow to green
            t = (d - 0.3) / 0.2
            colors[i] = [1.0 - t, 1.0, 0.0]
        elif d <= 0.7:
            # Green to cyan
            t = (d - 0.5) / 0.2
            colors[i] = [0.0, 1.0, t]
        elif d <= 0.85:
            # Cyan to light blue
            t = (d - 0.7) / 0.15
            colors[i] = [0.0, 1.0 - 0.5 * t, 1.0]
        else:
            # Light blue to blue
            t = (d - 0.85) / 0.15
            colors[i] = [0.0, 0.5 - 0.5 * t, 1.0]

    print("  Heatmap generated")
    return colors


def view_point_cloud(pcd, heatmap_colors=None, title="Point Cloud Viewer"):
    """Interactive point cloud viewer with heatmap toggle"""

    print("\n" + "=" * 60)
    print("POINT CLOUD VIEWER")
    print("=" * 60)
    print("Controls:")
    print("  Mouse drag: Rotate view")
    print("  Scroll: Zoom")
    if heatmap_colors is not None:
        print("  H: Toggle heatmap (Blue=far, Red=on crack)")
    print("  R: Reset to original colors")
    print("  +/-: Increase/decrease point size")
    print("  Q: Quit")
    print("=" * 60)

    # Store original colors
    original_colors = np.asarray(pcd.colors).copy() if pcd.has_colors() else None

    # State
    show_heatmap = [False]
    point_size = [2.0]

    def toggle_heatmap(vis):
        if heatmap_colors is None:
            print("No heatmap data available")
            return False

        show_heatmap[0] = not show_heatmap[0]
        if show_heatmap[0]:
            pcd.colors = o3d.utility.Vector3dVector(heatmap_colors)
            print("Heatmap: ON (Blue=far from crack, Red=on crack)")
        elif original_colors is not None:
            pcd.colors = o3d.utility.Vector3dVector(original_colors)
            print("Heatmap: OFF (original colors)")
        else:
            # No original colors, set to white
            pcd.colors = o3d.utility.Vector3dVector(
                np.ones((len(pcd.points), 3)) * 0.7
            )
            print("Heatmap: OFF (no original colors)")
        vis.update_geometry(pcd)
        return False

    def reset_colors(vis):
        show_heatmap[0] = False
        if original_colors is not None:
            pcd.colors = o3d.utility.Vector3dVector(original_colors)
        else:
            pcd.colors = o3d.utility.Vector3dVector(
                np.ones((len(pcd.points), 3)) * 0.7
            )
        print("Reset to original colors")
        vis.update_geometry(pcd)
        return False

    def increase_point_size(vis):
        point_size[0] = min(point_size[0] + 0.5, 10.0)
        vis.get_render_option().point_size = point_size[0]
        print(f"Point size: {point_size[0]}")
        return False

    def decrease_point_size(vis):
        point_size[0] = max(point_size[0] - 0.5, 1.0)
        vis.get_render_option().point_size = point_size[0]
        print(f"Point size: {point_size[0]}")
        return False

    # Create visualizer
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(window_name=title, width=1280, height=720)

    # Add geometry
    vis.add_geometry(pcd)

    # Add coordinate frame
    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3)
    vis.add_geometry(coord_frame)

    # Register key callbacks
    vis.register_key_callback(ord('H'), toggle_heatmap)
    vis.register_key_callback(ord('R'), reset_colors)
    vis.register_key_callback(ord('+'), increase_point_size)
    vis.register_key_callback(ord('='), increase_point_size)  # For keyboards without numpad
    vis.register_key_callback(ord('-'), decrease_point_size)

    # Set render options
    opt = vis.get_render_option()
    opt.point_size = point_size[0]
    opt.background_color = np.array([0.1, 0.1, 0.1])
    opt.show_coordinate_frame = True

    # Run
    vis.run()
    vis.destroy_window()


def main():
    parser = argparse.ArgumentParser(
        description="Point Cloud Viewer with Heatmap Support",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # View a point cloud
  python pcd_viewer.py --pcd ./pointcloud.pcd

  # View with heatmap from crack detection JSON
  python pcd_viewer.py --pcd ./pointcloud.pcd --cracks ./crack_report.json

  # View pre-generated heatmap PLY
  python pcd_viewer.py --pcd ./annotated_cracks_heatmap_20240101_120000.ply

  # Adjust heatmap distance (default 0.5m)
  python pcd_viewer.py --pcd ./scan.pcd --cracks ./cracks.json --heatmap-distance 1.0
        """
    )

    parser.add_argument("--pcd", "-p", type=str, required=True,
                       help="Path to point cloud file (PCD, PLY, XYZ, etc.)")
    parser.add_argument("--cracks", "-c", type=str,
                       help="Path to crack detection JSON report (optional)")
    parser.add_argument("--heatmap-distance", "-d", type=float, default=0.5,
                       help="Max distance for heatmap gradient in meters (default: 0.5)")
    parser.add_argument("--start-heatmap", "-s", action="store_true",
                       help="Start with heatmap view enabled")

    args = parser.parse_args()

    # Load point cloud
    pcd = load_point_cloud(args.pcd)
    if pcd is None:
        sys.exit(1)

    # Generate heatmap if crack data provided
    heatmap_colors = None
    if args.cracks:
        crack_points = load_crack_points_from_json(args.cracks)
        if crack_points is not None and len(crack_points) > 0:
            try:
                heatmap_colors = generate_heatmap_colors(
                    pcd, crack_points, args.heatmap_distance
                )
            except ImportError:
                print("Warning: scipy not found, heatmap generation requires scipy")
                print("Install with: pip install scipy")

    # Check if file is already a heatmap file
    filepath = Path(args.pcd)
    if "heatmap" in filepath.stem.lower():
        print("Note: This appears to be a pre-generated heatmap file")
        print("      The colors shown are the crack proximity heatmap")

    # If starting with heatmap and we have colors, apply them
    if args.start_heatmap and heatmap_colors is not None:
        pcd.colors = o3d.utility.Vector3dVector(heatmap_colors)
        print("Starting with heatmap view")

    # View
    title = f"PCD Viewer - {filepath.name}"
    view_point_cloud(pcd, heatmap_colors, title)

    print("\nViewer closed")


if __name__ == "__main__":
    main()

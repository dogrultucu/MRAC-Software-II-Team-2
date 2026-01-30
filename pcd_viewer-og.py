"""
Point Cloud Viewer & Simulator using Open3D
============================================
Opens and visualizes PCD, PLY, and other point cloud formats.

Usage:
    python pcd_viewer.py                     # Opens file dialog
    python pcd_viewer.py path/to/file.pcd    # Opens specific file
    python pcd_viewer.py --demo              # Shows demo point cloud

Controls:
    Mouse drag     - Rotate view
    Scroll         - Zoom in/out
    Shift + drag   - Pan
    R              - Reset view
    +/-            - Increase/decrease point size
    C              - Toggle coordinate frame
    N              - Toggle normals (if available)
    B              - Toggle bounding box
    H              - Print help
    Q/Esc          - Quit
"""

import argparse
import sys
from pathlib import Path

import numpy as np

try:
    import open3d as o3d
except ImportError:
    print("Open3D not installed. Install with: pip install open3d")
    sys.exit(1)


class PointCloudViewer:
    """Interactive point cloud viewer with simulation features"""

    def __init__(self):
        self.pcd = None
        self.original_pcd = None
        self.vis = None
        self.show_coord_frame = True
        self.show_normals = False
        self.show_bbox = False
        self.point_size = 2.0
        self.geometries = []

    def load(self, file_path: str) -> bool:
        """Load point cloud from file"""
        path = Path(file_path)

        if not path.exists():
            print(f"Error: File not found: {file_path}")
            return False

        supported = ['.pcd', '.ply', '.xyz', '.xyzn', '.xyzrgb', '.pts', '.obj']
        if path.suffix.lower() not in supported:
            # Try loading LAS/LAZ with laspy
            if path.suffix.lower() in ['.las', '.laz']:
                return self._load_las(path)
            print(f"Error: Unsupported format: {path.suffix}")
            print(f"Supported formats: {', '.join(supported + ['.las', '.laz'])}")
            return False

        try:
            print(f"Loading: {file_path}")
            self.pcd = o3d.io.read_point_cloud(str(path))

            if len(self.pcd.points) == 0:
                print("Error: Point cloud is empty")
                return False

            # Keep original for reset
            self.original_pcd = o3d.geometry.PointCloud(self.pcd)

            self._print_info()
            return True

        except Exception as e:
            print(f"Error loading file: {e}")
            return False

    def _load_las(self, path: Path) -> bool:
        """Load LAS/LAZ files using laspy"""
        try:
            import laspy
        except ImportError:
            print("laspy not installed. Install with: pip install laspy[lazrs]")
            return False

        try:
            print(f"Loading LAS file: {path}")
            las = laspy.read(str(path))

            points = np.vstack([las.x, las.y, las.z]).T

            self.pcd = o3d.geometry.PointCloud()
            self.pcd.points = o3d.utility.Vector3dVector(points)

            # Try to get colors
            if hasattr(las, 'red') and hasattr(las, 'green') and hasattr(las, 'blue'):
                colors = np.vstack([las.red, las.green, las.blue]).T
                # Normalize (LAS uses 16-bit colors)
                if colors.max() > 255:
                    colors = colors / 65535.0
                else:
                    colors = colors / 255.0
                self.pcd.colors = o3d.utility.Vector3dVector(colors)

            self.original_pcd = o3d.geometry.PointCloud(self.pcd)
            self._print_info()
            return True

        except Exception as e:
            print(f"Error loading LAS file: {e}")
            return False

    def _print_info(self):
        """Print point cloud information"""
        print("\n" + "=" * 50)
        print("POINT CLOUD INFO")
        print("=" * 50)
        print(f"Points:      {len(self.pcd.points):,}")
        print(f"Has colors:  {self.pcd.has_colors()}")
        print(f"Has normals: {self.pcd.has_normals()}")

        points = np.asarray(self.pcd.points)
        print(f"\nBounding Box:")
        print(f"  Min: [{points.min(axis=0)[0]:.3f}, {points.min(axis=0)[1]:.3f}, {points.min(axis=0)[2]:.3f}]")
        print(f"  Max: [{points.max(axis=0)[0]:.3f}, {points.max(axis=0)[1]:.3f}, {points.max(axis=0)[2]:.3f}]")

        extent = points.max(axis=0) - points.min(axis=0)
        print(f"  Size: [{extent[0]:.3f}, {extent[1]:.3f}, {extent[2]:.3f}]")
        print("=" * 50)

    def create_demo(self):
        """Create a demo point cloud"""
        print("Creating demo point cloud...")

        # Create a sample structure with "cracks"
        points = []
        colors = []

        # Base surface (concrete-like)
        for x in np.linspace(-5, 5, 100):
            for y in np.linspace(-5, 5, 100):
                z = np.sin(x * 0.5) * 0.1 + np.cos(y * 0.5) * 0.1
                z += np.random.normal(0, 0.02)
                points.append([x, y, z])
                colors.append([0.6, 0.6, 0.6])  # Gray concrete

        # Add some "cracks" (red lines)
        for t in np.linspace(0, 1, 200):
            # Crack 1 - diagonal
            x = -3 + t * 4
            y = -2 + t * 3 + np.sin(t * 10) * 0.2
            z = np.sin(x * 0.5) * 0.1 + np.cos(y * 0.5) * 0.1 + 0.05
            points.append([x, y, z])
            colors.append([0.8, 0.1, 0.1])  # Red crack

            # Crack 2 - horizontal
            x = -2 + t * 5
            y = 2 + np.sin(t * 8) * 0.3
            z = np.sin(x * 0.5) * 0.1 + np.cos(y * 0.5) * 0.1 + 0.05
            points.append([x, y, z])
            colors.append([1.0, 0.5, 0.0])  # Orange crack

        self.pcd = o3d.geometry.PointCloud()
        self.pcd.points = o3d.utility.Vector3dVector(np.array(points))
        self.pcd.colors = o3d.utility.Vector3dVector(np.array(colors))

        self.original_pcd = o3d.geometry.PointCloud(self.pcd)
        self._print_info()

    def estimate_normals(self):
        """Estimate normals if not present"""
        if not self.pcd.has_normals():
            print("Estimating normals...")
            self.pcd.estimate_normals(
                search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
            )
            print("Normals estimated.")

    def visualize(self):
        """Open interactive visualization window"""
        if self.pcd is None:
            print("No point cloud loaded")
            return

        print("\n" + "=" * 50)
        print("CONTROLS")
        print("=" * 50)
        print("Mouse drag     - Rotate view")
        print("Scroll         - Zoom in/out")
        print("Shift + drag   - Pan")
        print("Ctrl + drag    - Roll")
        print("+/-            - Point size")
        print("R              - Reset view")
        print("C              - Toggle coordinate frame")
        print("N              - Toggle normals")
        print("B              - Toggle bounding box")
        print("1-9            - Set point size")
        print("H              - Print help")
        print("Q/Esc          - Quit")
        print("=" * 50 + "\n")

        # Create visualizer
        self.vis = o3d.visualization.VisualizerWithKeyCallback()
        self.vis.create_window(window_name="Point Cloud Viewer", width=1280, height=720)

        # Add point cloud
        self.vis.add_geometry(self.pcd)
        self.geometries.append(self.pcd)

        # Add coordinate frame
        coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=1.0, origin=[0, 0, 0]
        )
        self.vis.add_geometry(coord_frame)
        self.geometries.append(coord_frame)

        # Register key callbacks
        self.vis.register_key_callback(ord('C'), self._toggle_coord_frame)
        self.vis.register_key_callback(ord('N'), self._toggle_normals)
        self.vis.register_key_callback(ord('B'), self._toggle_bbox)
        self.vis.register_key_callback(ord('R'), self._reset_view)
        self.vis.register_key_callback(ord('H'), self._print_help)
        self.vis.register_key_callback(ord('+'), self._increase_point_size)
        self.vis.register_key_callback(ord('='), self._increase_point_size)
        self.vis.register_key_callback(ord('-'), self._decrease_point_size)

        # Number keys for point size
        for i in range(1, 10):
            self.vis.register_key_callback(ord(str(i)), lambda vis, i=i: self._set_point_size(vis, i))

        # Set render options
        opt = self.vis.get_render_option()
        opt.point_size = self.point_size
        opt.background_color = np.array([0.1, 0.1, 0.1])
        opt.show_coordinate_frame = True

        # Run visualization
        self.vis.run()
        self.vis.destroy_window()

    def _toggle_coord_frame(self, vis):
        """Toggle coordinate frame visibility"""
        self.show_coord_frame = not self.show_coord_frame
        opt = vis.get_render_option()
        opt.show_coordinate_frame = self.show_coord_frame
        print(f"Coordinate frame: {'ON' if self.show_coord_frame else 'OFF'}")
        return False

    def _toggle_normals(self, vis):
        """Toggle normals visualization"""
        self.show_normals = not self.show_normals

        if self.show_normals:
            if not self.pcd.has_normals():
                self.estimate_normals()

            # Create line set for normals
            points = np.asarray(self.pcd.points)
            normals = np.asarray(self.pcd.normals)

            # Subsample for visualization
            step = max(1, len(points) // 1000)
            indices = np.arange(0, len(points), step)

            line_points = []
            lines = []
            scale = 0.1  # Normal length

            for idx in indices:
                p = points[idx]
                n = normals[idx]
                line_points.append(p)
                line_points.append(p + n * scale)
                lines.append([len(line_points) - 2, len(line_points) - 1])

            line_set = o3d.geometry.LineSet()
            line_set.points = o3d.utility.Vector3dVector(np.array(line_points))
            line_set.lines = o3d.utility.Vector2iVector(np.array(lines))
            line_set.colors = o3d.utility.Vector3dVector([[0, 1, 0]] * len(lines))

            vis.add_geometry(line_set)
            self.geometries.append(line_set)
            print("Normals: ON")
        else:
            # Remove normals line set (last added geometry if it exists)
            if len(self.geometries) > 2:
                vis.remove_geometry(self.geometries[-1], reset_bounding_box=False)
                self.geometries.pop()
            print("Normals: OFF")

        return False

    def _toggle_bbox(self, vis):
        """Toggle bounding box"""
        self.show_bbox = not self.show_bbox

        if self.show_bbox:
            bbox = self.pcd.get_axis_aligned_bounding_box()
            bbox.color = (1, 0, 0)
            vis.add_geometry(bbox)
            self.geometries.append(bbox)
            print("Bounding box: ON")
        else:
            if len(self.geometries) > 2:
                vis.remove_geometry(self.geometries[-1], reset_bounding_box=False)
                self.geometries.pop()
            print("Bounding box: OFF")

        return False

    def _reset_view(self, vis):
        """Reset view to default"""
        vis.reset_view_point(True)
        print("View reset")
        return False

    def _print_help(self, vis):
        """Print help"""
        print("\nControls:")
        print("  Mouse drag: Rotate")
        print("  Scroll: Zoom")
        print("  Shift+drag: Pan")
        print("  C: Coordinate frame")
        print("  N: Normals")
        print("  B: Bounding box")
        print("  R: Reset view")
        print("  +/-: Point size")
        print("  1-9: Set point size")
        print("  Q: Quit\n")
        return False

    def _increase_point_size(self, vis):
        """Increase point size"""
        self.point_size = min(20, self.point_size + 0.5)
        opt = vis.get_render_option()
        opt.point_size = self.point_size
        print(f"Point size: {self.point_size}")
        return False

    def _decrease_point_size(self, vis):
        """Decrease point size"""
        self.point_size = max(0.5, self.point_size - 0.5)
        opt = vis.get_render_option()
        opt.point_size = self.point_size
        print(f"Point size: {self.point_size}")
        return False

    def _set_point_size(self, vis, size):
        """Set specific point size"""
        self.point_size = float(size)
        opt = vis.get_render_option()
        opt.point_size = self.point_size
        print(f"Point size: {self.point_size}")
        return False


def select_file():
    """Open file dialog to select point cloud"""
    try:
        import tkinter as tk
        from tkinter import filedialog

        root = tk.Tk()
        root.withdraw()

        filetypes = [
            ("Point Cloud Files", "*.pcd *.ply *.xyz *.pts *.las *.laz"),
            ("PCD files", "*.pcd"),
            ("PLY files", "*.ply"),
            ("XYZ files", "*.xyz"),
            ("LAS files", "*.las *.laz"),
            ("All files", "*.*")
        ]

        file_path = filedialog.askopenfilename(
            title="Select Point Cloud File",
            filetypes=filetypes
        )

        root.destroy()
        return file_path if file_path else None

    except Exception:
        print("Could not open file dialog. Please provide file path as argument.")
        return None


def main():
    parser = argparse.ArgumentParser(
        description="Point Cloud Viewer using Open3D",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python pcd_viewer.py scan.pcd
    python pcd_viewer.py model.ply
    python pcd_viewer.py --demo
        """
    )

    parser.add_argument("file", nargs="?", help="Path to point cloud file")
    parser.add_argument("--demo", action="store_true", help="Show demo point cloud")

    args = parser.parse_args()

    viewer = PointCloudViewer()

    if args.demo:
        viewer.create_demo()
    elif args.file:
        if not viewer.load(args.file):
            sys.exit(1)
    else:
        # Try to open file dialog
        file_path = select_file()
        if file_path:
            if not viewer.load(file_path):
                sys.exit(1)
        else:
            print("No file selected. Use --demo for a demo or provide a file path.")
            sys.exit(1)

    viewer.visualize()


if __name__ == "__main__":
    main()

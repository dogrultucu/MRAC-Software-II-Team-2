"""
Images to Point Cloud Converter
===============================
Converts a set of overlapping images into a 3D point cloud using
Structure from Motion (SfM) techniques.

Requirements:
    pip install opencv-python open3d numpy tqdm

Usage:
    python images_to_pcd.py --images ./photos --output ./output
    python images_to_pcd.py --images ./photos --output ./output --method sift --dense

No GPU/CUDA required - runs entirely on CPU.
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime

import cv2
import numpy as np
from tqdm import tqdm

try:
    import open3d as o3d
except ImportError:
    print("Error: Open3D required. Install with: pip install open3d")
    sys.exit(1)


class ImageToPointCloud:
    """Convert images to 3D point cloud using Structure from Motion"""

    def __init__(self, output_dir: Path, feature_method: str = "sift",
                 max_features: int = 8000):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.feature_method = feature_method.lower()
        self.max_features = max_features

        # Storage
        self.images = []
        self.image_paths = []
        self.keypoints = []
        self.descriptors = []
        self.K = None  # Camera intrinsic matrix
        self.image_size = None

        # Results
        self.points_3d = []
        self.colors = []
        self.camera_poses = []

    def load_images(self, images_dir: Path) -> int:
        """Load all images from directory"""
        images_dir = Path(images_dir)

        extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']
        image_files = []
        for ext in extensions:
            image_files.extend(images_dir.glob(f"*{ext}"))
            image_files.extend(images_dir.glob(f"*{ext.upper()}"))

        image_files = sorted(set(image_files))

        if len(image_files) < 2:
            print(f"Error: Need at least 2 images, found {len(image_files)}")
            return 0

        print(f"\nLoading {len(image_files)} images...")

        for img_path in tqdm(image_files, desc="Loading"):
            img = cv2.imread(str(img_path))
            if img is not None:
                self.images.append(img)
                self.image_paths.append(img_path)

        if len(self.images) > 0:
            h, w = self.images[0].shape[:2]
            self.image_size = (w, h)

            # Estimate camera intrinsics (assuming standard lens)
            focal = max(w, h) * 1.2
            cx, cy = w / 2, h / 2
            self.K = np.array([
                [focal, 0, cx],
                [0, focal, cy],
                [0, 0, 1]
            ], dtype=np.float64)

            print(f"  Loaded {len(self.images)} images ({w}x{h})")
            print(f"  Estimated focal length: {focal:.1f}px")

        return len(self.images)

    def extract_features(self):
        """Extract features from all images"""
        print(f"\nExtracting features ({self.feature_method.upper()})...")

        if self.feature_method == "sift":
            detector = cv2.SIFT_create(nfeatures=self.max_features)
        elif self.feature_method == "orb":
            detector = cv2.ORB_create(nfeatures=self.max_features)
        elif self.feature_method == "akaze":
            detector = cv2.AKAZE_create()
        else:
            print(f"Unknown method {self.feature_method}, using SIFT")
            detector = cv2.SIFT_create(nfeatures=self.max_features)

        for img in tqdm(self.images, desc="Features"):
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Enhance contrast for better features
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            gray = clahe.apply(gray)

            kp, desc = detector.detectAndCompute(gray, None)

            self.keypoints.append(kp)
            self.descriptors.append(desc)

        total_features = sum(len(kp) for kp in self.keypoints)
        print(f"  Extracted {total_features:,} total features")

    def match_features(self, idx1: int, idx2: int, ratio_thresh: float = 0.75):
        """Match features between two images"""
        desc1 = self.descriptors[idx1]
        desc2 = self.descriptors[idx2]

        if desc1 is None or desc2 is None:
            return [], []

        if len(desc1) < 10 or len(desc2) < 10:
            return [], []

        # Choose matcher based on descriptor type
        if self.feature_method == "orb":
            matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        else:
            matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)

        try:
            matches = matcher.knnMatch(desc1, desc2, k=2)
        except cv2.error:
            return [], []

        # Apply ratio test
        good_matches = []
        for m_n in matches:
            if len(m_n) == 2:
                m, n = m_n
                if m.distance < ratio_thresh * n.distance:
                    good_matches.append(m)

        if len(good_matches) < 10:
            return [], []

        # Get point coordinates
        pts1 = np.float32([self.keypoints[idx1][m.queryIdx].pt for m in good_matches])
        pts2 = np.float32([self.keypoints[idx2][m.trainIdx].pt for m in good_matches])

        return pts1, pts2

    def estimate_pose(self, pts1: np.ndarray, pts2: np.ndarray):
        """Estimate relative camera pose from matched points"""
        if len(pts1) < 8:
            return None, None, None

        # Find essential matrix
        E, mask = cv2.findEssentialMat(
            pts1, pts2, self.K,
            method=cv2.RANSAC,
            prob=0.999,
            threshold=1.0
        )

        if E is None:
            return None, None, None

        # Recover pose
        _, R, t, pose_mask = cv2.recoverPose(E, pts1, pts2, self.K, mask=mask)

        # Filter inliers
        inlier_mask = (mask.ravel() == 1) & (pose_mask.ravel() > 0)

        return R, t, inlier_mask

    def triangulate_points(self, pts1: np.ndarray, pts2: np.ndarray,
                          P1: np.ndarray, P2: np.ndarray,
                          img1: np.ndarray) -> tuple:
        """Triangulate 3D points from 2D correspondences"""
        if len(pts1) < 5:
            return np.array([]), np.array([])

        # Triangulate
        pts1_h = pts1.T  # 2xN
        pts2_h = pts2.T  # 2xN

        points_4d = cv2.triangulatePoints(P1, P2, pts1_h, pts2_h)
        points_4d /= points_4d[3]  # Normalize homogeneous

        points_3d = points_4d[:3].T  # Nx3

        # Filter invalid points
        # - Must be in front of both cameras
        # - Reasonable depth range
        valid = (points_3d[:, 2] > 0) & (points_3d[:, 2] < 1000)

        # Filter by reprojection error
        if np.sum(valid) > 0:
            reproj = (P1 @ points_4d).T
            reproj = reproj[:, :2] / reproj[:, 2:3]
            reproj_error = np.linalg.norm(reproj - pts1, axis=1)
            valid &= (reproj_error < 5.0)

        points_3d = points_3d[valid]
        pts1_valid = pts1[valid]

        # Get colors from first image
        colors = []
        h, w = img1.shape[:2]
        for pt in pts1_valid:
            x, y = int(pt[0]), int(pt[1])
            if 0 <= x < w and 0 <= y < h:
                bgr = img1[y, x]
                rgb = bgr[::-1] / 255.0  # BGR to RGB, normalize
                colors.append(rgb)
            else:
                colors.append([0.5, 0.5, 0.5])

        return points_3d, np.array(colors)

    def reconstruct_incremental(self):
        """Incremental Structure from Motion reconstruction"""
        print("\n" + "=" * 60)
        print("STRUCTURE FROM MOTION RECONSTRUCTION")
        print("=" * 60)

        n_images = len(self.images)
        if n_images < 2:
            print("Error: Need at least 2 images")
            return False

        # Initialize with first pair
        print("\nInitializing with first image pair...")

        pts1, pts2 = self.match_features(0, 1)
        if len(pts1) < 20:
            print("  Not enough matches for first pair, trying next...")
            # Try to find a good initial pair
            for i in range(min(n_images - 1, 5)):
                for j in range(i + 1, min(n_images, i + 5)):
                    pts1, pts2 = self.match_features(i, j)
                    if len(pts1) >= 20:
                        print(f"  Using images {i} and {j} as initial pair")
                        break
                if len(pts1) >= 20:
                    break

        if len(pts1) < 20:
            print("Error: Could not find good initial pair")
            return False

        R, t, inlier_mask = self.estimate_pose(pts1, pts2)
        if R is None:
            print("Error: Could not estimate initial pose")
            return False

        pts1_inlier = pts1[inlier_mask]
        pts2_inlier = pts2[inlier_mask]

        print(f"  Initial pair: {len(pts1_inlier)} inlier matches")

        # Set up projection matrices
        P1 = self.K @ np.hstack([np.eye(3), np.zeros((3, 1))])
        P2 = self.K @ np.hstack([R, t])

        # Triangulate initial points
        points_3d, colors = self.triangulate_points(
            pts1_inlier, pts2_inlier, P1, P2, self.images[0]
        )

        self.points_3d.extend(points_3d.tolist())
        self.colors.extend(colors.tolist())

        print(f"  Initial triangulation: {len(points_3d)} points")

        # Store camera poses
        self.camera_poses = [
            np.eye(4),  # First camera at origin
            np.vstack([np.hstack([R, t]), [0, 0, 0, 1]])  # Second camera
        ]

        # Add remaining images incrementally
        print("\nAdding images incrementally...")

        for i in tqdm(range(2, n_images), desc="Processing"):
            best_matches = 0
            best_pair_idx = -1
            best_pts1, best_pts2 = None, None

            # Find best matching previous image
            for j in range(max(0, i - 5), i):
                pts1, pts2 = self.match_features(j, i)
                if len(pts1) > best_matches:
                    best_matches = len(pts1)
                    best_pair_idx = j
                    best_pts1, best_pts2 = pts1, pts2

            if best_matches < 15:
                continue

            # Estimate pose relative to best matching image
            R_rel, t_rel, inlier_mask = self.estimate_pose(best_pts1, best_pts2)
            if R_rel is None:
                continue

            pts1_inlier = best_pts1[inlier_mask]
            pts2_inlier = best_pts2[inlier_mask]

            # Get previous camera pose
            if best_pair_idx < len(self.camera_poses):
                prev_pose = self.camera_poses[best_pair_idx]
                R_prev = prev_pose[:3, :3]
                t_prev = prev_pose[:3, 3:4]

                # Compute new absolute pose
                R_new = R_rel @ R_prev
                t_new = R_rel @ t_prev + t_rel

                new_pose = np.vstack([np.hstack([R_new, t_new]), [0, 0, 0, 1]])
                self.camera_poses.append(new_pose)

                # Triangulate new points
                P_prev = self.K @ np.hstack([R_prev, t_prev])
                P_new = self.K @ np.hstack([R_new, t_new])

                new_points, new_colors = self.triangulate_points(
                    pts1_inlier, pts2_inlier, P_prev, P_new,
                    self.images[best_pair_idx]
                )

                if len(new_points) > 0:
                    self.points_3d.extend(new_points.tolist())
                    self.colors.extend(new_colors.tolist())

        print(f"\n  Total points reconstructed: {len(self.points_3d):,}")
        print(f"  Camera poses recovered: {len(self.camera_poses)}")

        return len(self.points_3d) > 0

    def reconstruct_pairwise(self):
        """Simpler pairwise reconstruction (more points, less accurate)"""
        print("\n" + "=" * 60)
        print("PAIRWISE RECONSTRUCTION")
        print("=" * 60)

        n_images = len(self.images)

        # Process consecutive pairs
        print("\nProcessing image pairs...")

        for i in tqdm(range(n_images - 1), desc="Pairs"):
            # Try matching with next few images
            for j in range(i + 1, min(i + 4, n_images)):
                pts1, pts2 = self.match_features(i, j)

                if len(pts1) < 15:
                    continue

                R, t, inlier_mask = self.estimate_pose(pts1, pts2)
                if R is None:
                    continue

                pts1_inlier = pts1[inlier_mask]
                pts2_inlier = pts2[inlier_mask]

                # Projection matrices
                P1 = self.K @ np.hstack([np.eye(3), np.zeros((3, 1))])
                P2 = self.K @ np.hstack([R, t])

                # Triangulate
                points_3d, colors = self.triangulate_points(
                    pts1_inlier, pts2_inlier, P1, P2, self.images[i]
                )

                if len(points_3d) > 0:
                    self.points_3d.extend(points_3d.tolist())
                    self.colors.extend(colors.tolist())

        print(f"\n  Total points: {len(self.points_3d):,}")
        return len(self.points_3d) > 0

    def create_point_cloud(self) -> o3d.geometry.PointCloud:
        """Create Open3D point cloud from reconstructed points"""
        pcd = o3d.geometry.PointCloud()

        if len(self.points_3d) == 0:
            return pcd

        points = np.array(self.points_3d)
        colors = np.array(self.colors) if self.colors else None

        # Remove outliers using statistical filtering
        pcd.points = o3d.utility.Vector3dVector(points)
        if colors is not None and len(colors) == len(points):
            pcd.colors = o3d.utility.Vector3dVector(colors)

        # Statistical outlier removal
        print("\nCleaning point cloud...")
        pcd_clean, ind = pcd.remove_statistical_outlier(
            nb_neighbors=20, std_ratio=2.0
        )
        print(f"  Removed {len(points) - len(ind)} outliers")

        # Estimate normals
        print("  Estimating normals...")
        pcd_clean.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(
                radius=0.1, max_nn=30
            )
        )

        return pcd_clean

    def densify_point_cloud(self, pcd: o3d.geometry.PointCloud,
                           depth: int = 8) -> o3d.geometry.PointCloud:
        """Attempt to densify using Poisson reconstruction"""
        print("\nDensifying point cloud...")

        if len(pcd.points) < 100:
            print("  Too few points for densification")
            return pcd

        try:
            # Poisson surface reconstruction
            mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
                pcd, depth=depth
            )

            # Remove low-density vertices
            vertices_to_remove = densities < np.quantile(densities, 0.01)
            mesh.remove_vertices_by_mask(vertices_to_remove)

            # Sample points from mesh
            dense_pcd = mesh.sample_points_uniformly(
                number_of_points=len(pcd.points) * 3
            )

            # Transfer colors using nearest neighbor
            if pcd.has_colors():
                tree = o3d.geometry.KDTreeFlann(pcd)
                colors = []
                orig_colors = np.asarray(pcd.colors)

                for pt in np.asarray(dense_pcd.points):
                    [_, idx, _] = tree.search_knn_vector_3d(pt, 1)
                    colors.append(orig_colors[idx[0]])

                dense_pcd.colors = o3d.utility.Vector3dVector(np.array(colors))

            print(f"  Densified: {len(pcd.points)} -> {len(dense_pcd.points)} points")
            return dense_pcd

        except Exception as e:
            print(f"  Densification failed: {e}")
            return pcd

    def save(self, pcd: o3d.geometry.PointCloud, name: str = "point_cloud"):
        """Save point cloud to files"""
        print("\n" + "=" * 60)
        print("SAVING RESULTS")
        print("=" * 60)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save PLY
        ply_path = self.output_dir / f"{name}_{timestamp}.ply"
        o3d.io.write_point_cloud(str(ply_path), pcd)
        print(f"  Saved: {ply_path}")

        # Save PCD
        pcd_path = self.output_dir / f"{name}_{timestamp}.pcd"
        o3d.io.write_point_cloud(str(pcd_path), pcd)
        print(f"  Saved: {pcd_path}")

        # Save XYZ (simple text format)
        xyz_path = self.output_dir / f"{name}_{timestamp}.xyz"
        points = np.asarray(pcd.points)
        if pcd.has_colors():
            colors = (np.asarray(pcd.colors) * 255).astype(int)
            data = np.hstack([points, colors])
            np.savetxt(str(xyz_path), data, fmt='%.6f %.6f %.6f %d %d %d')
        else:
            np.savetxt(str(xyz_path), points, fmt='%.6f')
        print(f"  Saved: {xyz_path}")

        return ply_path, pcd_path

    def visualize(self, pcd: o3d.geometry.PointCloud):
        """Visualize the point cloud"""
        print("\nOpening viewer...")
        print("  Controls: Mouse=Rotate, Scroll=Zoom, Q=Quit")

        # Add coordinate frame
        coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=0.5, origin=[0, 0, 0]
        )

        o3d.visualization.draw_geometries(
            [pcd, coord_frame],
            window_name="Reconstructed Point Cloud",
            width=1280,
            height=720,
            point_show_normal=False
        )


def main():
    parser = argparse.ArgumentParser(
        description="Convert images to 3D point cloud",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python images_to_pcd.py --images ./photos --output ./output
  python images_to_pcd.py --images ./photos --output ./output --method sift
  python images_to_pcd.py --images ./photos --output ./output --dense --visualize
        """
    )

    parser.add_argument("--images", "-i", type=str, required=True,
                       help="Path to folder containing images")
    parser.add_argument("--output", "-o", type=str, default="./output_pcd",
                       help="Output directory (default: ./output_pcd)")
    parser.add_argument("--method", "-m", type=str, default="sift",
                       choices=["sift", "orb", "akaze"],
                       help="Feature detection method (default: sift)")
    parser.add_argument("--max-features", type=int, default=8000,
                       help="Maximum features per image (default: 8000)")
    parser.add_argument("--mode", type=str, default="incremental",
                       choices=["incremental", "pairwise"],
                       help="Reconstruction mode (default: incremental)")
    parser.add_argument("--dense", "-d", action="store_true",
                       help="Attempt to densify the point cloud")
    parser.add_argument("--visualize", "-v", action="store_true",
                       help="Visualize result after reconstruction")
    parser.add_argument("--name", "-n", type=str, default="point_cloud",
                       help="Output file name prefix (default: point_cloud)")

    args = parser.parse_args()

    # Check input directory
    images_dir = Path(args.images)
    if not images_dir.exists():
        print(f"Error: Images directory not found: {images_dir}")
        sys.exit(1)

    # Create converter
    converter = ImageToPointCloud(
        output_dir=args.output,
        feature_method=args.method,
        max_features=args.max_features
    )

    # Load images
    n_images = converter.load_images(images_dir)
    if n_images < 2:
        sys.exit(1)

    # Extract features
    converter.extract_features()

    # Reconstruct
    if args.mode == "incremental":
        success = converter.reconstruct_incremental()
    else:
        success = converter.reconstruct_pairwise()

    if not success:
        print("Reconstruction failed!")
        sys.exit(1)

    # Create point cloud
    pcd = converter.create_point_cloud()

    if len(pcd.points) == 0:
        print("Error: No points reconstructed")
        sys.exit(1)

    # Densify if requested
    if args.dense:
        pcd = converter.densify_point_cloud(pcd)

    # Save
    ply_path, pcd_path = converter.save(pcd, name=args.name)

    # Summary
    print("\n" + "=" * 60)
    print("RECONSTRUCTION COMPLETE")
    print("=" * 60)
    print(f"  Points: {len(pcd.points):,}")
    print(f"  Output: {ply_path}")

    # Visualize if requested
    if args.visualize:
        converter.visualize(pcd)


if __name__ == "__main__":
    main()

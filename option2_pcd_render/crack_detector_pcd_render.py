"""
Option 2: Point Cloud First - Render Views and Detect
======================================================
Workflow:
1. Build point cloud from photos using photogrammetry
2. Render multiple 2D views from the point cloud
3. Run YOLOv8 crack detection on rendered views
4. Map detections back to 3D point cloud coordinates
5. Export annotated point cloud

Setup:
    conda env create -f environment.yml
    conda activate crack_detect_pcd
    python crack_detector_pcd_render.py --photos ./photos --model ./best.pt --output ./output

Alternative: Use existing point cloud
    python crack_detector_pcd_render.py --pcd ./existing.ply --model ./best.pt --output ./output
"""

import argparse
import json
import math
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from tqdm import tqdm


@dataclass
class ViewCamera:
    """Camera parameters for a rendered view"""
    position: np.ndarray
    look_at: np.ndarray
    up: np.ndarray
    intrinsic: np.ndarray
    extrinsic: np.ndarray
    width: int
    height: int
    view_id: int


@dataclass
class CrackDetection2D:
    """2D crack detection from rendered view"""
    view_id: int
    camera: ViewCamera
    bbox: tuple
    confidence: float
    severity: str
    pixel_coords: list = field(default_factory=list)


@dataclass
class CrackDetection3D:
    """3D crack detection on point cloud"""
    points_3d: np.ndarray
    point_indices: list
    severity: str
    confidence: float
    source_views: list
    centroid: np.ndarray = None

    def __post_init__(self):
        if self.points_3d is not None and len(self.points_3d) > 0:
            self.centroid = np.mean(self.points_3d, axis=0)


class ICPRefiner:
    """ICP-based point cloud refinement for improved reconstruction quality"""

    def __init__(self, max_correspondence_distance: float = 0.05,
                 max_iterations: int = 50):
        self.max_correspondence_distance = max_correspondence_distance
        self.max_iterations = max_iterations

    def refine_point_to_point(self, source, target):
        """
        Point-to-point ICP alignment.
        Minimizes distances between corresponding points.
        """
        import open3d as o3d

        print("  Applying Point-to-Point ICP refinement...")

        initial_fitness = self._evaluate_registration(source, target)
        print(f"    Initial fitness: {initial_fitness:.4f}")

        reg_result = o3d.pipelines.registration.registration_icp(
            source, target,
            self.max_correspondence_distance,
            np.eye(4),
            o3d.pipelines.registration.TransformationEstimationPointToPoint(),
            o3d.pipelines.registration.ICPConvergenceCriteria(
                max_iteration=self.max_iterations
            )
        )

        source_refined = o3d.geometry.PointCloud(source)
        source_refined.transform(reg_result.transformation)

        print(f"    Final fitness: {reg_result.fitness:.4f}")
        print(f"    RMSE: {reg_result.inlier_rmse:.6f}")

        return source_refined, reg_result.transformation

    def refine_point_to_plane(self, source, target):
        """
        Point-to-plane ICP alignment.
        Better for smooth surfaces, requires normals.
        """
        import open3d as o3d

        print("  Applying Point-to-Plane ICP refinement...")

        if not source.has_normals():
            source.estimate_normals(
                search_param=o3d.geometry.KDTreeSearchParamHybrid(
                    radius=0.1, max_nn=30
                )
            )
        if not target.has_normals():
            target.estimate_normals(
                search_param=o3d.geometry.KDTreeSearchParamHybrid(
                    radius=0.1, max_nn=30
                )
            )

        initial_fitness = self._evaluate_registration(source, target)
        print(f"    Initial fitness: {initial_fitness:.4f}")

        reg_result = o3d.pipelines.registration.registration_icp(
            source, target,
            self.max_correspondence_distance,
            np.eye(4),
            o3d.pipelines.registration.TransformationEstimationPointToPlane(),
            o3d.pipelines.registration.ICPConvergenceCriteria(
                max_iteration=self.max_iterations
            )
        )

        source_refined = o3d.geometry.PointCloud(source)
        source_refined.transform(reg_result.transformation)

        print(f"    Final fitness: {reg_result.fitness:.4f}")
        print(f"    RMSE: {reg_result.inlier_rmse:.6f}")

        return source_refined, reg_result.transformation

    def refine_colored_icp(self, source, target, lambda_geometric: float = 0.968):
        """
        Colored ICP alignment.
        Uses both geometry and color for alignment.
        """
        import open3d as o3d

        print("  Applying Colored ICP refinement...")

        if not source.has_colors() or not target.has_colors():
            print("    Warning: Missing colors, using Point-to-Point ICP")
            return self.refine_point_to_point(source, target)

        if not source.has_normals():
            source.estimate_normals(
                search_param=o3d.geometry.KDTreeSearchParamHybrid(
                    radius=0.1, max_nn=30
                )
            )
        if not target.has_normals():
            target.estimate_normals(
                search_param=o3d.geometry.KDTreeSearchParamHybrid(
                    radius=0.1, max_nn=30
                )
            )

        initial_fitness = self._evaluate_registration(source, target)
        print(f"    Initial fitness: {initial_fitness:.4f}")

        reg_result = o3d.pipelines.registration.registration_colored_icp(
            source, target,
            self.max_correspondence_distance,
            np.eye(4),
            o3d.pipelines.registration.TransformationEstimationForColoredICP(
                lambda_geometric=lambda_geometric
            ),
            o3d.pipelines.registration.ICPConvergenceCriteria(
                max_iteration=self.max_iterations
            )
        )

        source_refined = o3d.geometry.PointCloud(source)
        source_refined.transform(reg_result.transformation)

        print(f"    Final fitness: {reg_result.fitness:.4f}")
        print(f"    RMSE: {reg_result.inlier_rmse:.6f}")

        return source_refined, reg_result.transformation

    def refine_multi_scale(self, source, target, voxel_sizes=[0.05, 0.025, 0.01],
                          icp_type: str = "colored"):
        """Multi-scale ICP for robust alignment."""
        import open3d as o3d

        print(f"  Applying Multi-scale {icp_type.upper()} ICP refinement...")

        current_transformation = np.eye(4)

        for i, voxel_size in enumerate(voxel_sizes):
            print(f"    Scale {i+1}/{len(voxel_sizes)} (voxel: {voxel_size})")

            source_down = source.voxel_down_sample(voxel_size)
            target_down = target.voxel_down_sample(voxel_size)

            source_down.estimate_normals(
                search_param=o3d.geometry.KDTreeSearchParamHybrid(
                    radius=voxel_size * 2, max_nn=30
                )
            )
            target_down.estimate_normals(
                search_param=o3d.geometry.KDTreeSearchParamHybrid(
                    radius=voxel_size * 2, max_nn=30
                )
            )

            source_down.transform(current_transformation)
            max_dist = voxel_size * 2

            if icp_type == "colored" and source.has_colors() and target.has_colors():
                reg = o3d.pipelines.registration.registration_colored_icp(
                    source_down, target_down, max_dist, np.eye(4),
                    o3d.pipelines.registration.TransformationEstimationForColoredICP(),
                    o3d.pipelines.registration.ICPConvergenceCriteria(
                        max_iteration=self.max_iterations
                    )
                )
            elif icp_type == "point_to_plane":
                reg = o3d.pipelines.registration.registration_icp(
                    source_down, target_down, max_dist, np.eye(4),
                    o3d.pipelines.registration.TransformationEstimationPointToPlane(),
                    o3d.pipelines.registration.ICPConvergenceCriteria(
                        max_iteration=self.max_iterations
                    )
                )
            else:
                reg = o3d.pipelines.registration.registration_icp(
                    source_down, target_down, max_dist, np.eye(4),
                    o3d.pipelines.registration.TransformationEstimationPointToPoint(),
                    o3d.pipelines.registration.ICPConvergenceCriteria(
                        max_iteration=self.max_iterations
                    )
                )

            current_transformation = reg.transformation @ current_transformation
            print(f"      Fitness: {reg.fitness:.4f}, RMSE: {reg.inlier_rmse:.6f}")

        source_refined = o3d.geometry.PointCloud(source)
        source_refined.transform(current_transformation)

        return source_refined, current_transformation

    def _evaluate_registration(self, source, target):
        """Evaluate alignment quality"""
        import open3d as o3d

        evaluation = o3d.pipelines.registration.evaluate_registration(
            source, target, self.max_correspondence_distance, np.eye(4)
        )
        return evaluation.fitness

    def merge_with_icp(self, point_clouds: list, icp_type: str = "colored"):
        """Incrementally merge multiple point clouds using ICP."""
        import open3d as o3d

        if len(point_clouds) == 0:
            return None
        if len(point_clouds) == 1:
            return point_clouds[0]

        print(f"\n  Merging {len(point_clouds)} clouds with {icp_type} ICP...")

        merged = o3d.geometry.PointCloud(point_clouds[0])

        for i, pcd in enumerate(point_clouds[1:], 1):
            print(f"    Aligning cloud {i+1}/{len(point_clouds)}...")

            if icp_type == "colored":
                aligned, _ = self.refine_colored_icp(pcd, merged)
            elif icp_type == "point_to_plane":
                aligned, _ = self.refine_point_to_plane(pcd, merged)
            else:
                aligned, _ = self.refine_point_to_point(pcd, merged)

            merged += aligned

            if len(merged.points) > 500000:
                merged = merged.voxel_down_sample(0.005)

        merged, _ = merged.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
        print(f"  Final merged cloud: {len(merged.points)} points")

        return merged


class PointCloudBuilder:
    """Build point cloud from photos using photogrammetry"""

    def __init__(self, output_dir: Path, use_icp: bool = True,
                 icp_type: str = "colored"):
        self.output_dir = output_dir
        self.point_cloud = None
        self.colors = None
        self.normals = None
        self.use_icp = use_icp
        self.icp_type = icp_type
        self.icp_refiner = ICPRefiner() if use_icp else None

    def build_from_photos(self, photos_dir: Path) -> bool:
        """Build point cloud from photos"""
        import open3d as o3d

        print("\n" + "=" * 60)
        print("BUILDING POINT CLOUD FROM PHOTOS")
        print("=" * 60)

        image_paths = self._get_image_paths(photos_dir)

        if len(image_paths) < 3:
            print(f"Error: Need at least 3 images, found {len(image_paths)}")
            return False

        print(f"Processing {len(image_paths)} images...")

        # Try COLMAP first if available
        if self._check_colmap():
            success = self._build_with_colmap(photos_dir, image_paths)
            if success:
                return True

        # Fallback to Open3D
        return self._build_with_open3d(image_paths)

    def load_existing(self, pcd_path: Path) -> bool:
        """Load an existing point cloud file"""
        import open3d as o3d

        print(f"\nLoading existing point cloud: {pcd_path}")

        supported_formats = ['.ply', '.pcd', '.xyz', '.pts', '.las', '.laz']
        ext = pcd_path.suffix.lower()

        if ext not in supported_formats:
            print(f"Error: Unsupported format {ext}")
            return False

        try:
            if ext in ['.las', '.laz']:
                # Use laspy for LAS/LAZ files
                import laspy
                las = laspy.read(str(pcd_path))
                points = np.vstack([las.x, las.y, las.z]).T

                # Get colors if available
                if hasattr(las, 'red'):
                    colors = np.vstack([las.red, las.green, las.blue]).T / 65535.0
                else:
                    colors = np.ones_like(points) * 0.5

                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(points)
                pcd.colors = o3d.utility.Vector3dVector(colors)
            else:
                pcd = o3d.io.read_point_cloud(str(pcd_path))

            self.point_cloud = pcd
            self.colors = np.asarray(pcd.colors) if pcd.has_colors() else None

            # Estimate normals if not present
            if not pcd.has_normals():
                print("  Estimating normals...")
                pcd.estimate_normals(
                    search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
                )

            self.normals = np.asarray(pcd.normals)

            print(f"  Loaded {len(pcd.points)} points")
            return True

        except Exception as e:
            print(f"Error loading point cloud: {e}")
            return False

    def _get_image_paths(self, photos_dir: Path) -> list:
        """Get all image paths from directory"""
        extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
        paths = []
        for ext in extensions:
            paths.extend(photos_dir.glob(f"*{ext}"))
            paths.extend(photos_dir.glob(f"*{ext.upper()}"))
        return sorted(set(paths))

    def _check_colmap(self) -> bool:
        """Check if COLMAP is available"""
        try:
            result = subprocess.run(["colmap", "--help"], capture_output=True)
            return result.returncode == 0
        except FileNotFoundError:
            return False

    def _build_with_colmap(self, photos_dir: Path, image_paths: list) -> bool:
        """Build point cloud using COLMAP"""
        import open3d as o3d

        print("Using COLMAP for reconstruction...")

        workspace = self.output_dir / "colmap_workspace"
        workspace.mkdir(parents=True, exist_ok=True)

        database = workspace / "database.db"
        sparse = workspace / "sparse"
        dense = workspace / "dense"

        sparse.mkdir(exist_ok=True)
        dense.mkdir(exist_ok=True)

        try:
            # Feature extraction
            print("  Extracting features...")
            subprocess.run([
                "colmap", "feature_extractor",
                "--database_path", str(database),
                "--image_path", str(photos_dir),
                "--ImageReader.single_camera", "1"
            ], check=True, capture_output=True)

            # Matching
            print("  Matching features...")
            subprocess.run([
                "colmap", "exhaustive_matcher",
                "--database_path", str(database)
            ], check=True, capture_output=True)

            # Sparse reconstruction
            print("  Sparse reconstruction...")
            subprocess.run([
                "colmap", "mapper",
                "--database_path", str(database),
                "--image_path", str(photos_dir),
                "--output_path", str(sparse)
            ], check=True, capture_output=True)

            # Dense reconstruction
            print("  Dense reconstruction...")
            subprocess.run([
                "colmap", "image_undistorter",
                "--image_path", str(photos_dir),
                "--input_path", str(sparse / "0"),
                "--output_path", str(dense)
            ], check=True, capture_output=True)

            subprocess.run([
                "colmap", "patch_match_stereo",
                "--workspace_path", str(dense)
            ], check=True, capture_output=True)

            subprocess.run([
                "colmap", "stereo_fusion",
                "--workspace_path", str(dense),
                "--output_path", str(dense / "fused.ply")
            ], check=True, capture_output=True)

            # Load result
            pcd = o3d.io.read_point_cloud(str(dense / "fused.ply"))
            self.point_cloud = pcd

            # Estimate normals
            pcd.estimate_normals(
                search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
            )
            self.normals = np.asarray(pcd.normals)
            self.colors = np.asarray(pcd.colors) if pcd.has_colors() else None

            # Save a copy
            o3d.io.write_point_cloud(str(self.output_dir / "reconstructed.ply"), pcd)

            print(f"  Built point cloud with {len(pcd.points)} points")
            return True

        except subprocess.CalledProcessError as e:
            print(f"COLMAP failed: {e}")
            return False

    def _build_with_open3d(self, image_paths: list) -> bool:
        """Build point cloud using Open3D feature matching"""
        import open3d as o3d

        print("Using Open3D for reconstruction...")

        # Read first image for dimensions
        first_img = cv2.imread(str(image_paths[0]))
        h, w = first_img.shape[:2]

        # Estimate intrinsics
        focal = max(w, h) * 1.2
        cx, cy = w / 2, h / 2

        # Feature detection
        orb = cv2.ORB_create(nfeatures=5000)
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

        all_kp = []
        all_desc = []

        print("  Extracting features...")
        for path in tqdm(image_paths):
            img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            kp, desc = orb.detectAndCompute(img, None)
            all_kp.append(kp)
            all_desc.append(desc)

        # Triangulate between image pairs
        print("  Triangulating...")
        all_points = []
        all_colors = []
        partial_point_clouds = []  # For ICP merging

        K = np.array([[focal, 0, cx], [0, focal, cy], [0, 0, 1]])

        for i in tqdm(range(len(image_paths) - 1)):
            if all_desc[i] is None or all_desc[i+1] is None:
                continue

            matches = bf.match(all_desc[i], all_desc[i+1])
            matches = sorted(matches, key=lambda x: x.distance)[:500]

            if len(matches) < 20:
                continue

            pts1 = np.float32([all_kp[i][m.queryIdx].pt for m in matches])
            pts2 = np.float32([all_kp[i+1][m.trainIdx].pt for m in matches])

            E, mask = cv2.findEssentialMat(pts1, pts2, K)
            if E is None:
                continue

            _, R, t, mask = cv2.recoverPose(E, pts1, pts2, K)

            P1 = K @ np.hstack([np.eye(3), np.zeros((3, 1))])
            P2 = K @ np.hstack([R, t])

            pts1_h = pts1[mask.ravel() == 1].T
            pts2_h = pts2[mask.ravel() == 1].T

            if pts1_h.shape[1] < 5:
                continue

            pts4d = cv2.triangulatePoints(P1, P2, pts1_h, pts2_h)
            pts4d /= pts4d[3]

            valid = (pts4d[2] > 0) & (pts4d[2] < 100)
            pts3d = pts4d[:3, valid].T

            all_points.extend(pts3d.tolist())

            # Get colors
            pair_colors = []
            img_color = cv2.imread(str(image_paths[i]))
            for j, pt in enumerate(pts1_h.T):
                if valid[j]:
                    x, y = int(pt[0]), int(pt[1])
                    if 0 <= x < w and 0 <= y < h:
                        c = img_color[y, x] / 255.0
                        all_colors.append(c[::-1])
                        pair_colors.append(c[::-1])

            # Store partial point cloud for ICP merging
            if self.use_icp and len(pts3d) > 10:
                partial_pcd = o3d.geometry.PointCloud()
                partial_pcd.points = o3d.utility.Vector3dVector(
                    pts3d.astype(np.float64)
                )
                if pair_colors:
                    partial_pcd.colors = o3d.utility.Vector3dVector(
                        np.array(pair_colors[:len(pts3d)]).astype(np.float64)
                    )
                partial_point_clouds.append(partial_pcd)

        if len(all_points) < 100:
            print("Warning: Sparse reconstruction")

        # Create point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(np.array(all_points))

        if all_colors:
            pcd.colors = o3d.utility.Vector3dVector(np.array(all_colors[:len(all_points)]))

        # Clean up
        pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)

        # Apply ICP refinement if enabled and we have partial clouds
        if self.use_icp and len(partial_point_clouds) > 1:
            print("\n  Applying ICP refinement...")
            pcd = self.icp_refiner.merge_with_icp(
                partial_point_clouds, icp_type=self.icp_type
            )

        # Estimate normals
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
        )

        self.point_cloud = pcd
        self.normals = np.asarray(pcd.normals)
        self.colors = np.asarray(pcd.colors) if pcd.has_colors() else None

        # Save
        o3d.io.write_point_cloud(str(self.output_dir / "reconstructed.ply"), pcd)

        print(f"  Built point cloud with {len(pcd.points)} points")
        return True


class ViewRenderer:
    """Render 2D views from point cloud"""

    def __init__(self, point_cloud, output_dir: Path,
                 render_width: int = 1280, render_height: int = 960,
                 render_mode: str = "surface"):
        self.point_cloud = point_cloud
        self.output_dir = output_dir
        self.render_width = render_width
        self.render_height = render_height
        self.render_mode = render_mode  # "surface" or "points"
        self.views_dir = output_dir / "rendered_views"
        self.views_dir.mkdir(parents=True, exist_ok=True)
        self.cameras = []
        self.mesh = None

        # Create mesh for surface rendering
        if render_mode == "surface":
            self.mesh = self._create_mesh()

    def _create_mesh(self):
        """Create a mesh from point cloud using Poisson surface reconstruction"""
        import open3d as o3d

        print("\n  Creating surface mesh from point cloud...")

        pcd = self.point_cloud

        # Ensure normals exist and are oriented consistently
        if not pcd.has_normals():
            print("    Estimating normals...")
            pcd.estimate_normals(
                search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
            )

        # Orient normals consistently (important for Poisson)
        print("    Orienting normals...")
        pcd.orient_normals_consistent_tangent_plane(k=15)

        # Poisson surface reconstruction
        print("    Running Poisson surface reconstruction...")
        try:
            mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
                pcd, depth=9, width=0, scale=1.1, linear_fit=False
            )

            # Remove low-density vertices (artifacts)
            densities = np.asarray(densities)
            density_threshold = np.quantile(densities, 0.01)
            vertices_to_remove = densities < density_threshold
            mesh.remove_vertices_by_mask(vertices_to_remove)

            # Transfer colors from point cloud to mesh
            if pcd.has_colors():
                print("    Transferring colors to mesh...")
                mesh.vertex_colors = mesh.vertex_colors  # Initialize
                # Use nearest neighbor to transfer colors
                pcd_tree = o3d.geometry.KDTreeFlann(pcd)
                mesh_vertices = np.asarray(mesh.vertices)
                mesh_colors = np.zeros_like(mesh_vertices)
                pcd_colors = np.asarray(pcd.colors)

                for i, vertex in enumerate(mesh_vertices):
                    [k, idx, _] = pcd_tree.search_knn_vector_3d(vertex, 1)
                    if k > 0:
                        mesh_colors[i] = pcd_colors[idx[0]]

                mesh.vertex_colors = o3d.utility.Vector3dVector(mesh_colors)

            # Compute vertex normals for smooth shading
            mesh.compute_vertex_normals()

            print(f"    Mesh created: {len(mesh.vertices)} vertices, {len(mesh.triangles)} triangles")
            return mesh

        except Exception as e:
            print(f"    Poisson reconstruction failed: {e}")
            print("    Trying Ball Pivoting Algorithm...")

            try:
                # Estimate radius for ball pivoting
                distances = pcd.compute_nearest_neighbor_distance()
                avg_dist = np.mean(distances)
                radii = [avg_dist * 2, avg_dist * 4, avg_dist * 8]

                mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
                    pcd, o3d.utility.DoubleVector(radii)
                )

                if pcd.has_colors():
                    mesh.vertex_colors = pcd.colors

                mesh.compute_vertex_normals()
                print(f"    Mesh created: {len(mesh.vertices)} vertices, {len(mesh.triangles)} triangles")
                return mesh

            except Exception as e2:
                print(f"    Ball Pivoting also failed: {e2}")
                print("    Falling back to point rendering mode")
                self.render_mode = "points"
                return None

    def generate_viewpoints(self, num_views: int = 14,
                           view_type: str = "ortho14") -> list[ViewCamera]:
        """Generate camera viewpoints around the point cloud

        view_type options:
            - "ortho14": 6 orthographic (front/back/left/right/top/bottom) + 8 corner views (recommended)
            - "spherical": Random spherical distribution
            - "orbit": Horizontal orbit
        """
        import open3d as o3d

        # Get point cloud bounding box
        points = np.asarray(self.point_cloud.points)
        center = points.mean(axis=0)
        bbox = self.point_cloud.get_axis_aligned_bounding_box()
        extent = bbox.get_extent()
        radius = np.linalg.norm(extent) * 1.5

        cameras = []

        if view_type == "ortho14":
            # 14 systematic views for complete coverage
            # 6 orthographic (axis-aligned) + 8 corner (diagonal) views
            view_id = 0

            # === 6 ORTHOGRAPHIC VIEWS (axis-aligned) ===
            ortho_views = [
                # (direction from center, up vector, name)
                (np.array([0, 1, 0]), np.array([0, 0, 1]), "front"),      # Front (+Y)
                (np.array([0, -1, 0]), np.array([0, 0, 1]), "back"),      # Back (-Y)
                (np.array([-1, 0, 0]), np.array([0, 0, 1]), "left"),      # Left (-X)
                (np.array([1, 0, 0]), np.array([0, 0, 1]), "right"),      # Right (+X)
                (np.array([0, 0, 1]), np.array([0, 1, 0]), "top"),        # Top (+Z)
                (np.array([0, 0, -1]), np.array([0, 1, 0]), "bottom"),    # Bottom (-Z)
            ]

            for direction, up, name in ortho_views:
                position = center + direction * radius
                look_at = center

                intrinsic, extrinsic = self._build_camera_matrices(
                    position, look_at, up
                )

                cameras.append(ViewCamera(
                    position=position,
                    look_at=look_at,
                    up=up,
                    intrinsic=intrinsic,
                    extrinsic=extrinsic,
                    width=self.render_width,
                    height=self.render_height,
                    view_id=view_id
                ))
                view_id += 1

            # === 8 CORNER VIEWS (45° diagonal angles) ===
            # These catch cracks that might only be visible at oblique angles
            sqrt3 = 1.0 / np.sqrt(3)

            corner_views = [
                # Upper corners (elevated 45°)
                (np.array([sqrt3, sqrt3, sqrt3]), np.array([0, 0, 1]), "front-right-top"),
                (np.array([-sqrt3, sqrt3, sqrt3]), np.array([0, 0, 1]), "front-left-top"),
                (np.array([sqrt3, -sqrt3, sqrt3]), np.array([0, 0, 1]), "back-right-top"),
                (np.array([-sqrt3, -sqrt3, sqrt3]), np.array([0, 0, 1]), "back-left-top"),
                # Lower corners (depressed 45°)
                (np.array([sqrt3, sqrt3, -sqrt3]), np.array([0, 0, 1]), "front-right-bottom"),
                (np.array([-sqrt3, sqrt3, -sqrt3]), np.array([0, 0, 1]), "front-left-bottom"),
                (np.array([sqrt3, -sqrt3, -sqrt3]), np.array([0, 0, 1]), "back-right-bottom"),
                (np.array([-sqrt3, -sqrt3, -sqrt3]), np.array([0, 0, 1]), "back-left-bottom"),
            ]

            for direction, up, name in corner_views:
                position = center + direction * radius
                look_at = center

                # Adjust up vector for bottom views to avoid gimbal lock
                forward = look_at - position
                forward = forward / np.linalg.norm(forward)

                # Check if looking nearly straight up/down
                if abs(np.dot(forward, up)) > 0.9:
                    up = np.array([0, 1, 0])

                intrinsic, extrinsic = self._build_camera_matrices(
                    position, look_at, up
                )

                cameras.append(ViewCamera(
                    position=position,
                    look_at=look_at,
                    up=up,
                    intrinsic=intrinsic,
                    extrinsic=extrinsic,
                    width=self.render_width,
                    height=self.render_height,
                    view_id=view_id
                ))
                view_id += 1

            print("  Generated 14 systematic views:")
            print("    - 6 orthographic (front/back/left/right/top/bottom)")
            print("    - 8 corner views (45 degree diagonals)")

        elif view_type == "spherical":
            # Generate spherical viewpoints
            phi_samples = int(np.sqrt(num_views))
            theta_samples = num_views // phi_samples

            view_id = 0
            for i in range(phi_samples):
                phi = np.pi * (i + 0.5) / phi_samples  # Elevation
                for j in range(theta_samples):
                    theta = 2 * np.pi * j / theta_samples  # Azimuth

                    # Camera position on sphere
                    x = center[0] + radius * np.sin(phi) * np.cos(theta)
                    y = center[1] + radius * np.sin(phi) * np.sin(theta)
                    z = center[2] + radius * np.cos(phi)

                    position = np.array([x, y, z])

                    # Look at center
                    look_at = center

                    # Up vector (approximate)
                    forward = look_at - position
                    forward = forward / np.linalg.norm(forward)

                    right = np.cross(forward, np.array([0, 0, 1]))
                    if np.linalg.norm(right) < 0.1:
                        right = np.cross(forward, np.array([0, 1, 0]))
                    right = right / np.linalg.norm(right)

                    up = np.cross(right, forward)
                    up = up / np.linalg.norm(up)

                    # Build camera matrices
                    intrinsic, extrinsic = self._build_camera_matrices(
                        position, look_at, up
                    )

                    cameras.append(ViewCamera(
                        position=position,
                        look_at=look_at,
                        up=up,
                        intrinsic=intrinsic,
                        extrinsic=extrinsic,
                        width=self.render_width,
                        height=self.render_height,
                        view_id=view_id
                    ))
                    view_id += 1

        elif view_type == "orbit":
            # Horizontal orbit around object
            for i in range(num_views):
                theta = 2 * np.pi * i / num_views

                x = center[0] + radius * np.cos(theta)
                y = center[1] + radius * np.sin(theta)
                z = center[2] + extent[2] * 0.3  # Slightly above center

                position = np.array([x, y, z])
                look_at = center
                up = np.array([0, 0, 1])

                intrinsic, extrinsic = self._build_camera_matrices(
                    position, look_at, up
                )

                cameras.append(ViewCamera(
                    position=position,
                    look_at=look_at,
                    up=up,
                    intrinsic=intrinsic,
                    extrinsic=extrinsic,
                    width=self.render_width,
                    height=self.render_height,
                    view_id=i
                ))

        self.cameras = cameras
        return cameras

    def _build_camera_matrices(self, position, look_at, up):
        """Build intrinsic and extrinsic camera matrices"""
        # Intrinsic matrix
        fx = fy = self.render_width * 1.2
        cx = self.render_width / 2
        cy = self.render_height / 2

        intrinsic = np.array([
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1]
        ])

        # Extrinsic matrix (world to camera)
        forward = look_at - position
        forward = forward / np.linalg.norm(forward)

        right = np.cross(forward, up)
        right = right / np.linalg.norm(right)

        up_corrected = np.cross(right, forward)

        # Rotation matrix
        R = np.array([
            right,
            -up_corrected,
            forward
        ])

        # Translation
        t = -R @ position

        extrinsic = np.eye(4)
        extrinsic[:3, :3] = R
        extrinsic[:3, 3] = t

        return intrinsic, extrinsic

    def render_views(self, cameras: list[ViewCamera] = None) -> list[tuple[Path, ViewCamera]]:
        """Render all viewpoints and save as images"""
        if cameras is None:
            cameras = self.cameras

        print(f"\nRendering {len(cameras)} views (mode: {self.render_mode})...")

        if self.render_mode == "surface" and self.mesh is not None:
            rendered_views = self._render_mesh(cameras)
        else:
            rendered_views = self._render_projection(cameras)

        print(f"  Saved {len(rendered_views)} rendered views to {self.views_dir}")
        return rendered_views

    def _render_mesh(self, cameras: list[ViewCamera]) -> list[tuple[Path, ViewCamera]]:
        """Render mesh as solid surfaces using raycasting"""
        import open3d as o3d

        rendered_views = []

        # Create raycasting scene
        print("  Setting up mesh raycasting...")
        mesh = self.mesh

        # Create legacy mesh for raycasting
        scene = o3d.t.geometry.RaycastingScene()
        mesh_t = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
        scene.add_triangles(mesh_t)

        # Get mesh colors for shading
        has_colors = mesh.has_vertex_colors()
        if has_colors:
            vertex_colors = np.asarray(mesh.vertex_colors)
        else:
            vertex_colors = np.ones((len(mesh.vertices), 3)) * 0.7

        triangles = np.asarray(mesh.triangles)

        for camera in tqdm(cameras, desc="Rendering mesh views"):
            # Create image
            img = np.ones((camera.height, camera.width, 3), dtype=np.uint8) * 30

            # Camera parameters
            R = camera.extrinsic[:3, :3]
            t = camera.extrinsic[:3, 3]
            K_inv = np.linalg.inv(camera.intrinsic)

            # Camera center in world coordinates
            camera_center = -R.T @ t

            # Generate rays for each pixel
            rays_list = []
            pixel_coords = []

            # Sample pixels (use stride for speed, can adjust for quality)
            stride = 1  # Set to 2 or 3 for faster but lower quality
            for v in range(0, camera.height, stride):
                for u in range(0, camera.width, stride):
                    # Ray direction in camera coordinates
                    pixel_h = np.array([u, v, 1.0])
                    ray_cam = K_inv @ pixel_h
                    ray_world = R.T @ ray_cam
                    ray_world = ray_world / np.linalg.norm(ray_world)

                    rays_list.append([
                        camera_center[0], camera_center[1], camera_center[2],
                        ray_world[0], ray_world[1], ray_world[2]
                    ])
                    pixel_coords.append((u, v))

            # Cast all rays at once
            rays = o3d.core.Tensor(rays_list, dtype=o3d.core.Dtype.Float32)
            result = scene.cast_rays(rays)

            # Get hit information
            hit = result['t_hit'].numpy()
            triangle_ids = result['primitive_ids'].numpy()
            primitive_uvs = result['primitive_uvs'].numpy()

            # Color pixels based on hits
            for i, (u, v) in enumerate(pixel_coords):
                if hit[i] != np.inf and triangle_ids[i] >= 0:
                    tri_id = triangle_ids[i]
                    if tri_id < len(triangles):
                        # Get triangle vertex indices
                        tri = triangles[tri_id]

                        # Interpolate color using barycentric coordinates
                        uv = primitive_uvs[i]
                        w0 = 1 - uv[0] - uv[1]
                        w1 = uv[0]
                        w2 = uv[1]

                        # Get vertex colors
                        c0 = vertex_colors[tri[0]]
                        c1 = vertex_colors[tri[1]]
                        c2 = vertex_colors[tri[2]]

                        # Interpolate
                        color = w0 * c0 + w1 * c1 + w2 * c2
                        color = np.clip(color * 255, 0, 255).astype(np.uint8)

                        # Fill pixels (if stride > 1, fill block)
                        for dv in range(stride):
                            for du in range(stride):
                                pv, pu = v + dv, u + du
                                if pv < camera.height and pu < camera.width:
                                    img[pv, pu] = color[::-1]  # RGB to BGR

            # Save image
            view_path = self.views_dir / f"view_{camera.view_id:04d}.png"
            cv2.imwrite(str(view_path), img)
            rendered_views.append((view_path, camera))

        return rendered_views

    def _render_projection(self, cameras: list[ViewCamera]) -> list[tuple[Path, ViewCamera]]:
        """Render by projecting 3D points to 2D (Windows compatible)"""
        rendered_views = []
        points = np.asarray(self.point_cloud.points)

        # Get colors if available
        if self.point_cloud.has_colors():
            colors = (np.asarray(self.point_cloud.colors) * 255).astype(np.uint8)
        else:
            colors = np.ones((len(points), 3), dtype=np.uint8) * 128

        for camera in tqdm(cameras, desc="Rendering views"):
            # Create blank image with dark background
            img = np.ones((camera.height, camera.width, 3), dtype=np.uint8) * 30

            # Transform points to camera coordinates
            R = camera.extrinsic[:3, :3]
            t = camera.extrinsic[:3, 3]

            # World to camera transformation
            points_cam = (R @ points.T).T + t

            # Filter points in front of camera (positive Z)
            valid_depth = points_cam[:, 2] > 0.1

            # Project to 2D using intrinsic matrix
            K = camera.intrinsic
            points_2d_h = (K @ points_cam.T).T

            # Avoid division by zero
            z_vals = points_2d_h[:, 2:3]
            z_vals[z_vals == 0] = 0.001
            points_2d = points_2d_h[:, :2] / z_vals

            # Filter points within image bounds
            valid_x = (points_2d[:, 0] >= 0) & (points_2d[:, 0] < camera.width)
            valid_y = (points_2d[:, 1] >= 0) & (points_2d[:, 1] < camera.height)
            valid = valid_depth & valid_x & valid_y

            # Get valid points and colors
            pts_2d_valid = points_2d[valid].astype(np.int32)
            colors_valid = colors[valid]
            depths_valid = points_cam[valid, 2]

            if len(pts_2d_valid) > 0:
                # Sort by depth (far to near) for proper occlusion
                depth_order = np.argsort(-depths_valid)
                pts_2d_valid = pts_2d_valid[depth_order]
                colors_valid = colors_valid[depth_order]

                # Draw points with size based on depth for better visualization
                for (x, y), color in zip(pts_2d_valid, colors_valid):
                    # BGR format for OpenCV
                    cv2.circle(img, (x, y), 2, color.tolist()[::-1], -1)

            # Save image
            view_path = self.views_dir / f"view_{camera.view_id:04d}.png"
            cv2.imwrite(str(view_path), img)
            rendered_views.append((view_path, camera))

        return rendered_views

    def render_depth_maps(self, cameras: list[ViewCamera] = None) -> dict[int, np.ndarray]:
        """Render depth maps for each viewpoint (projection-based, Windows compatible)"""
        if cameras is None:
            cameras = self.cameras

        print("Rendering depth maps...")

        depth_maps = {}
        points = np.asarray(self.point_cloud.points)

        for camera in tqdm(cameras, desc="Rendering depth"):
            # Create depth buffer
            depth_buffer = np.full((camera.height, camera.width), np.inf, dtype=np.float32)

            # Transform points to camera coordinates
            R = camera.extrinsic[:3, :3]
            t = camera.extrinsic[:3, 3]
            points_cam = (R @ points.T).T + t

            # Filter points in front of camera
            valid_depth = points_cam[:, 2] > 0.1

            # Project to 2D
            K = camera.intrinsic
            points_2d_h = (K @ points_cam.T).T
            z_vals = points_2d_h[:, 2:3]
            z_vals[z_vals == 0] = 0.001
            points_2d = points_2d_h[:, :2] / z_vals

            # Filter points within image bounds
            valid_x = (points_2d[:, 0] >= 0) & (points_2d[:, 0] < camera.width)
            valid_y = (points_2d[:, 1] >= 0) & (points_2d[:, 1] < camera.height)
            valid = valid_depth & valid_x & valid_y

            # Get valid points
            pts_2d_valid = points_2d[valid].astype(np.int32)
            depths_valid = points_cam[valid, 2]

            # Fill depth buffer (keep closest depth)
            for (x, y), d in zip(pts_2d_valid, depths_valid):
                if d < depth_buffer[y, x]:
                    depth_buffer[y, x] = d

            # Replace inf with 0 for invalid pixels
            depth_buffer[depth_buffer == np.inf] = 0

            depth_maps[camera.view_id] = depth_buffer

        return depth_maps


class CrackDetector:
    """Detect cracks in rendered views using YOLOv8 or OpenCV"""

    def __init__(self, model_path: str, confidence: float = 0.25, force_cpu: bool = False,
                 detector_type: str = "yolo"):
        self.conf = confidence
        self.device = 'cpu'
        self.detector_type = detector_type.lower()

        # Force OpenCV mode if requested
        if self.detector_type == "opencv":
            print("\n" + "=" * 40)
            print("DETECTOR: OpenCV (edge detection)")
            print("=" * 40)
            self.model = None
            self.use_yolo = False
            return

        print("\n" + "=" * 40)
        print("DETECTOR: YOLOv8")
        print("=" * 40)

        try:
            import torch

            # Check CUDA compatibility
            if torch.cuda.is_available() and not force_cpu:
                try:
                    _ = torch.zeros(1).cuda()
                    del _
                    self.device = 'cuda'
                    print(f"Using CUDA: {torch.cuda.get_device_name(0)}")
                except Exception as cuda_err:
                    print(f"CUDA not compatible: {cuda_err}")
                    print("Using CPU (normal for new GPUs like RTX 5080)")
                    self.device = 'cpu'
            else:
                print("Using CPU for inference")

            from ultralytics import YOLO
            self.model = YOLO(model_path)
            self.use_yolo = True
            print(f"Loaded YOLOv8 model: {model_path}")

        except Exception as e:
            print(f"Could not load YOLOv8: {e}")
            print("Using OpenCV fallback detection")
            self.model = None
            self.use_yolo = False

    def detect(self, image_path: Path, camera: ViewCamera) -> list[CrackDetection2D]:
        """Detect cracks in a single view"""
        image = cv2.imread(str(image_path))
        if image is None:
            return []

        if self.use_yolo and self.model is not None:
            return self._detect_yolo(image, image_path, camera)
        else:
            return self._detect_opencv(image, image_path, camera)

    def _detect_yolo(self, image: np.ndarray, image_path: Path,
                    camera: ViewCamera) -> list[CrackDetection2D]:
        """YOLOv8 detection with segmentation support"""
        detections = []

        results = self.model(image, conf=self.conf, verbose=False, device=self.device)

        for r in results:
            # Check if segmentation masks are available
            has_masks = r.masks is not None and len(r.masks) > 0

            for i, box in enumerate(r.boxes):
                x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                conf = float(box.conf[0])

                width = x2 - x1
                if width > 50:
                    severity = "SEVERE"
                elif width > 25:
                    severity = "MODERATE"
                else:
                    severity = "MINOR"

                # Use segmentation mask if available (YOLOv8-seg model)
                if has_masks and i < len(r.masks):
                    # Get mask polygon points
                    mask_xy = r.masks[i].xy
                    if len(mask_xy) > 0 and len(mask_xy[0]) > 0:
                        # Convert polygon to pixel coordinates
                        polygon = mask_xy[0].astype(int)

                        # Create binary mask from polygon
                        h, w = image.shape[:2]
                        mask = np.zeros((h, w), dtype=np.uint8)
                        cv2.fillPoly(mask, [polygon], 255)

                        # Get all pixels inside the mask (TRUE SEGMENTATION)
                        coords = np.column_stack(np.where(mask > 0))
                        pixel_coords = [(int(c[1]), int(c[0])) for c in coords]

                        # Subsample if too many (but keep more for segmentation)
                        if len(pixel_coords) > 2000:
                            indices = np.random.choice(len(pixel_coords), 2000, replace=False)
                            pixel_coords = [pixel_coords[i] for i in indices]

                        print(f"    Segmentation mask: {len(pixel_coords)} pixels")
                    else:
                        # Fallback to edge detection in bbox
                        roi = image[y1:y2, x1:x2]
                        pixel_coords = self._extract_crack_pixels(roi, x1, y1)
                else:
                    # No segmentation mask - use edge detection in bounding box
                    roi = image[y1:y2, x1:x2]
                    pixel_coords = self._extract_crack_pixels(roi, x1, y1)

                detections.append(CrackDetection2D(
                    view_id=camera.view_id,
                    camera=camera,
                    bbox=(x1, y1, x2, y2),
                    confidence=conf,
                    severity=severity,
                    pixel_coords=pixel_coords
                ))

        return detections

    def _detect_opencv(self, image: np.ndarray, image_path: Path,
                      camera: ViewCamera) -> list[CrackDetection2D]:
        """OpenCV fallback detection"""
        detections = []

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Denoise
        denoised = cv2.fastNlMeansDenoising(gray, h=10)

        # Enhance
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(denoised)

        # Edge detection
        edges = cv2.Canny(enhanced, 30, 100)

        # Morphological cleanup
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        edges = cv2.dilate(edges, kernel, iterations=1)
        edges = cv2.erode(edges, kernel, iterations=1)

        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 50:
                continue

            x, y, w, h = cv2.boundingRect(contour)
            aspect = max(w, h) / (min(w, h) + 1)

            if aspect > 2:  # Crack-like
                if w > 30:
                    severity = "SEVERE"
                elif w > 15:
                    severity = "MODERATE"
                else:
                    severity = "MINOR"

                # Get contour pixels
                mask = np.zeros(gray.shape, dtype=np.uint8)
                cv2.drawContours(mask, [contour], -1, 255, -1)
                coords = np.column_stack(np.where(mask > 0))
                pixel_coords = [(int(c[1]), int(c[0])) for c in coords]

                if len(pixel_coords) > 500:
                    indices = np.random.choice(len(pixel_coords), 500, replace=False)
                    pixel_coords = [pixel_coords[i] for i in indices]

                detections.append(CrackDetection2D(
                    view_id=camera.view_id,
                    camera=camera,
                    bbox=(x, y, x + w, y + h),
                    confidence=0.5,
                    severity=severity,
                    pixel_coords=pixel_coords
                ))

        return detections

    def _extract_crack_pixels(self, roi: np.ndarray, offset_x: int,
                             offset_y: int) -> list:
        """Extract crack pixel coordinates from ROI"""
        if roi.size == 0:
            return []

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        edges = cv2.Canny(enhanced, 50, 150)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        edges = cv2.dilate(edges, kernel, iterations=1)

        coords = np.column_stack(np.where(edges > 0))
        pixel_coords = [(int(c[1] + offset_x), int(c[0] + offset_y)) for c in coords]

        if len(pixel_coords) > 1000:
            indices = np.random.choice(len(pixel_coords), 1000, replace=False)
            pixel_coords = [pixel_coords[i] for i in indices]

        return pixel_coords

    def detect_batch(self, rendered_views: list[tuple[Path, ViewCamera]]) -> list[CrackDetection2D]:
        """Detect cracks in all rendered views"""
        all_detections = []

        for view_path, camera in tqdm(rendered_views, desc="Detecting cracks"):
            detections = self.detect(view_path, camera)
            all_detections.extend(detections)

        return all_detections


class BackProjector:
    """Back-project 2D detections to 3D point cloud"""

    def __init__(self, point_cloud, output_dir: Path,
                 min_points: int = 50, max_scatter_ratio: float = 0.3):
        import open3d as o3d

        self.point_cloud = point_cloud
        self.output_dir = output_dir
        self.points = np.asarray(point_cloud.points)
        self.min_points = min_points  # Minimum points for a valid crack
        self.max_scatter_ratio = max_scatter_ratio  # Max ratio of extent to point count

        # Build KD-tree
        self.kd_tree = o3d.geometry.KDTreeFlann(point_cloud)

    def backproject_detections(self, detections: list[CrackDetection2D],
                              depth_maps: dict = None) -> list[CrackDetection3D]:
        """Back-project all 2D detections to 3D"""
        print("\n" + "=" * 60)
        print("BACK-PROJECTING TO 3D")
        print("=" * 60)

        all_crack_3d = []
        filtered_count = 0

        for det in tqdm(detections, desc="Back-projecting"):
            points_3d, indices = self._backproject_single(det, depth_maps)

            if len(points_3d) >= self.min_points:
                pts_array = np.array(points_3d)

                # Check spatial coherence - filter out scattered false positives
                if self._is_spatially_coherent(pts_array):
                    all_crack_3d.append(CrackDetection3D(
                        points_3d=pts_array,
                        point_indices=indices,
                        severity=det.severity,
                        confidence=det.confidence,
                        source_views=[det.view_id]
                    ))
                else:
                    filtered_count += 1
            else:
                filtered_count += 1

        if filtered_count > 0:
            print(f"  Filtered {filtered_count} scattered/small detections (false positives)")

        # Merge nearby cracks
        merged = self._merge_nearby(all_crack_3d)

        print(f"  Found {len(merged)} unique crack regions")
        return merged

    def _is_spatially_coherent(self, points: np.ndarray) -> bool:
        """
        Check if points form a spatially coherent pattern (like a crack).
        Returns False for scattered points that are likely false positives.

        Real cracks should have:
        - Points that are relatively close together
        - A linear or curved pattern (high aspect ratio bounding box)
        - Not scattered randomly across the surface
        """
        if len(points) < 10:
            return False

        # Calculate bounding box
        min_coords = points.min(axis=0)
        max_coords = points.max(axis=0)
        extent = max_coords - min_coords

        # Calculate average nearest neighbor distance
        from scipy.spatial import cKDTree
        tree = cKDTree(points)
        distances, _ = tree.query(points, k=2)  # k=2 because closest is self
        avg_nn_dist = np.mean(distances[:, 1])

        # Calculate the diagonal of the bounding box
        bbox_diagonal = np.linalg.norm(extent)

        # Scatter ratio: if points are too spread out relative to their count, it's likely noise
        # A real crack should have points packed relatively close together
        point_density = len(points) / (bbox_diagonal + 0.001)

        # Filter 1: If points are too scattered (low density), reject
        # For a coherent crack, we expect higher density
        if point_density < 20:  # Less than 20 points per meter of extent
            return False

        # Filter 2: If average nearest neighbor distance is too large relative to extent
        # Scattered false positives have large gaps between points
        if avg_nn_dist > 0.1:  # If average gap is > 10cm, likely scattered
            return False

        # Filter 3: Check if the extent in all dimensions is reasonable
        # Very large extent with few points = scattered
        max_extent = max(extent)
        if max_extent > 1.0 and len(points) < 200:  # >1m extent needs more points
            return False

        return True

    def _backproject_single(self, detection: CrackDetection2D,
                           depth_maps: dict = None) -> tuple[list, list]:
        """Back-project a single detection"""
        camera = detection.camera
        K_inv = np.linalg.inv(camera.intrinsic)

        # Camera center
        R = camera.extrinsic[:3, :3]
        t = camera.extrinsic[:3, 3]
        camera_center = -R.T @ t

        points_3d = []
        indices = []

        # Use more pixels for denser crack mapping (was 200, now 1000)
        max_pixels = min(len(detection.pixel_coords), 1000)
        for u, v in detection.pixel_coords[:max_pixels]:
            # Use depth if available
            if depth_maps and camera.view_id in depth_maps:
                depth_map = depth_maps[camera.view_id]
                if 0 <= v < depth_map.shape[0] and 0 <= u < depth_map.shape[1]:
                    depth = depth_map[v, u]
                    if depth > 0 and depth < 100:
                        # Unproject with known depth
                        pixel_h = np.array([u, v, 1.0])
                        ray_cam = K_inv @ pixel_h
                        ray_cam = ray_cam / np.linalg.norm(ray_cam)

                        point_cam = ray_cam * depth
                        point_world = R.T @ (point_cam - t)

                        # Find nearest point
                        [k, idx, dist] = self.kd_tree.search_knn_vector_3d(point_world, 1)
                        if k > 0 and dist[0] < 0.5:
                            points_3d.append(self.points[idx[0]])
                            indices.append(idx[0])
                        continue

            # Ray casting fallback
            pixel_h = np.array([u, v, 1.0])
            ray_cam = K_inv @ pixel_h
            ray_world = R.T @ ray_cam
            ray_world = ray_world / np.linalg.norm(ray_world)

            best_point = None
            best_idx = None
            best_dist = float('inf')

            for depth in np.linspace(0.5, 50, 30):
                point_on_ray = camera_center + depth * ray_world

                [k, idx, dist] = self.kd_tree.search_knn_vector_3d(point_on_ray, 1)

                if k > 0 and dist[0] < best_dist and dist[0] < 0.5:
                    best_dist = dist[0]
                    best_point = self.points[idx[0]]
                    best_idx = idx[0]

            if best_point is not None:
                points_3d.append(best_point)
                indices.append(best_idx)

        return points_3d, indices

    def _merge_nearby(self, cracks: list[CrackDetection3D],
                     threshold: float = 0.3) -> list[CrackDetection3D]:
        """Merge nearby crack detections"""
        if len(cracks) < 2:
            return cracks

        merged = []
        used = set()

        for i, c1 in enumerate(cracks):
            if i in used:
                continue

            merged_points = list(c1.points_3d)
            merged_indices = list(c1.point_indices)
            merged_views = list(c1.source_views)
            max_conf = c1.confidence
            max_sev = c1.severity

            for j, c2 in enumerate(cracks[i+1:], i+1):
                if j in used:
                    continue

                if c1.centroid is not None and c2.centroid is not None:
                    dist = np.linalg.norm(c1.centroid - c2.centroid)

                    if dist < threshold:
                        merged_points.extend(c2.points_3d)
                        merged_indices.extend(c2.point_indices)
                        merged_views.extend(c2.source_views)
                        max_conf = max(max_conf, c2.confidence)

                        sev_order = {"SEVERE": 3, "MODERATE": 2, "MINOR": 1}
                        if sev_order.get(c2.severity, 0) > sev_order.get(max_sev, 0):
                            max_sev = c2.severity

                        used.add(j)

            merged.append(CrackDetection3D(
                points_3d=np.array(merged_points),
                point_indices=list(set(merged_indices)),
                severity=max_sev,
                confidence=max_conf,
                source_views=list(set(merged_views))
            ))
            used.add(i)

        return merged


class AnnotatedExporter:
    """Export annotated point cloud with crack markings"""

    def __init__(self, output_dir: Path):
        self.output_dir = output_dir

    def generate_crack_heatmap(self, pcd, cracks: list,
                               max_distance: float = 1.0):
        """
        Generate a heat map based on proximity to cracks.
        Blue = far from cracks, Red = on/near cracks.
        """
        import open3d as o3d
        from scipy.spatial import cKDTree

        print("  Generating crack proximity heat map...")

        points = np.asarray(pcd.points)

        # Collect all crack points
        all_crack_points = []
        for crack in cracks:
            if crack.points_3d is not None and len(crack.points_3d) > 0:
                all_crack_points.extend(crack.points_3d.tolist())

        if len(all_crack_points) == 0:
            print("    No crack points found, using original colors")
            return pcd

        crack_points = np.array(all_crack_points)
        crack_tree = cKDTree(crack_points)
        distances, _ = crack_tree.query(points, k=1)
        normalized = np.clip(distances / max_distance, 0, 1)

        # Generate heat map colors
        colors = self._distance_to_heatmap_color(normalized)

        heatmap_pcd = o3d.geometry.PointCloud()
        heatmap_pcd.points = o3d.utility.Vector3dVector(points)
        heatmap_pcd.colors = o3d.utility.Vector3dVector(colors)

        if pcd.has_normals():
            heatmap_pcd.normals = pcd.normals

        print(f"    Heat map generated: {len(points)} points colored")
        print(f"    Distance range: {distances.min():.3f} to {distances.max():.3f}")

        return heatmap_pcd

    def _distance_to_heatmap_color(self, normalized_distances: np.ndarray) -> np.ndarray:
        """Convert normalized distances to heat map colors (0=Red, 1=Blue)"""
        colors = np.zeros((len(normalized_distances), 3))

        for i, d in enumerate(normalized_distances):
            if d <= 0.0:
                colors[i] = [1.0, 0.0, 0.0]  # Red
            elif d <= 0.15:
                t = d / 0.15
                colors[i] = [1.0, 0.5 * t, 0.0]
            elif d <= 0.3:
                t = (d - 0.15) / 0.15
                colors[i] = [1.0, 0.5 + 0.5 * t, 0.0]
            elif d <= 0.5:
                t = (d - 0.3) / 0.2
                colors[i] = [1.0 - t, 1.0, 0.0]
            elif d <= 0.7:
                t = (d - 0.5) / 0.2
                colors[i] = [0.0, 1.0, t]
            elif d <= 0.85:
                t = (d - 0.7) / 0.15
                colors[i] = [0.0, 1.0 - 0.5 * t, 1.0]
            else:
                t = (d - 0.85) / 0.15
                colors[i] = [0.0, 0.5 - 0.5 * t, 1.0]

        return colors

    def export(self, point_cloud, cracks: list[CrackDetection3D],
              output_name: str = "annotated_pcd",
              heatmap_max_distance: float = 0.5):
        """Export annotated point cloud"""
        import open3d as o3d

        print("\n" + "=" * 60)
        print("EXPORTING RESULTS")
        print("=" * 60)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Color map
        severity_colors = {
            "SEVERE": np.array([1.0, 0.0, 0.0]),
            "MODERATE": np.array([1.0, 0.5, 0.0]),
            "MINOR": np.array([1.0, 1.0, 0.0])
        }

        # Original point cloud
        pcd_original = o3d.geometry.PointCloud(point_cloud)
        orig_path = self.output_dir / f"{output_name}_original_{timestamp}.ply"
        o3d.io.write_point_cloud(str(orig_path), pcd_original)
        print(f"  Saved original: {orig_path}")

        # Create colored version with crack highlights
        pcd_colored = o3d.geometry.PointCloud(point_cloud)

        if pcd_colored.has_colors():
            colors = np.asarray(pcd_colored.colors).copy()
        else:
            colors = np.ones((len(pcd_colored.points), 3)) * 0.5

        # Mark crack points
        for crack in cracks:
            color = severity_colors.get(crack.severity, np.array([1, 0, 1]))
            for idx in crack.point_indices:
                if idx < len(colors):
                    colors[idx] = color

        pcd_colored.colors = o3d.utility.Vector3dVector(colors)

        colored_path = self.output_dir / f"{output_name}_colored_{timestamp}.ply"
        o3d.io.write_point_cloud(str(colored_path), pcd_colored)
        print(f"  Saved colored: {colored_path}")

        # PCD format
        pcd_path = self.output_dir / f"{output_name}_colored_{timestamp}.pcd"
        o3d.io.write_point_cloud(str(pcd_path), pcd_colored)
        print(f"  Saved PCD: {pcd_path}")

        # Crack points only
        if cracks:
            all_crack_points = []
            all_crack_colors = []

            for crack in cracks:
                color = severity_colors.get(crack.severity, np.array([1, 0, 1]))
                for pt in crack.points_3d:
                    all_crack_points.append(pt)
                    all_crack_colors.append(color)

            crack_pcd = o3d.geometry.PointCloud()
            crack_pcd.points = o3d.utility.Vector3dVector(np.array(all_crack_points))
            crack_pcd.colors = o3d.utility.Vector3dVector(np.array(all_crack_colors))

            cracks_path = self.output_dir / f"{output_name}_cracks_only_{timestamp}.ply"
            o3d.io.write_point_cloud(str(cracks_path), crack_pcd)
            print(f"  Saved cracks only: {cracks_path}")

        # Generate and save crack proximity heat map
        if len(cracks) > 0:
            heatmap_pcd = self.generate_crack_heatmap(
                point_cloud, cracks, max_distance=heatmap_max_distance
            )
            heatmap_path = self.output_dir / f"{output_name}_heatmap_{timestamp}.ply"
            o3d.io.write_point_cloud(str(heatmap_path), heatmap_pcd)
            print(f"  Saved heat map: {heatmap_path}")

            heatmap_pcd_path = self.output_dir / f"{output_name}_heatmap_{timestamp}.pcd"
            o3d.io.write_point_cloud(str(heatmap_pcd_path), heatmap_pcd)
            print(f"  Saved heat map PCD: {heatmap_pcd_path}")
        else:
            # No cracks - save "clean" heatmap (all blue)
            print("  NO CRACKS DETECTED - saving clean heat map (all blue)")
            clean_pcd = o3d.geometry.PointCloud(point_cloud)
            clean_colors = np.zeros((len(clean_pcd.points), 3))
            clean_colors[:, 2] = 1.0  # All blue
            clean_pcd.colors = o3d.utility.Vector3dVector(clean_colors)

            heatmap_path = self.output_dir / f"{output_name}_heatmap_CLEAN_{timestamp}.ply"
            o3d.io.write_point_cloud(str(heatmap_path), clean_pcd)
            print(f"  Saved clean heat map: {heatmap_path}")

            heatmap_pcd_path = self.output_dir / f"{output_name}_heatmap_CLEAN_{timestamp}.pcd"
            o3d.io.write_point_cloud(str(heatmap_pcd_path), clean_pcd)
            print(f"  Saved clean heat map PCD: {heatmap_pcd_path}")

        # JSON report
        self._export_report(cracks, timestamp, output_name)

        return colored_path

    def _export_report(self, cracks: list[CrackDetection3D],
                       timestamp: str, output_name: str):
        """Export JSON report"""
        report = {
            "timestamp": timestamp,
            "total_cracks": len(cracks),
            "severity_summary": {
                "SEVERE": sum(1 for c in cracks if c.severity == "SEVERE"),
                "MODERATE": sum(1 for c in cracks if c.severity == "MODERATE"),
                "MINOR": sum(1 for c in cracks if c.severity == "MINOR")
            },
            "cracks": []
        }

        for i, crack in enumerate(cracks):
            report["cracks"].append({
                "id": i + 1,
                "severity": crack.severity,
                "confidence": crack.confidence,
                "num_points": len(crack.points_3d),
                "centroid": crack.centroid.tolist() if crack.centroid is not None else None,
                "source_views": crack.source_views,
                "point_indices": crack.point_indices[:100]  # Limit for readability
            })

        report_path = self.output_dir / f"{output_name}_report_{timestamp}.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)

        print(f"  Saved report: {report_path}")


def visualize(point_cloud, cracks: list[CrackDetection3D],
              heatmap_distance: float = 0.5):
    """Interactive 3D visualization with heat map toggle"""
    import open3d as o3d
    from scipy.spatial import cKDTree

    print("\n" + "=" * 60)
    print("VISUALIZATION")
    print("=" * 60)
    print("Controls:")
    print("  Drag: Rotate")
    print("  Scroll: Zoom")
    print("  H: Toggle crack heat map (Blue=no cracks, Red=crack detected)")
    print("  R: Reset to original colors")
    print("  Q: Quit")
    print("=" * 60)

    # Prepare point cloud
    pcd = o3d.geometry.PointCloud(point_cloud)
    points = np.asarray(pcd.points)

    # Get or create original colors
    if pcd.has_colors():
        original_colors = np.asarray(pcd.colors).copy()
    else:
        # Default gray colors if no colors exist
        original_colors = np.ones((len(points), 3)) * 0.5
        pcd.colors = o3d.utility.Vector3dVector(original_colors)

    # Pre-compute heat map colors
    heatmap_colors = None
    if len(cracks) > 0:
        all_crack_points = []
        for crack in cracks:
            if crack.points_3d is not None and len(crack.points_3d) > 0:
                all_crack_points.extend(crack.points_3d.tolist())

        if len(all_crack_points) > 0:
            crack_points = np.array(all_crack_points)
            crack_tree = cKDTree(crack_points)
            distances, _ = crack_tree.query(points, k=1)
            normalized = np.clip(distances / heatmap_distance, 0, 1)

            heatmap_colors = np.zeros((len(normalized), 3))
            for i, d in enumerate(normalized):
                if d <= 0.0:
                    heatmap_colors[i] = [1.0, 0.0, 0.0]
                elif d <= 0.15:
                    t = d / 0.15
                    heatmap_colors[i] = [1.0, 0.5 * t, 0.0]
                elif d <= 0.3:
                    t = (d - 0.15) / 0.15
                    heatmap_colors[i] = [1.0, 0.5 + 0.5 * t, 0.0]
                elif d <= 0.5:
                    t = (d - 0.3) / 0.2
                    heatmap_colors[i] = [1.0 - t, 1.0, 0.0]
                elif d <= 0.7:
                    t = (d - 0.5) / 0.2
                    heatmap_colors[i] = [0.0, 1.0, t]
                elif d <= 0.85:
                    t = (d - 0.7) / 0.15
                    heatmap_colors[i] = [0.0, 1.0 - 0.5 * t, 1.0]
                else:
                    t = (d - 0.85) / 0.15
                    heatmap_colors[i] = [0.0, 0.5 - 0.5 * t, 1.0]
            print(f"  Cracks detected: {len(cracks)} regions")
    else:
        # No cracks detected - create "clean" heatmap (all blue)
        print("  NO CRACKS DETECTED - Surface is clean!")
        heatmap_colors = np.zeros((len(points), 3))
        heatmap_colors[:, 2] = 1.0  # All blue = no cracks nearby

    # Start with heatmap ON by default so user sees crack status immediately
    show_heatmap = [True]
    pcd.colors = o3d.utility.Vector3dVector(heatmap_colors)
    if len(cracks) > 0:
        print(f"  Starting with HEAT MAP ON - {len(cracks)} crack regions shown in RED")
    else:
        print("  Starting with HEAT MAP ON - ALL BLUE means NO CRACKS DETECTED")

    def toggle_heatmap(vis):
        show_heatmap[0] = not show_heatmap[0]
        if show_heatmap[0]:
            pcd.colors = o3d.utility.Vector3dVector(heatmap_colors)
            if len(cracks) > 0:
                print("Heat map: ON (Red=crack, Blue=clean)")
            else:
                print("Heat map: ON (All Blue = NO CRACKS DETECTED)")
        else:
            pcd.colors = o3d.utility.Vector3dVector(original_colors)
            print("Heat map: OFF (original colors)")
        # Force geometry update and redraw
        vis.update_geometry(pcd)
        vis.poll_events()
        vis.update_renderer()
        return False

    def reset_colors(vis):
        show_heatmap[0] = False
        pcd.colors = o3d.utility.Vector3dVector(original_colors)
        print("Reset to original colors")
        # Force geometry update and redraw
        vis.update_geometry(pcd)
        vis.poll_events()
        vis.update_renderer()
        return False

    # Create visualizer
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(window_name="Crack Detection - Option 2", width=1280, height=720)

    vis.add_geometry(pcd)

    # Add crack markers
    for crack in cracks:
        if crack.centroid is not None:
            sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.03)
            sphere.translate(crack.centroid)
            if crack.severity == "SEVERE":
                sphere.paint_uniform_color([1, 0, 0])
            elif crack.severity == "MODERATE":
                sphere.paint_uniform_color([1, 0.5, 0])
            else:
                sphere.paint_uniform_color([1, 1, 0])
            vis.add_geometry(sphere)

    # Coordinate frame
    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5)
    vis.add_geometry(coord_frame)

    # Register key callbacks
    vis.register_key_callback(ord('H'), toggle_heatmap)
    vis.register_key_callback(ord('R'), reset_colors)

    # Set render options
    opt = vis.get_render_option()
    opt.point_size = 2.0
    opt.background_color = np.array([0.1, 0.1, 0.1])

    vis.run()
    vis.destroy_window()


def main():
    parser = argparse.ArgumentParser(
        description="Point Cloud First - Render Views and Detect Cracks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Build from photos
  python crack_detector_pcd_render.py --photos ./images --model ./best.pt

  # Use existing point cloud
  python crack_detector_pcd_render.py --pcd ./scan.ply --model ./best.pt

  # With visualization
  python crack_detector_pcd_render.py --pcd ./scan.ply --model ./best.pt --visualize
        """
    )

    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--photos", "-p", type=str,
                            help="Path to folder containing photos")
    input_group.add_argument("--pcd", type=str,
                            help="Path to existing point cloud file")

    parser.add_argument("--model", "-m", type=str,
                       default="crack_detector/train/weights/best.pt",
                       help="Path to YOLOv8 model")
    parser.add_argument("--output", "-o", type=str, default="./output_option2",
                       help="Output directory")
    parser.add_argument("--num-views", type=int, default=14,
                       help="Number of views to render (default: 14, ignored for ortho14)")
    parser.add_argument("--view-type", choices=["ortho14", "spherical", "orbit"],
                       default="ortho14",
                       help="View type: ortho14 (6 axis + 8 corners), spherical, or orbit")
    parser.add_argument("--confidence", "-c", type=float, default=0.5,
                       help="Detection confidence threshold (default: 0.5, use higher for fewer false positives)")
    parser.add_argument("--render-width", type=int, default=1280,
                       help="Rendered image width")
    parser.add_argument("--render-height", type=int, default=960,
                       help="Rendered image height")
    parser.add_argument("--render-mode", choices=["surface", "points"], default="surface",
                       help="Render mode: 'surface' (solid mesh) or 'points' (scattered dots)")
    parser.add_argument("--visualize", "-v", action="store_true",
                       help="Show 3D visualization")
    parser.add_argument("--use-depth", action="store_true",
                       help="Use depth maps for back-projection")
    parser.add_argument("--cpu", action="store_true",
                       help="Force CPU mode (use if GPU is not compatible)")
    parser.add_argument("--use-icp", action="store_true", default=True,
                       help="Enable ICP refinement (default: enabled)")
    parser.add_argument("--no-icp", action="store_true",
                       help="Disable ICP refinement")
    parser.add_argument("--icp-type", choices=["point_to_point", "point_to_plane", "colored"],
                       default="colored",
                       help="ICP type: point_to_point, point_to_plane, or colored (default)")
    parser.add_argument("--heatmap-distance", type=float, default=0.5,
                       help="Max distance for heat map gradient (default: 0.5 meters)")
    parser.add_argument("--detector", type=str, choices=["yolo", "opencv"], default="yolo",
                       help="Detection method: 'yolo' (YOLOv8 model) or 'opencv' (edge detection)")
    parser.add_argument("--min-crack-points", type=int, default=50,
                       help="Minimum 3D points required for a valid crack detection (default: 50)")
    parser.add_argument("--max-scatter", type=float, default=0.3,
                       help="Maximum scatter ratio for crack filtering (default: 0.3)")

    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 60)
    print("OPTION 2: POINT CLOUD FIRST - RENDER AND DETECT")
    print("=" * 60)

    # ICP settings
    use_icp = not args.no_icp
    if use_icp:
        print(f"ICP refinement: ENABLED ({args.icp_type})")
    else:
        print("ICP refinement: DISABLED")

    # Detector type
    print(f"Crack detector: {args.detector.upper()}")
    print(f"Render mode: {args.render_mode.upper()} ({'solid mesh - better for crack detection' if args.render_mode == 'surface' else 'scattered points'})")
    print(f"Confidence threshold: {args.confidence}")
    print(f"Min crack points: {args.min_crack_points}")
    print("Spatial coherence filtering: ENABLED")

    # Step 1: Build/load point cloud
    builder = PointCloudBuilder(output_dir, use_icp=use_icp, icp_type=args.icp_type)

    if args.photos:
        photos_dir = Path(args.photos)
        if not photos_dir.exists():
            print(f"Error: Photos directory not found: {photos_dir}")
            sys.exit(1)

        print(f"Photos directory: {photos_dir}")
        success = builder.build_from_photos(photos_dir)
    else:
        pcd_path = Path(args.pcd)
        if not pcd_path.exists():
            print(f"Error: Point cloud not found: {pcd_path}")
            sys.exit(1)

        print(f"Point cloud file: {pcd_path}")
        success = builder.load_existing(pcd_path)

    if not success or builder.point_cloud is None:
        print("Failed to load/build point cloud")
        sys.exit(1)

    # Step 2: Generate viewpoints and render
    print("\n" + "=" * 60)
    print("STEP 2: RENDERING VIEWS")
    print("=" * 60)

    renderer = ViewRenderer(
        builder.point_cloud, output_dir,
        args.render_width, args.render_height,
        render_mode=args.render_mode
    )

    cameras = renderer.generate_viewpoints(args.num_views, args.view_type)
    rendered_views = renderer.render_views(cameras)

    # Optional: render depth maps
    depth_maps = None
    if args.use_depth:
        depth_maps = renderer.render_depth_maps(cameras)

    # Step 3: Detect cracks in rendered views
    print("\n" + "=" * 60)
    print("STEP 3: DETECTING CRACKS")
    print("=" * 60)

    detector = CrackDetector(args.model, args.confidence, force_cpu=args.cpu,
                             detector_type=args.detector)
    detections_2d = detector.detect_batch(rendered_views)

    print(f"\nTotal 2D detections: {len(detections_2d)}")

    # Step 4: Back-project to 3D
    projector = BackProjector(
        builder.point_cloud, output_dir,
        min_points=args.min_crack_points,
        max_scatter_ratio=args.max_scatter
    )
    cracks_3d = projector.backproject_detections(detections_2d, depth_maps)

    # Step 5: Export results
    exporter = AnnotatedExporter(output_dir)
    output_path = exporter.export(
        builder.point_cloud, cracks_3d,
        heatmap_max_distance=args.heatmap_distance
    )

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Detector used: {args.detector.upper()}")
    print(f"Render mode: {args.render_mode} ({'solid mesh' if args.render_mode == 'surface' else 'scattered points'})")
    print(f"View type: {args.view_type} ({len(rendered_views)} views)")
    print(f"Point cloud size: {len(builder.point_cloud.points)} points")
    print(f"Views rendered: {len(rendered_views)}")
    print(f"2D detections: {len(detections_2d)}")
    print(f"3D crack regions: {len(cracks_3d)}")
    print(f"  - Severe: {sum(1 for c in cracks_3d if c.severity == 'SEVERE')}")
    print(f"  - Moderate: {sum(1 for c in cracks_3d if c.severity == 'MODERATE')}")
    print(f"  - Minor: {sum(1 for c in cracks_3d if c.severity == 'MINOR')}")
    print(f"\nOutput saved to: {output_dir}")

    # Visualization
    if args.visualize:
        visualize(builder.point_cloud, cracks_3d, args.heatmap_distance)


if __name__ == "__main__":
    main()

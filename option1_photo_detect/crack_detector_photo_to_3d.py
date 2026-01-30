"""
Option 1: Photo-First Crack Detection with 3D Mapping
======================================================
Workflow:
1. Run YOLOv8 crack detection on original photos
2. Perform photogrammetry reconstruction (SfM)
3. Map 2D crack detections to 3D point cloud coordinates
4. Export annotated point cloud with crack locations

Setup:
    conda env create -f environment.yml
    conda activate crack_detect_photo
    python crack_detector_photo_to_3d.py --photos ./photos --model ./best.pt --output ./output
"""

import argparse
import json
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from tqdm import tqdm


@dataclass
class CrackDetection2D:
    """Stores a 2D crack detection from an image"""
    image_path: str
    bbox: tuple  # (x1, y1, x2, y2)
    confidence: float
    severity: str
    mask_points: Optional[np.ndarray] = None
    pixel_coords: list = field(default_factory=list)  # List of (u, v) crack pixels


@dataclass
class CrackDetection3D:
    """Stores a 3D crack detection mapped to point cloud"""
    points_3d: np.ndarray  # Nx3 array of 3D points
    severity: str
    confidence: float
    source_images: list
    centroid: np.ndarray = None

    def __post_init__(self):
        if self.points_3d is not None and len(self.points_3d) > 0:
            self.centroid = np.mean(self.points_3d, axis=0)


class YOLOCrackDetector:
    """YOLOv8-based crack detection on images"""

    def __init__(self, model_path: str, confidence_threshold: float = 0.25, force_cpu: bool = False):
        self.device = 'cpu'  # Default to CPU

        try:
            import torch

            # Check CUDA availability and compatibility
            if torch.cuda.is_available() and not force_cpu:
                try:
                    # Test if CUDA actually works with this GPU
                    _ = torch.zeros(1).cuda()
                    del _
                    self.device = 'cuda'
                    print(f"Using CUDA: {torch.cuda.get_device_name(0)}")
                except Exception as cuda_err:
                    print(f"CUDA available but not compatible: {cuda_err}")
                    print("Falling back to CPU (this is normal for very new GPUs)")
                    self.device = 'cpu'
            else:
                print("Using CPU for inference")

            from ultralytics import YOLO
            self.model = YOLO(model_path)
            self.conf_threshold = confidence_threshold
            print(f"Loaded YOLOv8 model: {model_path}")

        except Exception as e:
            print(f"Error loading model: {e}")
            print("Falling back to OpenCV-based detection")
            self.model = None

    def detect(self, image_path: str) -> list[CrackDetection2D]:
        """Detect cracks in a single image"""
        detections = []
        image = cv2.imread(image_path)

        if image is None:
            print(f"Warning: Could not read {image_path}")
            return detections

        if self.model is not None:
            # YOLOv8 detection (use CPU if CUDA not compatible)
            results = self.model(image, conf=self.conf_threshold, verbose=False, device=self.device)

            for r in results:
                for i, box in enumerate(r.boxes):
                    x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                    conf = float(box.conf[0])

                    # Determine severity based on bbox size
                    width = x2 - x1
                    if width > 50:
                        severity = "SEVERE"
                    elif width > 25:
                        severity = "MODERATE"
                    else:
                        severity = "MINOR"

                    # Extract crack pixels within bbox using edge detection
                    roi = image[y1:y2, x1:x2]
                    crack_pixels = self._extract_crack_pixels(roi, x1, y1)

                    # Get segmentation mask if available
                    mask_points = None
                    if r.masks is not None and i < len(r.masks):
                        mask_points = r.masks[i].xy[0] if len(r.masks[i].xy) > 0 else None

                    detections.append(CrackDetection2D(
                        image_path=image_path,
                        bbox=(x1, y1, x2, y2),
                        confidence=conf,
                        severity=severity,
                        mask_points=mask_points,
                        pixel_coords=crack_pixels
                    ))
        else:
            # Fallback: OpenCV edge-based detection
            detections = self._opencv_detect(image, image_path)

        return detections

    def _extract_crack_pixels(self, roi: np.ndarray, offset_x: int, offset_y: int) -> list:
        """Extract crack pixel coordinates from ROI using edge detection"""
        if roi.size == 0:
            return []

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

        # Enhance contrast
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)

        # Edge detection
        edges = cv2.Canny(enhanced, 50, 150)

        # Morphological cleanup
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        edges = cv2.dilate(edges, kernel, iterations=1)

        # Get pixel coordinates
        coords = np.column_stack(np.where(edges > 0))

        # Convert to (u, v) format with offset (image coords: x=col, y=row)
        pixel_coords = [(int(c[1] + offset_x), int(c[0] + offset_y)) for c in coords]

        # Subsample if too many points
        if len(pixel_coords) > 1000:
            indices = np.random.choice(len(pixel_coords), 1000, replace=False)
            pixel_coords = [pixel_coords[i] for i in indices]

        return pixel_coords

    def _opencv_detect(self, image: np.ndarray, image_path: str) -> list[CrackDetection2D]:
        """Fallback OpenCV-based crack detection"""
        detections = []
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Denoise and enhance
        denoised = cv2.fastNlMeansDenoising(gray, h=10)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(denoised)

        # Edge detection
        edges = cv2.Canny(enhanced, 30, 100)

        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 50:
                continue

            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = max(w, h) / (min(w, h) + 1)

            if aspect_ratio > 2:  # Crack-like shape
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
                    image_path=image_path,
                    bbox=(x, y, x + w, y + h),
                    confidence=0.5,
                    severity=severity,
                    pixel_coords=pixel_coords
                ))

        return detections

    def detect_batch(self, image_paths: list[str]) -> dict[str, list[CrackDetection2D]]:
        """Detect cracks in multiple images"""
        all_detections = {}

        for path in tqdm(image_paths, desc="Detecting cracks"):
            detections = self.detect(path)
            if detections:
                all_detections[path] = detections

        return all_detections


class ICPRefiner:
    """ICP-based point cloud refinement for improved reconstruction quality"""

    def __init__(self, max_correspondence_distance: float = 0.5,
                 max_iterations: int = 50):
        """
        Args:
            max_correspondence_distance: Initial search distance for correspondences.
                                         Will automatically try larger distances if needed.
            max_iterations: Maximum ICP iterations per scale
        """
        self.max_correspondence_distance = max_correspondence_distance
        self.max_iterations = max_iterations

    def refine_point_to_point(self, source, target):
        """
        Point-to-point ICP alignment.
        Minimizes distances between corresponding points.
        Best for: General alignment, when normals aren't reliable.
        """
        import open3d as o3d

        print("  Applying Point-to-Point ICP refinement...")

        # Check for empty point clouds
        if len(source.points) == 0 or len(target.points) == 0:
            print("    Warning: Empty point cloud, skipping alignment")
            return source, np.eye(4)

        # Initial alignment check
        initial_fitness = self._evaluate_registration(source, target)
        print(f"    Initial fitness: {initial_fitness:.4f}")

        # Try with progressively larger distances
        distances_to_try = [
            self.max_correspondence_distance,
            self.max_correspondence_distance * 10,
            self.max_correspondence_distance * 50,
            self.max_correspondence_distance * 200
        ]

        reg_result = None
        for dist in distances_to_try:
            try:
                reg_result = o3d.pipelines.registration.registration_icp(
                    source, target,
                    dist,
                    np.eye(4),
                    o3d.pipelines.registration.TransformationEstimationPointToPoint(),
                    o3d.pipelines.registration.ICPConvergenceCriteria(
                        max_iteration=self.max_iterations
                    )
                )
                if reg_result.fitness > 0:
                    print(f"    Found correspondences at distance: {dist:.4f}")
                    break
            except RuntimeError:
                continue

        # If still no result, return source unchanged
        if reg_result is None or reg_result.fitness == 0:
            print("    Warning: No correspondences found, returning unaligned cloud")
            return source, np.eye(4)

        source_refined = o3d.geometry.PointCloud(source)
        source_refined.transform(reg_result.transformation)

        print(f"    Final fitness: {reg_result.fitness:.4f}")
        print(f"    RMSE: {reg_result.inlier_rmse:.6f}")

        return source_refined, reg_result.transformation

    def refine_point_to_plane(self, source, target):
        """
        Point-to-plane ICP alignment.
        Minimizes distances from points to the tangent planes.
        Best for: Smooth surfaces, requires good normals.
        """
        import open3d as o3d

        print("  Applying Point-to-Plane ICP refinement...")

        # Ensure both have normals
        if not source.has_normals():
            source.estimate_normals(
                search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
            )
        if not target.has_normals():
            target.estimate_normals(
                search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
            )

        initial_fitness = self._evaluate_registration(source, target)
        print(f"    Initial fitness: {initial_fitness:.4f}")

        # Run Point-to-Plane ICP
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
        Best for: Textured surfaces, photogrammetry data.
        """
        import open3d as o3d

        print("  Applying Colored ICP refinement...")

        # Check for empty point clouds
        if len(source.points) == 0 or len(target.points) == 0:
            print("    Warning: Empty point cloud, skipping alignment")
            return source, np.eye(4)

        # Check for colors
        if not source.has_colors() or not target.has_colors():
            print("    Warning: Missing colors, falling back to Point-to-Point ICP")
            return self.refine_point_to_point(source, target)

        # Ensure normals exist
        if not source.has_normals():
            source.estimate_normals(
                search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
            )
        if not target.has_normals():
            target.estimate_normals(
                search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
            )

        initial_fitness = self._evaluate_registration(source, target)
        print(f"    Initial fitness: {initial_fitness:.4f}")

        # Try with progressively larger correspondence distances if needed
        distances_to_try = [
            self.max_correspondence_distance,
            self.max_correspondence_distance * 5,
            self.max_correspondence_distance * 20,
            self.max_correspondence_distance * 100
        ]

        reg_result = None
        for dist in distances_to_try:
            try:
                reg_result = o3d.pipelines.registration.registration_colored_icp(
                    source, target,
                    dist,
                    np.eye(4),
                    o3d.pipelines.registration.TransformationEstimationForColoredICP(
                        lambda_geometric=lambda_geometric
                    ),
                    o3d.pipelines.registration.ICPConvergenceCriteria(
                        max_iteration=self.max_iterations
                    )
                )
                if reg_result.fitness > 0:
                    print(f"    Found correspondences at distance: {dist:.4f}")
                    break
            except RuntimeError as e:
                if "No correspondences found" in str(e):
                    print(f"    No correspondences at distance {dist:.4f}, trying larger...")
                    continue
                else:
                    raise

        # If still no result, fall back to point-to-point with larger distance
        if reg_result is None or reg_result.fitness == 0:
            print("    Warning: Colored ICP failed, falling back to Point-to-Point ICP")
            return self.refine_point_to_point(source, target)

        source_refined = o3d.geometry.PointCloud(source)
        source_refined.transform(reg_result.transformation)

        print(f"    Final fitness: {reg_result.fitness:.4f}")
        print(f"    RMSE: {reg_result.inlier_rmse:.6f}")

        return source_refined, reg_result.transformation

    def refine_multi_scale(self, source, target, voxel_sizes=[0.05, 0.025, 0.01],
                          icp_type: str = "colored"):
        """
        Multi-scale ICP for robust alignment.
        Starts with coarse alignment and progressively refines.
        """
        import open3d as o3d

        print(f"  Applying Multi-scale {icp_type.upper()} ICP refinement...")

        current_transformation = np.eye(4)
        source_down = source

        for i, voxel_size in enumerate(voxel_sizes):
            print(f"    Scale {i+1}/{len(voxel_sizes)} (voxel size: {voxel_size})")

            # Downsample
            source_down = source.voxel_down_sample(voxel_size)
            target_down = target.voxel_down_sample(voxel_size)

            # Estimate normals
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

            # Apply current transformation
            source_down.transform(current_transformation)

            # Select ICP type
            max_dist = voxel_size * 2

            if icp_type == "colored" and source.has_colors() and target.has_colors():
                reg_result = o3d.pipelines.registration.registration_colored_icp(
                    source_down, target_down, max_dist, np.eye(4),
                    o3d.pipelines.registration.TransformationEstimationForColoredICP(),
                    o3d.pipelines.registration.ICPConvergenceCriteria(
                        max_iteration=self.max_iterations
                    )
                )
            elif icp_type == "point_to_plane":
                reg_result = o3d.pipelines.registration.registration_icp(
                    source_down, target_down, max_dist, np.eye(4),
                    o3d.pipelines.registration.TransformationEstimationPointToPlane(),
                    o3d.pipelines.registration.ICPConvergenceCriteria(
                        max_iteration=self.max_iterations
                    )
                )
            else:
                reg_result = o3d.pipelines.registration.registration_icp(
                    source_down, target_down, max_dist, np.eye(4),
                    o3d.pipelines.registration.TransformationEstimationPointToPoint(),
                    o3d.pipelines.registration.ICPConvergenceCriteria(
                        max_iteration=self.max_iterations
                    )
                )

            # Accumulate transformation
            current_transformation = reg_result.transformation @ current_transformation
            print(f"      Fitness: {reg_result.fitness:.4f}, RMSE: {reg_result.inlier_rmse:.6f}")

        # Apply final transformation to original source
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
        """
        Incrementally merge multiple point clouds using ICP.
        Args:
            point_clouds: List of Open3D point clouds
            icp_type: "point_to_point", "point_to_plane", or "colored"
        """
        import open3d as o3d

        if len(point_clouds) == 0:
            return None

        if len(point_clouds) == 1:
            return point_clouds[0]

        print(f"\n  Merging {len(point_clouds)} point clouds with {icp_type} ICP...")

        # Start with first point cloud as reference
        merged = o3d.geometry.PointCloud(point_clouds[0])

        for i, pcd in enumerate(point_clouds[1:], 1):
            print(f"    Aligning cloud {i+1}/{len(point_clouds)}...")

            # Select refinement method
            if icp_type == "colored":
                aligned, _ = self.refine_colored_icp(pcd, merged)
            elif icp_type == "point_to_plane":
                aligned, _ = self.refine_point_to_plane(pcd, merged)
            else:
                aligned, _ = self.refine_point_to_point(pcd, merged)

            # Merge
            merged += aligned

            # Optional: downsample to prevent memory issues
            if len(merged.points) > 500000:
                merged = merged.voxel_down_sample(0.005)

        # Final cleanup
        merged, _ = merged.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)

        print(f"  Final merged cloud: {len(merged.points)} points")
        return merged


class PhotogrammetryReconstructor:
    """Handles SfM reconstruction using COLMAP or Open3D"""

    def __init__(self, output_dir: Path, use_colmap: bool = True, use_icp: bool = True,
                 icp_type: str = "colored"):
        self.output_dir = output_dir
        self.use_colmap = use_colmap
        self.use_icp = use_icp
        self.icp_type = icp_type
        self.icp_refiner = ICPRefiner() if use_icp else None
        self.cameras = {}
        self.images_data = {}
        self.points3d = None
        self.point_cloud = None

    def reconstruct(self, image_folder: Path) -> bool:
        """Run photogrammetry reconstruction"""
        print("\n" + "=" * 60)
        print("PHOTOGRAMMETRY RECONSTRUCTION")
        print("=" * 60)

        if self.use_colmap and self._check_colmap():
            return self._reconstruct_colmap(image_folder)
        else:
            return self._reconstruct_open3d(image_folder)

    def _check_colmap(self) -> bool:
        """Check if COLMAP is available"""
        try:
            result = subprocess.run(["colmap", "--help"], capture_output=True)
            return result.returncode == 0
        except FileNotFoundError:
            print("COLMAP not found, using Open3D reconstruction")
            return False

    def _reconstruct_colmap(self, image_folder: Path) -> bool:
        """Reconstruct using COLMAP"""
        print("Using COLMAP for reconstruction...")

        database_path = self.output_dir / "database.db"
        sparse_dir = self.output_dir / "sparse"
        dense_dir = self.output_dir / "dense"

        sparse_dir.mkdir(parents=True, exist_ok=True)
        dense_dir.mkdir(parents=True, exist_ok=True)

        try:
            # Feature extraction
            print("  Extracting features...")
            subprocess.run([
                "colmap", "feature_extractor",
                "--database_path", str(database_path),
                "--image_path", str(image_folder),
                "--ImageReader.single_camera", "1",
                "--SiftExtraction.use_gpu", "1"
            ], check=True, capture_output=True)

            # Feature matching
            print("  Matching features...")
            subprocess.run([
                "colmap", "exhaustive_matcher",
                "--database_path", str(database_path),
                "--SiftMatching.use_gpu", "1"
            ], check=True, capture_output=True)

            # Sparse reconstruction
            print("  Running sparse reconstruction...")
            subprocess.run([
                "colmap", "mapper",
                "--database_path", str(database_path),
                "--image_path", str(image_folder),
                "--output_path", str(sparse_dir)
            ], check=True, capture_output=True)

            # Dense reconstruction
            print("  Running dense reconstruction...")
            subprocess.run([
                "colmap", "image_undistorter",
                "--image_path", str(image_folder),
                "--input_path", str(sparse_dir / "0"),
                "--output_path", str(dense_dir),
                "--output_type", "COLMAP"
            ], check=True, capture_output=True)

            subprocess.run([
                "colmap", "patch_match_stereo",
                "--workspace_path", str(dense_dir),
                "--workspace_format", "COLMAP",
                "--PatchMatchStereo.geom_consistency", "true"
            ], check=True, capture_output=True)

            subprocess.run([
                "colmap", "stereo_fusion",
                "--workspace_path", str(dense_dir),
                "--workspace_format", "COLMAP",
                "--input_type", "geometric",
                "--output_path", str(dense_dir / "fused.ply")
            ], check=True, capture_output=True)

            # Load results
            self._load_colmap_model(sparse_dir / "0")
            self._load_point_cloud(dense_dir / "fused.ply")

            print("  Reconstruction complete!")
            return True

        except subprocess.CalledProcessError as e:
            print(f"COLMAP error: {e}")
            print("Falling back to Open3D...")
            return self._reconstruct_open3d(image_folder)
        except Exception as e:
            print(f"Error: {e}")
            return False

    def _reconstruct_open3d(self, image_folder: Path) -> bool:
        """Reconstruct using Open3D"""
        import open3d as o3d

        print("Using Open3D for reconstruction...")

        image_paths = sorted(list(image_folder.glob("*.jpg")) +
                            list(image_folder.glob("*.jpeg")) +
                            list(image_folder.glob("*.png")))

        if len(image_paths) < 3:
            print("Error: Need at least 3 images for reconstruction")
            return False

        print(f"  Processing {len(image_paths)} images...")

        # Read first image to get dimensions
        first_img = cv2.imread(str(image_paths[0]))
        height, width = first_img.shape[:2]

        # Estimate camera intrinsics (assuming typical camera)
        focal_length = max(width, height) * 1.2
        cx, cy = width / 2, height / 2

        intrinsic = o3d.camera.PinholeCameraIntrinsic(
            width, height, focal_length, focal_length, cx, cy
        )

        # Feature matching and pose estimation
        print("  Extracting and matching features...")

        # Use ORB features for matching
        orb = cv2.ORB_create(nfeatures=5000)
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

        all_keypoints = []
        all_descriptors = []

        for img_path in tqdm(image_paths, desc="  Feature extraction"):
            img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            kp, desc = orb.detectAndCompute(img, None)
            all_keypoints.append(kp)
            all_descriptors.append(desc)

        # Build point cloud through triangulation
        print("  Triangulating points...")

        points_3d = []
        colors = []
        partial_point_clouds = []  # For ICP merging

        # Simple stereo reconstruction between consecutive image pairs
        for i in tqdm(range(len(image_paths) - 1), desc="  Stereo matching"):
            if all_descriptors[i] is None or all_descriptors[i+1] is None:
                continue

            matches = bf.match(all_descriptors[i], all_descriptors[i+1])
            matches = sorted(matches, key=lambda x: x.distance)[:500]

            if len(matches) < 20:
                continue

            pts1 = np.float32([all_keypoints[i][m.queryIdx].pt for m in matches])
            pts2 = np.float32([all_keypoints[i+1][m.trainIdx].pt for m in matches])

            # Estimate essential matrix
            E, mask = cv2.findEssentialMat(pts1, pts2, focal_length, (cx, cy))
            if E is None:
                continue

            _, R, t, mask = cv2.recoverPose(E, pts1, pts2, focal=focal_length, pp=(cx, cy))

            # Create projection matrices
            P1 = np.hstack([np.eye(3), np.zeros((3, 1))])
            P2 = np.hstack([R, t])

            # Camera matrices
            K = np.array([[focal_length, 0, cx],
                         [0, focal_length, cy],
                         [0, 0, 1]])

            P1 = K @ P1
            P2 = K @ P2

            # Triangulate points
            pts1_h = pts1[mask.ravel() == 1].T
            pts2_h = pts2[mask.ravel() == 1].T

            if pts1_h.shape[1] < 5:
                continue

            points_4d = cv2.triangulatePoints(P1, P2, pts1_h, pts2_h)
            points_4d /= points_4d[3]

            # Filter valid points
            valid = (points_4d[2] > 0) & (points_4d[2] < 100)
            pts_3d = points_4d[:3, valid].T

            # Get colors from first image
            pair_colors = []
            img_color = cv2.imread(str(image_paths[i]))
            for j, pt2d in enumerate(pts1_h.T):
                if valid[j]:
                    x, y = int(pt2d[0]), int(pt2d[1])
                    if 0 <= x < width and 0 <= y < height:
                        color = img_color[y, x] / 255.0
                        colors.append(color[::-1])  # BGR to RGB
                        pair_colors.append(color[::-1])

            points_3d.extend(pts_3d.tolist())

            # Store partial point cloud for ICP merging
            if self.use_icp and len(pts_3d) > 10:
                partial_pcd = o3d.geometry.PointCloud()
                partial_pcd.points = o3d.utility.Vector3dVector(pts_3d.astype(np.float64))
                if pair_colors:
                    partial_pcd.colors = o3d.utility.Vector3dVector(
                        np.array(pair_colors[:len(pts_3d)]).astype(np.float64)
                    )
                partial_point_clouds.append(partial_pcd)

            # Store camera data for mapping
            img_name = image_paths[i].name
            self.images_data[str(image_paths[i])] = {
                'R': R,
                't': t,
                'K': K,
                'width': width,
                'height': height
            }

        if len(points_3d) == 0:
            print("ERROR: No 3D points could be reconstructed!")
            print("Possible causes:")
            print("  - Images have insufficient overlap")
            print("  - Images are too different (lighting, angle)")
            print("  - Not enough features detected")
            print("\nTrying alternative SIFT-based reconstruction...")
            return self._reconstruct_sift(image_paths, width, height)

        if len(points_3d) < 100:
            print("Warning: Few points reconstructed. Results may be sparse.")

        points_3d = np.array(points_3d)

        # Ensure points_3d has correct shape
        if points_3d.ndim == 1:
            points_3d = points_3d.reshape(-1, 3)

        if len(points_3d) == 0 or points_3d.shape[1] != 3:
            print("ERROR: Invalid point cloud shape")
            return False

        colors = np.array(colors) if colors else np.ones((len(points_3d), 3)) * 0.5

        # Ensure colors array matches points
        if len(colors) < len(points_3d):
            colors = np.vstack([colors, np.ones((len(points_3d) - len(colors), 3)) * 0.5])
        elif len(colors) > len(points_3d):
            colors = colors[:len(points_3d)]

        # Create Open3D point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points_3d.astype(np.float64))
        pcd.colors = o3d.utility.Vector3dVector(colors[:len(points_3d)].astype(np.float64))

        # Statistical outlier removal
        pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)

        # Apply ICP refinement if enabled and we have partial clouds
        if self.use_icp and len(partial_point_clouds) > 1:
            print("\n  Applying ICP refinement...")
            pcd = self.icp_refiner.merge_with_icp(
                partial_point_clouds, icp_type=self.icp_type
            )

        # Save point cloud
        output_path = self.output_dir / "reconstructed.ply"
        o3d.io.write_point_cloud(str(output_path), pcd)

        self.point_cloud = pcd
        self.points3d = np.asarray(pcd.points)

        print(f"  Reconstruction complete: {len(self.points3d)} points")
        print(f"  Saved to: {output_path}")

        return True

    def _reconstruct_sift(self, image_paths: list, width: int, height: int) -> bool:
        """Fallback reconstruction using SIFT features (more robust)"""
        import open3d as o3d

        print("  Using SIFT features for better matching...")

        # SIFT tends to work better than ORB for SfM
        sift = cv2.SIFT_create(nfeatures=3000)
        bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)

        focal_length = max(width, height) * 1.2
        cx, cy = width / 2, height / 2
        K = np.array([[focal_length, 0, cx],
                     [0, focal_length, cy],
                     [0, 0, 1]])

        all_points_3d = []
        all_colors = []
        partial_point_clouds = []  # For ICP merging

        # Process image pairs with more overlap options
        print(f"  Processing {len(image_paths)} images...")

        for i in tqdm(range(len(image_paths)), desc="  SIFT reconstruction"):
            img1 = cv2.imread(str(image_paths[i]), cv2.IMREAD_GRAYSCALE)
            img1_color = cv2.imread(str(image_paths[i]))

            if img1 is None:
                continue

            kp1, desc1 = sift.detectAndCompute(img1, None)

            if desc1 is None or len(kp1) < 50:
                continue

            # Match with multiple subsequent images for better coverage
            for j in range(i + 1, min(i + 4, len(image_paths))):
                img2 = cv2.imread(str(image_paths[j]), cv2.IMREAD_GRAYSCALE)
                if img2 is None:
                    continue

                kp2, desc2 = sift.detectAndCompute(img2, None)

                if desc2 is None or len(kp2) < 50:
                    continue

                # Use ratio test for better matches
                matches = bf.knnMatch(desc1, desc2, k=2)

                good_matches = []
                for m, n in matches:
                    if m.distance < 0.75 * n.distance:
                        good_matches.append(m)

                if len(good_matches) < 15:
                    continue

                pts1 = np.float32([kp1[m.queryIdx].pt for m in good_matches])
                pts2 = np.float32([kp2[m.trainIdx].pt for m in good_matches])

                # Find fundamental matrix with RANSAC
                F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC, 3.0)
                if F is None:
                    continue

                # Filter inliers
                mask = mask.ravel().astype(bool)
                pts1_inliers = pts1[mask]
                pts2_inliers = pts2[mask]

                if len(pts1_inliers) < 10:
                    continue

                # Essential matrix from fundamental
                E = K.T @ F @ K

                # Recover pose
                _, R, t, _ = cv2.recoverPose(E, pts1_inliers, pts2_inliers, K)

                # Projection matrices
                P1 = K @ np.hstack([np.eye(3), np.zeros((3, 1))])
                P2 = K @ np.hstack([R, t])

                # Triangulate
                pts1_h = pts1_inliers.T
                pts2_h = pts2_inliers.T

                points_4d = cv2.triangulatePoints(P1, P2, pts1_h, pts2_h)
                points_4d /= points_4d[3]

                # Filter valid points (positive depth, reasonable range)
                valid = (points_4d[2] > 0.1) & (points_4d[2] < 500) & \
                        (np.abs(points_4d[0]) < 500) & (np.abs(points_4d[1]) < 500)

                pts_3d = points_4d[:3, valid].T

                # Get colors
                pair_colors = []
                for k, pt in enumerate(pts1_h.T):
                    if valid[k]:
                        x, y = int(pt[0]), int(pt[1])
                        if 0 <= x < width and 0 <= y < height:
                            color = img1_color[y, x] / 255.0
                            all_colors.append(color[::-1])
                            pair_colors.append(color[::-1])

                all_points_3d.extend(pts_3d.tolist())

                # Store partial point cloud for ICP merging
                if self.use_icp and len(pts_3d) > 10:
                    partial_pcd = o3d.geometry.PointCloud()
                    partial_pcd.points = o3d.utility.Vector3dVector(
                        pts_3d.astype(np.float64)
                    )
                    if pair_colors:
                        partial_pcd.colors = o3d.utility.Vector3dVector(
                            np.array(pair_colors[:len(pts_3d)]).astype(np.float64)
                        )
                    partial_point_clouds.append(partial_pcd)

            # Store camera data
            self.images_data[str(image_paths[i])] = {
                'R': np.eye(3),
                't': np.zeros(3),
                'K': K,
                'width': width,
                'height': height
            }

        if len(all_points_3d) == 0:
            print("ERROR: SIFT reconstruction also failed.")
            print("Please ensure your images have sufficient overlap and features.")
            return False

        print(f"  Reconstructed {len(all_points_3d)} points")

        points_3d = np.array(all_points_3d)
        colors = np.array(all_colors) if all_colors else np.ones((len(points_3d), 3)) * 0.5

        if len(colors) < len(points_3d):
            colors = np.vstack([colors, np.ones((len(points_3d) - len(colors), 3)) * 0.5])

        # Create point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points_3d.astype(np.float64))
        pcd.colors = o3d.utility.Vector3dVector(colors[:len(points_3d)].astype(np.float64))

        # Remove outliers
        if len(pcd.points) > 20:
            pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)

        # Apply ICP refinement if enabled and we have partial clouds
        if self.use_icp and len(partial_point_clouds) > 1:
            print("\n  Applying ICP refinement...")
            pcd = self.icp_refiner.merge_with_icp(
                partial_point_clouds, icp_type=self.icp_type
            )

        # Save
        output_path = self.output_dir / "reconstructed.ply"
        o3d.io.write_point_cloud(str(output_path), pcd)

        self.point_cloud = pcd
        self.points3d = np.asarray(pcd.points)

        print(f"  SIFT reconstruction complete: {len(self.points3d)} points")
        print(f"  Saved to: {output_path}")

        return True

    def _load_colmap_model(self, model_path: Path):
        """Load COLMAP sparse model data"""
        try:
            import pycolmap
            reconstruction = pycolmap.Reconstruction(str(model_path))

            for image_id, image in reconstruction.images.items():
                self.images_data[image.name] = {
                    'R': image.rotmat(),
                    't': image.tvec,
                    'camera_id': image.camera_id
                }

            for camera_id, camera in reconstruction.cameras.items():
                self.cameras[camera_id] = {
                    'model': camera.model_name,
                    'width': camera.width,
                    'height': camera.height,
                    'params': camera.params
                }

        except ImportError:
            print("pycolmap not available, loading from text files...")
            self._load_colmap_text(model_path)

    def _load_colmap_text(self, model_path: Path):
        """Load COLMAP model from text files"""
        # Load cameras
        cameras_file = model_path / "cameras.txt"
        if cameras_file.exists():
            with open(cameras_file) as f:
                for line in f:
                    if line.startswith("#"):
                        continue
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        cam_id = int(parts[0])
                        self.cameras[cam_id] = {
                            'model': parts[1],
                            'width': int(parts[2]),
                            'height': int(parts[3]),
                            'params': [float(p) for p in parts[4:]]
                        }

        # Load images
        images_file = model_path / "images.txt"
        if images_file.exists():
            with open(images_file) as f:
                lines = [l for l in f if not l.startswith("#")]
                for i in range(0, len(lines), 2):
                    parts = lines[i].strip().split()
                    if len(parts) >= 10:
                        qw, qx, qy, qz = map(float, parts[1:5])
                        tx, ty, tz = map(float, parts[5:8])
                        camera_id = int(parts[8])
                        name = parts[9]

                        # Quaternion to rotation matrix
                        R = self._quat_to_rot(qw, qx, qy, qz)

                        self.images_data[name] = {
                            'R': R,
                            't': np.array([tx, ty, tz]),
                            'camera_id': camera_id
                        }

    def _quat_to_rot(self, qw, qx, qy, qz):
        """Convert quaternion to rotation matrix"""
        R = np.array([
            [1 - 2*qy*qy - 2*qz*qz, 2*qx*qy - 2*qz*qw, 2*qx*qz + 2*qy*qw],
            [2*qx*qy + 2*qz*qw, 1 - 2*qx*qx - 2*qz*qz, 2*qy*qz - 2*qx*qw],
            [2*qx*qz - 2*qy*qw, 2*qy*qz + 2*qx*qw, 1 - 2*qx*qx - 2*qy*qy]
        ])
        return R

    def _load_point_cloud(self, ply_path: Path):
        """Load point cloud from PLY file"""
        import open3d as o3d

        if ply_path.exists():
            self.point_cloud = o3d.io.read_point_cloud(str(ply_path))
            self.points3d = np.asarray(self.point_cloud.points)
            print(f"  Loaded point cloud: {len(self.points3d)} points")

    def get_camera_params(self, image_name: str):
        """Get camera parameters for an image"""
        if image_name in self.images_data:
            img_data = self.images_data[image_name]

            if 'K' in img_data:
                return img_data['R'], img_data['t'], img_data['K']

            if 'camera_id' in img_data and img_data['camera_id'] in self.cameras:
                cam = self.cameras[img_data['camera_id']]
                params = cam['params']

                # Build K matrix based on camera model
                if cam['model'] in ['PINHOLE', 'SIMPLE_PINHOLE']:
                    fx = params[0]
                    fy = params[1] if len(params) > 1 else fx
                    cx = params[2] if len(params) > 2 else cam['width'] / 2
                    cy = params[3] if len(params) > 3 else cam['height'] / 2
                else:
                    fx = fy = params[0]
                    cx, cy = cam['width'] / 2, cam['height'] / 2

                K = np.array([[fx, 0, cx],
                             [0, fy, cy],
                             [0, 0, 1]])

                return img_data['R'], img_data['t'], K

        return None, None, None


class CrackMapper3D:
    """Maps 2D crack detections to 3D point cloud coordinates"""

    def __init__(self, reconstructor: PhotogrammetryReconstructor):
        self.reconstructor = reconstructor
        self.crack_detections_3d = []

    def map_detections(self, detections_2d: dict[str, list[CrackDetection2D]]) -> list[CrackDetection3D]:
        """Map all 2D detections to 3D"""
        import open3d as o3d

        print("\n" + "=" * 60)
        print("MAPPING DETECTIONS TO 3D")
        print("=" * 60)

        if self.reconstructor.points3d is None or len(self.reconstructor.points3d) == 0:
            print("Error: No 3D points available")
            return []

        # Build KD-tree for efficient nearest neighbor search
        pcd = self.reconstructor.point_cloud
        if pcd is None:
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(self.reconstructor.points3d)

        kd_tree = o3d.geometry.KDTreeFlann(pcd)

        all_crack_points_3d = []

        for image_path, dets in tqdm(detections_2d.items(), desc="Mapping to 3D"):
            image_name = Path(image_path).name
            R, t, K = self.reconstructor.get_camera_params(image_name)

            if R is None:
                # Try with full path
                R, t, K = self.reconstructor.get_camera_params(image_path)

            if R is None:
                print(f"  Warning: No camera params for {image_name}")
                continue

            for det in dets:
                points_3d = self._project_to_3d(det, R, t, K, kd_tree, pcd)

                if len(points_3d) > 0:
                    crack_3d = CrackDetection3D(
                        points_3d=np.array(points_3d),
                        severity=det.severity,
                        confidence=det.confidence,
                        source_images=[image_path]
                    )
                    all_crack_points_3d.append(crack_3d)

        # Merge nearby crack detections
        merged_cracks = self._merge_nearby_cracks(all_crack_points_3d)

        self.crack_detections_3d = merged_cracks
        print(f"  Mapped {len(merged_cracks)} unique crack regions to 3D")

        return merged_cracks

    def _project_to_3d(self, detection: CrackDetection2D, R, t, K, kd_tree, pcd) -> list:
        """Project 2D crack pixels to 3D using ray casting"""
        points_3d = []
        points_array = np.asarray(pcd.points)

        # Camera center in world coordinates
        camera_center = -R.T @ t

        # Get K inverse for unprojection
        K_inv = np.linalg.inv(K)

        for u, v in detection.pixel_coords[:200]:  # Limit points for efficiency
            # Unproject to ray direction
            pixel_h = np.array([u, v, 1.0])
            ray_cam = K_inv @ pixel_h
            ray_world = R.T @ ray_cam
            ray_world = ray_world / np.linalg.norm(ray_world)

            # Find closest point along ray
            # Sample points along ray and find nearest neighbor
            best_point = None
            best_dist = float('inf')

            for depth in np.linspace(0.5, 50, 50):
                point_on_ray = camera_center.flatten() + depth * ray_world

                # Query nearest neighbor
                [k, idx, dist] = kd_tree.search_knn_vector_3d(point_on_ray, 1)

                if k > 0 and dist[0] < best_dist and dist[0] < 0.5:
                    best_dist = dist[0]
                    best_point = points_array[idx[0]]

            if best_point is not None:
                points_3d.append(best_point)

        return points_3d

    def _merge_nearby_cracks(self, cracks: list[CrackDetection3D],
                             distance_threshold: float = 0.3) -> list[CrackDetection3D]:
        """Merge crack detections that are close together"""
        if len(cracks) < 2:
            return cracks

        merged = []
        used = set()

        for i, crack1 in enumerate(cracks):
            if i in used:
                continue

            merged_points = list(crack1.points_3d)
            merged_sources = list(crack1.source_images)
            max_confidence = crack1.confidence
            max_severity = crack1.severity

            for j, crack2 in enumerate(cracks[i+1:], i+1):
                if j in used:
                    continue

                # Check if centroids are close
                if crack1.centroid is not None and crack2.centroid is not None:
                    dist = np.linalg.norm(crack1.centroid - crack2.centroid)

                    if dist < distance_threshold:
                        merged_points.extend(crack2.points_3d)
                        merged_sources.extend(crack2.source_images)
                        max_confidence = max(max_confidence, crack2.confidence)

                        # Keep highest severity
                        severity_order = {"SEVERE": 3, "MODERATE": 2, "MINOR": 1}
                        if severity_order.get(crack2.severity, 0) > severity_order.get(max_severity, 0):
                            max_severity = crack2.severity

                        used.add(j)

            merged.append(CrackDetection3D(
                points_3d=np.array(merged_points),
                severity=max_severity,
                confidence=max_confidence,
                source_images=list(set(merged_sources))
            ))
            used.add(i)

        return merged


class PointCloudExporter:
    """Export annotated point cloud with crack markings"""

    def __init__(self, output_dir: Path):
        self.output_dir = output_dir

    def generate_crack_heatmap(self, pcd, crack_detections: list,
                               max_distance: float = 1.0):
        """
        Generate a heat map based on proximity to cracks.
        Blue = far from cracks, Red = on/near cracks.

        Args:
            pcd: Open3D point cloud
            crack_detections: List of CrackDetection3D objects
            max_distance: Maximum distance for color gradient (beyond = blue)

        Returns:
            Open3D point cloud with heat map colors
        """
        import open3d as o3d
        from scipy.spatial import cKDTree

        print("  Generating crack proximity heat map...")

        points = np.asarray(pcd.points)

        # Collect all crack points
        all_crack_points = []
        for crack in crack_detections:
            if crack.points_3d is not None and len(crack.points_3d) > 0:
                all_crack_points.extend(crack.points_3d.tolist())

        if len(all_crack_points) == 0:
            print("    No crack points found, using original colors")
            return pcd

        crack_points = np.array(all_crack_points)

        # Build KD-tree for crack points
        crack_tree = cKDTree(crack_points)

        # Query distances from each point to nearest crack
        distances, _ = crack_tree.query(points, k=1)

        # Normalize distances (0 = on crack, 1 = max_distance or beyond)
        normalized = np.clip(distances / max_distance, 0, 1)

        # Generate heat map colors (Blue -> Cyan -> Green -> Yellow -> Orange -> Red)
        # Using a more detailed color gradient for better visualization
        colors = self._distance_to_heatmap_color(normalized)

        # Create heat map point cloud
        heatmap_pcd = o3d.geometry.PointCloud()
        heatmap_pcd.points = o3d.utility.Vector3dVector(points)
        heatmap_pcd.colors = o3d.utility.Vector3dVector(colors)

        if pcd.has_normals():
            heatmap_pcd.normals = pcd.normals

        print(f"    Heat map generated: {len(points)} points colored")
        print(f"    Distance range: {distances.min():.3f} to {distances.max():.3f}")

        return heatmap_pcd

    def _distance_to_heatmap_color(self, normalized_distances: np.ndarray) -> np.ndarray:
        """
        Convert normalized distances to heat map colors.
        0 (on crack) = Red
        0.25 = Orange
        0.5 = Yellow
        0.75 = Cyan
        1.0 (far) = Blue
        """
        colors = np.zeros((len(normalized_distances), 3))

        for i, d in enumerate(normalized_distances):
            if d <= 0.0:
                # On crack - bright red
                colors[i] = [1.0, 0.0, 0.0]
            elif d <= 0.15:
                # Near crack - red to orange
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
                # Light blue to deep blue
                t = (d - 0.85) / 0.15
                colors[i] = [0.0, 0.5 - 0.5 * t, 1.0]

        return colors

    def export(self, reconstructor: PhotogrammetryReconstructor,
               crack_detections: list[CrackDetection3D],
               output_name: str = "annotated_cracks",
               heatmap_max_distance: float = 0.5):
        """Export point cloud with crack annotations"""
        import open3d as o3d

        print("\n" + "=" * 60)
        print("EXPORTING ANNOTATED POINT CLOUD")
        print("=" * 60)

        if reconstructor.point_cloud is None:
            print("Error: No point cloud to export")
            return

        # Create copy of original point cloud
        pcd = o3d.geometry.PointCloud(reconstructor.point_cloud)

        # Color mapping for severity
        severity_colors = {
            "SEVERE": np.array([1.0, 0.0, 0.0]),     # Red
            "MODERATE": np.array([1.0, 0.5, 0.0]),   # Orange
            "MINOR": np.array([1.0, 1.0, 0.0])       # Yellow
        }

        # Create crack point cloud
        crack_pcd = o3d.geometry.PointCloud()
        all_crack_points = []
        all_crack_colors = []

        for crack in crack_detections:
            color = severity_colors.get(crack.severity, np.array([1.0, 0.0, 1.0]))
            for point in crack.points_3d:
                all_crack_points.append(point)
                all_crack_colors.append(color)

        if all_crack_points:
            crack_pcd.points = o3d.utility.Vector3dVector(np.array(all_crack_points))
            crack_pcd.colors = o3d.utility.Vector3dVector(np.array(all_crack_colors))

        # Save separate files
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Original point cloud
        original_path = self.output_dir / f"{output_name}_original_{timestamp}.ply"
        o3d.io.write_point_cloud(str(original_path), pcd)
        print(f"  Saved original: {original_path}")

        # Crack points only
        if len(all_crack_points) > 0:
            cracks_path = self.output_dir / f"{output_name}_cracks_only_{timestamp}.ply"
            o3d.io.write_point_cloud(str(cracks_path), crack_pcd)
            print(f"  Saved cracks: {cracks_path}")

        # Combined with cracks highlighted
        combined = pcd + crack_pcd
        combined_path = self.output_dir / f"{output_name}_combined_{timestamp}.ply"
        o3d.io.write_point_cloud(str(combined_path), combined)
        print(f"  Saved combined: {combined_path}")

        # Export as PCD format as well
        pcd_path = self.output_dir / f"{output_name}_combined_{timestamp}.pcd"
        o3d.io.write_point_cloud(str(pcd_path), combined)
        print(f"  Saved PCD: {pcd_path}")

        # Generate and save crack proximity heat map
        if len(crack_detections) > 0:
            heatmap_pcd = self.generate_crack_heatmap(
                pcd, crack_detections, max_distance=heatmap_max_distance
            )
            heatmap_path = self.output_dir / f"{output_name}_heatmap_{timestamp}.ply"
            o3d.io.write_point_cloud(str(heatmap_path), heatmap_pcd)
            print(f"  Saved heat map: {heatmap_path}")

            # Also save as PCD
            heatmap_pcd_path = self.output_dir / f"{output_name}_heatmap_{timestamp}.pcd"
            o3d.io.write_point_cloud(str(heatmap_pcd_path), heatmap_pcd)
            print(f"  Saved heat map PCD: {heatmap_pcd_path}")

        # Generate JSON report
        self._export_report(crack_detections, timestamp)

        return combined_path

    def _export_report(self, crack_detections: list[CrackDetection3D], timestamp: str):
        """Export JSON report of crack detections"""
        report = {
            "timestamp": timestamp,
            "total_cracks": len(crack_detections),
            "severity_summary": {
                "SEVERE": sum(1 for c in crack_detections if c.severity == "SEVERE"),
                "MODERATE": sum(1 for c in crack_detections if c.severity == "MODERATE"),
                "MINOR": sum(1 for c in crack_detections if c.severity == "MINOR")
            },
            "cracks": []
        }

        for i, crack in enumerate(crack_detections):
            crack_data = {
                "id": i + 1,
                "severity": crack.severity,
                "confidence": crack.confidence,
                "num_points": len(crack.points_3d),
                "centroid": crack.centroid.tolist() if crack.centroid is not None else None,
                "source_images": crack.source_images,
                "bounding_box": {
                    "min": crack.points_3d.min(axis=0).tolist(),
                    "max": crack.points_3d.max(axis=0).tolist()
                } if len(crack.points_3d) > 0 else None
            }
            report["cracks"].append(crack_data)

        report_path = self.output_dir / f"crack_report_{timestamp}.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)

        print(f"  Saved report: {report_path}")


def visualize_results(reconstructor: PhotogrammetryReconstructor,
                     crack_detections: list[CrackDetection3D],
                     heatmap_distance: float = 0.5):
    """Interactive visualization of results with heat map toggle"""
    import open3d as o3d
    from scipy.spatial import cKDTree

    print("\n" + "=" * 60)
    print("VISUALIZATION")
    print("=" * 60)
    print("Controls:")
    print("  Mouse drag: Rotate")
    print("  Scroll: Zoom")
    print("  H: Toggle crack heat map (Blue=far, Red=near crack)")
    print("  R: Reset to original colors")
    print("  Q: Quit")
    print("=" * 60)

    if reconstructor.point_cloud is None:
        print("No point cloud to visualize")
        return

    # Prepare data for visualization
    pcd = o3d.geometry.PointCloud(reconstructor.point_cloud)
    original_colors = np.asarray(pcd.colors).copy() if pcd.has_colors() else None

    # Pre-compute heat map colors
    heatmap_colors = None
    if len(crack_detections) > 0:
        all_crack_points = []
        for crack in crack_detections:
            if crack.points_3d is not None and len(crack.points_3d) > 0:
                all_crack_points.extend(crack.points_3d.tolist())

        if len(all_crack_points) > 0:
            crack_points = np.array(all_crack_points)
            crack_tree = cKDTree(crack_points)
            points = np.asarray(pcd.points)
            distances, _ = crack_tree.query(points, k=1)
            normalized = np.clip(distances / heatmap_distance, 0, 1)

            # Generate heat map colors
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

    # Visualization state
    show_heatmap = [False]  # Use list for mutable closure

    def toggle_heatmap(vis):
        show_heatmap[0] = not show_heatmap[0]
        if show_heatmap[0] and heatmap_colors is not None:
            pcd.colors = o3d.utility.Vector3dVector(heatmap_colors)
            print("Heat map: ON (Blue=far, Red=near crack)")
        elif original_colors is not None:
            pcd.colors = o3d.utility.Vector3dVector(original_colors)
            print("Heat map: OFF (original colors)")
        vis.update_geometry(pcd)
        return False

    def reset_colors(vis):
        show_heatmap[0] = False
        if original_colors is not None:
            pcd.colors = o3d.utility.Vector3dVector(original_colors)
            print("Reset to original colors")
        vis.update_geometry(pcd)
        return False

    # Create visualizer with key callbacks
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(window_name="Crack Detection Results", width=1280, height=720)

    # Add geometries
    vis.add_geometry(pcd)

    # Add crack markers
    for crack in crack_detections:
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
        description="Photo-First Crack Detection with 3D Mapping",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python crack_detector_photo_to_3d.py --photos ./images --model ./best.pt
  python crack_detector_photo_to_3d.py --photos ./images --output ./results --visualize
        """
    )

    parser.add_argument("--photos", "-p", type=str, required=True,
                       help="Path to folder containing photos")
    parser.add_argument("--model", "-m", type=str,
                       default="crack_detector/train/weights/best.pt",
                       help="Path to YOLOv8 model weights")
    parser.add_argument("--output", "-o", type=str, default="./output_option1",
                       help="Output directory")
    parser.add_argument("--confidence", "-c", type=float, default=0.25,
                       help="Detection confidence threshold (default: 0.25)")
    parser.add_argument("--use-colmap", action="store_true",
                       help="Prefer COLMAP for reconstruction (if available)")
    parser.add_argument("--visualize", "-v", action="store_true",
                       help="Show interactive 3D visualization")
    parser.add_argument("--skip-reconstruction", action="store_true",
                       help="Skip reconstruction if point cloud already exists")
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

    args = parser.parse_args()

    # Validate paths
    photos_dir = Path(args.photos)
    if not photos_dir.exists():
        print(f"Error: Photos directory not found: {photos_dir}")
        sys.exit(1)

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get image files
    image_extensions = [".jpg", ".jpeg", ".png", ".bmp", ".tiff"]
    image_paths = []
    for ext in image_extensions:
        image_paths.extend(photos_dir.glob(f"*{ext}"))
        image_paths.extend(photos_dir.glob(f"*{ext.upper()}"))

    image_paths = sorted(set(image_paths))

    if len(image_paths) < 3:
        print(f"Error: Need at least 3 images, found {len(image_paths)}")
        sys.exit(1)

    print("\n" + "=" * 60)
    print("OPTION 1: PHOTO-FIRST CRACK DETECTION")
    print("=" * 60)
    print(f"Photos directory: {photos_dir}")
    print(f"Number of images: {len(image_paths)}")
    print(f"Model path: {args.model}")
    print(f"Output directory: {output_dir}")

    # Step 1: Detect cracks in photos
    print("\n" + "=" * 60)
    print("STEP 1: DETECTING CRACKS IN PHOTOS")
    print("=" * 60)

    detector = YOLOCrackDetector(args.model, args.confidence, force_cpu=args.cpu)
    detections_2d = detector.detect_batch([str(p) for p in image_paths])

    total_detections = sum(len(d) for d in detections_2d.values())
    print(f"\nTotal 2D detections: {total_detections}")
    print(f"Images with cracks: {len(detections_2d)}")

    # Step 2: Photogrammetry reconstruction
    print("\n" + "=" * 60)
    print("STEP 2: PHOTOGRAMMETRY RECONSTRUCTION")
    print("=" * 60)

    use_icp = not args.no_icp
    reconstructor = PhotogrammetryReconstructor(
        output_dir, args.use_colmap, use_icp=use_icp, icp_type=args.icp_type
    )

    if use_icp:
        print(f"ICP refinement: ENABLED ({args.icp_type})")
    else:
        print("ICP refinement: DISABLED")

    existing_pcd = output_dir / "reconstructed.ply"
    if args.skip_reconstruction and existing_pcd.exists():
        print(f"Loading existing point cloud: {existing_pcd}")
        reconstructor._load_point_cloud(existing_pcd)
    else:
        success = reconstructor.reconstruct(photos_dir)
        if not success:
            print("Reconstruction failed")
            sys.exit(1)

    # Step 3: Map 2D detections to 3D
    mapper = CrackMapper3D(reconstructor)
    crack_detections_3d = mapper.map_detections(detections_2d)

    # Step 4: Export results
    exporter = PointCloudExporter(output_dir)
    output_path = exporter.export(
        reconstructor, crack_detections_3d,
        heatmap_max_distance=args.heatmap_distance
    )

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Images processed: {len(image_paths)}")
    print(f"2D detections: {total_detections}")
    print(f"3D crack regions: {len(crack_detections_3d)}")
    print(f"  - Severe: {sum(1 for c in crack_detections_3d if c.severity == 'SEVERE')}")
    print(f"  - Moderate: {sum(1 for c in crack_detections_3d if c.severity == 'MODERATE')}")
    print(f"  - Minor: {sum(1 for c in crack_detections_3d if c.severity == 'MINOR')}")
    print(f"\nOutput saved to: {output_dir}")

    # Optional visualization
    if args.visualize:
        visualize_results(reconstructor, crack_detections_3d, args.heatmap_distance)


if __name__ == "__main__":
    main()

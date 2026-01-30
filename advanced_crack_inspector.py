"""
Advanced Concrete Crack Inspector
Features:
- Real-time 2D crack map overlay with heatmaps
- QA metrics (blur, features, brightness)
- Coverage status tracking
- Risk assessment with confidence preview
- Rescan list generation
"""

import cv2
import numpy as np
from pathlib import Path
from datetime import datetime
from collections import deque
import json


class QAMetrics:
    """Image Quality Assessment"""

    @staticmethod
    def calculate_blur(image):
        """Calculate blur score using Laplacian variance (higher = sharper)"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        # Normalize to 0-100 scale
        blur_score = min(100, laplacian_var / 5)
        return blur_score, laplacian_var

    @staticmethod
    def calculate_brightness(image):
        """Calculate brightness score (0-100)"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        brightness = np.mean(gray)
        # Optimal brightness is around 127 (middle)
        score = 100 - abs(brightness - 127) / 1.27
        return score, brightness

    @staticmethod
    def calculate_contrast(image):
        """Calculate contrast score"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        contrast = gray.std()
        # Normalize to 0-100
        score = min(100, contrast / 0.64)
        return score, contrast

    @staticmethod
    def calculate_feature_density(image):
        """Calculate feature density using corner detection"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        corners = cv2.goodFeaturesToTrack(gray, maxCorners=500, qualityLevel=0.01, minDistance=10)
        num_features = len(corners) if corners is not None else 0
        # Normalize based on image size
        density = (num_features / (image.shape[0] * image.shape[1])) * 100000
        score = min(100, density * 2)
        return score, num_features

    @staticmethod
    def get_qa_status(blur, brightness, contrast, features):
        """Get overall QA status"""
        scores = [blur, brightness, contrast, features]
        avg_score = np.mean(scores)

        if avg_score >= 70:
            return "GOOD", (0, 255, 0)
        elif avg_score >= 50:
            return "FAIR", (0, 255, 255)
        else:
            return "POOR", (0, 0, 255)


class CrackAnalyzer:
    """Crack detection and analysis"""

    def __init__(self):
        self.crack_history = deque(maxlen=100)  # Store recent detections
        self.heatmap = None
        self.coverage_map = None
        self.frame_count = 0
        self.rescan_list = []

    def detect_cracks(self, image):
        """Detect cracks using edge detection and morphological operations"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Denoise
        denoised = cv2.fastNlMeansDenoising(gray, h=10)

        # Enhance contrast with CLAHE
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(denoised)

        # Multi-scale edge detection
        edges1 = cv2.Canny(enhanced, 30, 100)
        edges2 = cv2.Canny(enhanced, 50, 150)
        edges = cv2.bitwise_or(edges1, edges2)

        # Morphological operations to connect crack segments
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        edges = cv2.dilate(edges, kernel, iterations=1)
        edges = cv2.erode(edges, kernel, iterations=1)

        return edges, enhanced

    def analyze_cracks(self, edges, image):
        """Analyze detected cracks and return metrics"""
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        cracks = []
        total_crack_length = 0
        max_width = 0

        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 30:  # Filter noise
                continue

            # Get bounding rect
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = max(w, h) / (min(w, h) + 1)

            # Cracks are typically elongated
            if aspect_ratio > 1.5:
                # Calculate crack length (arc length)
                length = cv2.arcLength(contour, False)
                total_crack_length += length

                # Estimate width
                width = min(w, h)
                max_width = max(max_width, width)

                # Classify severity
                if width > 10:
                    severity = "SEVERE"
                    color = (0, 0, 255)
                    risk = 0.9
                elif width > 5:
                    severity = "MODERATE"
                    color = (0, 165, 255)
                    risk = 0.6
                else:
                    severity = "MINOR"
                    color = (0, 255, 255)
                    risk = 0.3

                cracks.append({
                    'contour': contour,
                    'bbox': (x, y, w, h),
                    'length': length,
                    'width': width,
                    'severity': severity,
                    'color': color,
                    'risk': risk,
                    'area': area
                })

        return cracks, total_crack_length, max_width

    def update_heatmap(self, cracks, frame_shape):
        """Update cumulative heatmap based on crack detections"""
        if self.heatmap is None:
            self.heatmap = np.zeros((frame_shape[0], frame_shape[1]), dtype=np.float32)

        # Create current frame crack mask
        crack_mask = np.zeros((frame_shape[0], frame_shape[1]), dtype=np.float32)

        for crack in cracks:
            # Fill crack contour with intensity based on severity
            intensity = crack['risk']  # 0.3 to 0.9
            cv2.drawContours(crack_mask, [crack['contour']], -1, intensity, -1)  # Filled
            # Also dilate to make cracks more visible in heatmap
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
            dilated = cv2.dilate(crack_mask, kernel, iterations=1)
            crack_mask = np.maximum(crack_mask, dilated * 0.5)

        # Accumulate with slower decay for persistence
        self.heatmap = self.heatmap * 0.92 + crack_mask * 0.15

        # Normalize and enhance contrast
        heatmap_normalized = np.clip(self.heatmap * 3, 0, 1)  # Boost intensity
        heatmap_display = (heatmap_normalized * 255).astype(np.uint8)

        # Apply colormap (TURBO gives better color range than JET)
        heatmap_color = cv2.applyColorMap(heatmap_display, cv2.COLORMAP_TURBO)

        return heatmap_color

    def update_coverage(self, frame_shape, cracks):
        """Track which areas have been scanned"""
        if self.coverage_map is None:
            self.coverage_map = np.zeros((frame_shape[0], frame_shape[1]), dtype=np.uint8)

        # Mark current frame area as scanned
        self.coverage_map = np.clip(self.coverage_map + 5, 0, 255)

        # Calculate coverage percentage
        scanned_pixels = np.sum(self.coverage_map > 50)
        total_pixels = self.coverage_map.shape[0] * self.coverage_map.shape[1]
        coverage_pct = (scanned_pixels / total_pixels) * 100

        return coverage_pct

    def calculate_risk(self, cracks):
        """Calculate overall risk score"""
        if not cracks:
            return 0, "LOW"

        # Weight by severity
        total_risk = sum(c['risk'] * c['area'] for c in cracks)
        max_risk = max(c['risk'] for c in cracks)

        # Normalize
        risk_score = min(1.0, max_risk * 0.7 + (total_risk / 10000) * 0.3)

        if risk_score > 0.7:
            level = "HIGH"
        elif risk_score > 0.4:
            level = "MEDIUM"
        else:
            level = "LOW"

        return risk_score, level

    def add_to_rescan(self, frame_id, reason, confidence):
        """Add frame to rescan list"""
        self.rescan_list.append({
            'frame': frame_id,
            'timestamp': datetime.now().isoformat(),
            'reason': reason,
            'confidence': confidence
        })


class CrackInspectorUI:
    """UI rendering for the inspector"""

    @staticmethod
    def draw_crack_overlay(image, cracks, use_segmentation=True):
        """Draw 2D crack map overlay with segmentation masks"""
        overlay = image.copy()
        mask_overlay = np.zeros_like(image)

        for crack in cracks:
            if use_segmentation:
                # Draw filled segmentation mask
                cv2.drawContours(mask_overlay, [crack['contour']], -1, crack['color'], -1)

                # Draw contour outline for definition
                cv2.drawContours(overlay, [crack['contour']], -1, crack['color'], 2)

                # Add glow effect around cracks
                glow_mask = np.zeros(image.shape[:2], dtype=np.uint8)
                cv2.drawContours(glow_mask, [crack['contour']], -1, 255, -1)
                glow_mask = cv2.GaussianBlur(glow_mask, (21, 21), 0)
                glow_colored = np.zeros_like(image)
                glow_colored[:] = crack['color']
                glow_alpha = glow_mask.astype(np.float32) / 255 * 0.3
                for c in range(3):
                    mask_overlay[:, :, c] = np.clip(
                        mask_overlay[:, :, c] + (glow_colored[:, :, c] * glow_alpha).astype(np.uint8),
                        0, 255
                    )
            else:
                # Fallback to bounding box
                x, y, w, h = crack['bbox']
                cv2.rectangle(overlay, (x, y), (x+w, y+h), crack['color'], 2)

            # Label with severity and confidence if available
            x, y, w, h = crack['bbox']
            conf_text = f" {crack.get('confidence', 0)*100:.0f}%" if 'confidence' in crack else ""
            label = f"{crack['severity']}{conf_text}"

            # Background for text
            (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
            cv2.rectangle(overlay, (x, y-text_h-6), (x+text_w+4, y), (0, 0, 0), -1)
            cv2.putText(overlay, label, (x+2, y-4),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, crack['color'], 1)

        # Blend segmentation mask with original
        result = cv2.addWeighted(overlay, 0.7, mask_overlay, 0.5, 0)
        return result

    @staticmethod
    def draw_qa_panel(image, blur, brightness, contrast, features, qa_status):
        """Draw QA metrics panel"""
        h, w = image.shape[:2]
        panel_width = 200

        # Semi-transparent panel
        panel = np.zeros((150, panel_width, 3), dtype=np.uint8)
        panel[:] = (40, 40, 40)

        # Title
        cv2.putText(panel, "QA METRICS", (10, 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Metrics with bars
        metrics = [
            ("Sharpness", blur[0]),
            ("Brightness", brightness[0]),
            ("Contrast", contrast[0]),
            ("Features", features[0])
        ]

        y_offset = 40
        for name, score in metrics:
            # Label
            cv2.putText(panel, f"{name}: {score:.0f}%", (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

            # Progress bar
            bar_width = int((score / 100) * 100)
            color = (0, 255, 0) if score >= 70 else (0, 255, 255) if score >= 50 else (0, 0, 255)
            cv2.rectangle(panel, (90, y_offset - 10), (90 + bar_width, y_offset - 2), color, -1)
            cv2.rectangle(panel, (90, y_offset - 10), (190, y_offset - 2), (100, 100, 100), 1)

            y_offset += 25

        # QA Status
        status_text, status_color = qa_status
        cv2.putText(panel, f"Status: {status_text}", (10, y_offset + 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, status_color, 1)

        # Overlay panel on image
        image[10:160, w - panel_width - 10:w - 10] = panel

        return image

    @staticmethod
    def draw_risk_panel(image, risk_score, risk_level, crack_count, total_length):
        """Draw risk assessment panel"""
        h, w = image.shape[:2]
        panel_width = 200

        panel = np.zeros((120, panel_width, 3), dtype=np.uint8)
        panel[:] = (40, 40, 40)

        # Title
        cv2.putText(panel, "RISK ASSESSMENT", (10, 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Risk level with color
        risk_colors = {"HIGH": (0, 0, 255), "MEDIUM": (0, 165, 255), "LOW": (0, 255, 0)}
        color = risk_colors.get(risk_level, (255, 255, 255))

        cv2.putText(panel, f"Risk: {risk_level}", (10, 45),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # Confidence bar
        cv2.putText(panel, f"Confidence: {risk_score*100:.0f}%", (10, 70),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

        bar_width = int(risk_score * 150)
        cv2.rectangle(panel, (10, 75), (10 + bar_width, 85), color, -1)
        cv2.rectangle(panel, (10, 75), (160, 85), (100, 100, 100), 1)

        # Stats
        cv2.putText(panel, f"Cracks: {crack_count} | Length: {total_length:.0f}px", (10, 105),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1)

        # Overlay
        image[170:290, w - panel_width - 10:w - 10] = panel

        return image

    @staticmethod
    def draw_coverage_panel(image, coverage_pct, frame_count):
        """Draw coverage status panel"""
        h, w = image.shape[:2]
        panel_width = 200

        panel = np.zeros((60, panel_width, 3), dtype=np.uint8)
        panel[:] = (40, 40, 40)

        cv2.putText(panel, "COVERAGE STATUS", (10, 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        cv2.putText(panel, f"Scanned: {coverage_pct:.1f}%  Frame: {frame_count}", (10, 45),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

        # Coverage bar
        bar_width = int((coverage_pct / 100) * 150)
        cv2.rectangle(panel, (10, 48), (10 + bar_width, 55), (0, 255, 0), -1)
        cv2.rectangle(panel, (10, 48), (160, 55), (100, 100, 100), 1)

        image[300:360, w - panel_width - 10:w - 10] = panel

        return image


def run_inspector():
    """Main inspection loop"""
    print("="*60)
    print("ADVANCED CONCRETE CRACK INSPECTOR")
    print("="*60)

    # Try to load YOLOv8 model if available
    use_yolo = False
    model = None

    try:
        from ultralytics import YOLO
        # Check multiple possible model locations
        possible_paths = [
            Path('crack_detector/train/weights/best.pt'),
            Path.home() / 'crack_detector/train/weights/best.pt',
            Path('C:/Users/USER/crack_detector/train/weights/best.pt'),
            Path.cwd() / 'crack_detector/train/weights/best.pt',
        ]

        model_path = None
        for p in possible_paths:
            if p.exists():
                model_path = p
                break

        if model_path:
            model = YOLO(str(model_path))
            use_yolo = True
            print(f"Loaded YOLOv8 model: {model_path}")
        else:
            print("No trained YOLOv8 model found. Using OpenCV detection.")
    except ImportError:
        print("Ultralytics not installed. Using OpenCV detection.")

    # Initialize components
    analyzer = CrackAnalyzer()
    qa = QAMetrics()
    ui = CrackInspectorUI()

    # Open webcam
    print("\nOpening webcam...")
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open webcam")
        return

    print("\nControls:")
    print("  Q - Quit")
    print("  H - Toggle heatmap overlay")
    print("  S - Save current frame report")
    print("  R - Reset heatmap and coverage")
    print("  M - Toggle detection mode (YOLOv8/OpenCV)")

    show_heatmap = False
    frame_count = 0
    reports = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        display = frame.copy()

        # Calculate QA metrics
        blur = qa.calculate_blur(frame)
        brightness = qa.calculate_brightness(frame)
        contrast = qa.calculate_contrast(frame)
        features = qa.calculate_feature_density(frame)
        qa_status = qa.get_qa_status(blur[0], brightness[0], contrast[0], features[0])

        # Detect cracks
        if use_yolo and model is not None:
            # YOLOv8 detection
            results = model(frame, conf=0.25, verbose=False)
            cracks = []
            for r in results:
                for box in r.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = float(box.conf[0])
                    w, h = x2 - x1, y2 - y1

                    if w > 10:
                        severity = "SEVERE"
                        color = (0, 0, 255)
                        risk = 0.9
                    elif w > 5:
                        severity = "MODERATE"
                        color = (0, 165, 255)
                        risk = 0.6
                    else:
                        severity = "MINOR"
                        color = (0, 255, 255)
                        risk = 0.3

                    # Extract crack region and find actual contour using edge detection
                    roi = frame[max(0,y1):min(frame.shape[0],y2), max(0,x1):min(frame.shape[1],x2)]
                    if roi.size > 0:
                        gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                        # Enhance and detect edges within ROI
                        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4,4))
                        enhanced_roi = clahe.apply(gray_roi)
                        edges_roi = cv2.Canny(enhanced_roi, 30, 100)
                        # Find contours in ROI
                        roi_contours, _ = cv2.findContours(edges_roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        if roi_contours:
                            # Get largest contour and offset to frame coordinates
                            largest = max(roi_contours, key=cv2.contourArea)
                            contour = largest + np.array([x1, y1])
                        else:
                            contour = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]])
                    else:
                        contour = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]])

                    cracks.append({
                        'contour': contour,
                        'bbox': (x1, y1, w, h),
                        'length': max(w, h),
                        'width': min(w, h),
                        'severity': severity,
                        'color': color,
                        'risk': risk * conf,
                        'area': w * h,
                        'confidence': conf
                    })

            edges = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)
            total_length = sum(c['length'] for c in cracks)
            max_width = max([c['width'] for c in cracks], default=0)
        else:
            # OpenCV detection
            edges, enhanced = analyzer.detect_cracks(frame)
            cracks, total_length, max_width = analyzer.analyze_cracks(edges, frame)

        # Update heatmap (now based on detected cracks, not edges)
        heatmap = analyzer.update_heatmap(cracks, frame.shape)

        # Update coverage
        coverage_pct = analyzer.update_coverage(frame.shape, cracks)

        # Calculate risk
        risk_score, risk_level = analyzer.calculate_risk(cracks)

        # Check if rescan needed (poor QA or uncertain detection)
        if qa_status[0] == "POOR":
            analyzer.add_to_rescan(frame_count, "Poor image quality", blur[0])

        # Draw overlays
        if show_heatmap:
            display = cv2.addWeighted(display, 0.6, heatmap, 0.4, 0)

        display = ui.draw_crack_overlay(display, cracks)
        display = ui.draw_qa_panel(display, blur, brightness, contrast, features, qa_status)
        display = ui.draw_risk_panel(display, risk_score, risk_level, len(cracks), total_length)
        display = ui.draw_coverage_panel(display, coverage_pct, frame_count)

        # Mode indicator
        mode_text = "YOLOv8" if use_yolo else "OpenCV"
        cv2.putText(display, f"Mode: {mode_text} | H=Heatmap S=Save Q=Quit", (10, display.shape[0] - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        cv2.imshow('Advanced Crack Inspector', display)

        # Handle keys
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break
        elif key == ord('h'):
            show_heatmap = not show_heatmap
            print(f"Heatmap: {'ON' if show_heatmap else 'OFF'}")
        elif key == ord('s'):
            # Save report
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            report = {
                'timestamp': timestamp,
                'frame': frame_count,
                'qa_metrics': {
                    'blur': blur[0],
                    'brightness': brightness[0],
                    'contrast': contrast[0],
                    'features': features[0],
                    'status': qa_status[0]
                },
                'risk': {
                    'score': risk_score,
                    'level': risk_level
                },
                'cracks': {
                    'count': len(cracks),
                    'total_length': total_length,
                    'max_width': max_width
                },
                'coverage': coverage_pct
            }
            reports.append(report)

            # Save image
            img_path = f"crack_report_{timestamp}.jpg"
            cv2.imwrite(img_path, display)

            # Save JSON report
            json_path = f"crack_report_{timestamp}.json"
            with open(json_path, 'w') as f:
                json.dump(report, f, indent=2)

            print(f"Saved: {img_path}, {json_path}")

        elif key == ord('r'):
            analyzer.heatmap = None
            analyzer.coverage_map = None
            print("Reset heatmap and coverage")

        elif key == ord('m'):
            if model is not None:
                use_yolo = not use_yolo
                print(f"Switched to: {'YOLOv8' if use_yolo else 'OpenCV'}")

    cap.release()
    cv2.destroyAllWindows()

    # Final summary
    print("\n" + "="*60)
    print("INSPECTION SUMMARY")
    print("="*60)
    print(f"Total frames analyzed: {frame_count}")
    print(f"Reports saved: {len(reports)}")
    print(f"Items for rescan: {len(analyzer.rescan_list)}")

    if analyzer.rescan_list:
        print("\nRESCAN LIST:")
        for item in analyzer.rescan_list[-10:]:  # Show last 10
            print(f"  Frame {item['frame']}: {item['reason']} (conf: {item['confidence']:.1f}%)")

    # Save final summary
    if reports or analyzer.rescan_list:
        summary = {
            'total_frames': frame_count,
            'reports': reports,
            'rescan_list': analyzer.rescan_list
        }
        with open('inspection_summary.json', 'w') as f:
            json.dump(summary, f, indent=2)
        print("\nSaved: inspection_summary.json")


def analyze_image():
    """Analyze a single image"""
    print("="*60)
    print("SINGLE IMAGE ANALYSIS")
    print("="*60)

    image_path = input("\nEnter image path: ").strip()

    if not image_path or not Path(image_path).exists():
        print("Invalid path")
        return

    image = cv2.imread(image_path)
    if image is None:
        print("Could not read image")
        return

    # Initialize
    analyzer = CrackAnalyzer()
    qa = QAMetrics()
    ui = CrackInspectorUI()

    # Analyze
    blur = qa.calculate_blur(image)
    brightness = qa.calculate_brightness(image)
    contrast = qa.calculate_contrast(image)
    features = qa.calculate_feature_density(image)
    qa_status = qa.get_qa_status(blur[0], brightness[0], contrast[0], features[0])

    edges, enhanced = analyzer.detect_cracks(image)
    cracks, total_length, max_width = analyzer.analyze_cracks(edges, image)
    risk_score, risk_level = analyzer.calculate_risk(cracks)

    # Generate heatmap
    heatmap = cv2.applyColorMap(edges, cv2.COLORMAP_JET)

    # Draw results
    display = ui.draw_crack_overlay(image, cracks)
    display = ui.draw_qa_panel(display, blur, brightness, contrast, features, qa_status)
    display = ui.draw_risk_panel(display, risk_score, risk_level, len(cracks), total_length)

    # Print report
    print("\n" + "-"*40)
    print("ANALYSIS REPORT")
    print("-"*40)
    print(f"\nQA METRICS:")
    print(f"  Sharpness:  {blur[0]:.1f}%")
    print(f"  Brightness: {brightness[0]:.1f}%")
    print(f"  Contrast:   {contrast[0]:.1f}%")
    print(f"  Features:   {features[0]:.1f}%")
    print(f"  Status:     {qa_status[0]}")

    print(f"\nCRACK ANALYSIS:")
    print(f"  Cracks detected: {len(cracks)}")
    print(f"  Total length:    {total_length:.0f}px")
    print(f"  Max width:       {max_width}px")

    print(f"\nRISK ASSESSMENT:")
    print(f"  Risk level:  {risk_level}")
    print(f"  Confidence:  {risk_score*100:.0f}%")

    if qa_status[0] == "POOR":
        print(f"\n[!] RECOMMEND RESCAN - Poor image quality")

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    cv2.imwrite(f"analysis_{timestamp}.jpg", display)
    cv2.imwrite(f"heatmap_{timestamp}.jpg", heatmap)

    print(f"\nSaved: analysis_{timestamp}.jpg, heatmap_{timestamp}.jpg")

    # Show
    cv2.imshow('Analysis Result', display)
    cv2.imshow('Crack Heatmap', heatmap)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def main():
    print("\n" + "="*60)
    print("ADVANCED CONCRETE CRACK INSPECTOR")
    print("="*60)

    print("\nOptions:")
    print("1. Real-time webcam inspection")
    print("2. Analyze single image")

    choice = input("\nSelect option (1-2): ").strip()

    if choice == "1":
        run_inspector()
    elif choice == "2":
        analyze_image()
    else:
        print("Invalid option")


if __name__ == "__main__":
    main()

# Option 1: Photo-First Crack Detection with 3D Mapping

Detect cracks on original photos first, then map detections to a reconstructed 3D point cloud.

## Setup

```bash
# Create conda environment
conda env create -f environment.yml

# Activate environment
conda activate crack_detect_photo

# (Optional) Install COLMAP for better reconstruction
# Windows: Download from https://github.com/colmap/colmap/releases
# Add to PATH or place in project folder
```

## Usage

### Basic Usage
```bash
python crack_detector_photo_to_3d.py --photos ./your_photos --model ./best.pt
```

### With Your Trained Model
```bash
python crack_detector_photo_to_3d.py \
    --photos C:/Users/USER/project_photos \
    --model C:/Users/USER/crack_detector/train/weights/best.pt \
    --output ./results \
    --visualize
```

### All Options
```bash
python crack_detector_photo_to_3d.py --help
```

| Argument | Description | Default |
|----------|-------------|---------|
| `--photos`, `-p` | Path to photos folder | Required |
| `--model`, `-m` | YOLOv8 model path | `crack_detector/train/weights/best.pt` |
| `--output`, `-o` | Output directory | `./output_option1` |
| `--confidence`, `-c` | Detection threshold | `0.25` |
| `--use-colmap` | Prefer COLMAP reconstruction | `False` |
| `--visualize`, `-v` | Show 3D visualization | `False` |
| `--skip-reconstruction` | Use existing point cloud | `False` |

## Output Files

```
output_option1/
├── reconstructed.ply              # Original point cloud
├── annotated_cracks_original_*.ply    # Original without annotations
├── annotated_cracks_cracks_only_*.ply # Crack points only (colored)
├── annotated_cracks_combined_*.ply    # Combined with crack highlights
├── annotated_cracks_combined_*.pcd    # PCD format
└── crack_report_*.json            # Detection report
```

## Workflow

1. **Crack Detection**: YOLOv8 detects cracks in each original photo
2. **Photogrammetry**: Photos are reconstructed into a 3D point cloud
3. **Mapping**: 2D crack pixels are projected to 3D using camera poses
4. **Export**: Annotated point cloud with cracks colored by severity

## Color Coding
- **Red**: Severe cracks
- **Orange**: Moderate cracks
- **Yellow**: Minor cracks

## Tips

- Use overlapping photos (60-80% overlap) for good reconstruction
- Consistent lighting improves both detection and reconstruction
- More photos = denser point cloud but longer processing
- If COLMAP is available, it typically produces better results

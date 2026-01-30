# Option 2: Point Cloud First - Render Views and Detect

Build/load a point cloud first, render multiple 2D views, detect cracks, then map back to 3D.

## Setup

```bash
# Create conda environment
conda env create -f environment.yml

# Activate environment
conda activate crack_detect_pcd

# (Optional) Install COLMAP for better reconstruction from photos
# Windows: Download from https://github.com/colmap/colmap/releases
```

## Usage

### From Existing Point Cloud (Recommended)
```bash
python crack_detector_pcd_render.py --pcd ./your_scan.ply --model ./best.pt
```

### From Photos
```bash
python crack_detector_pcd_render.py --photos ./your_photos --model ./best.pt
```

### With Your Trained Model
```bash
python crack_detector_pcd_render.py \
    --pcd C:/Users/USER/scan.ply \
    --model C:/Users/USER/crack_detector/train/weights/best.pt \
    --output ./results \
    --num-views 30 \
    --visualize
```

### All Options
```bash
python crack_detector_pcd_render.py --help
```

| Argument | Description | Default |
|----------|-------------|---------|
| `--pcd` | Path to existing point cloud | - |
| `--photos`, `-p` | Path to photos folder (alternative) | - |
| `--model`, `-m` | YOLOv8 model path | `crack_detector/train/weights/best.pt` |
| `--output`, `-o` | Output directory | `./output_option2` |
| `--num-views` | Number of views to render | `20` |
| `--view-type` | `spherical` or `orbit` | `spherical` |
| `--confidence`, `-c` | Detection threshold | `0.25` |
| `--render-width` | Rendered image width | `1280` |
| `--render-height` | Rendered image height | `960` |
| `--use-depth` | Use depth maps for projection | `False` |
| `--visualize`, `-v` | Show 3D visualization | `False` |

## Supported Point Cloud Formats

- `.ply` - PLY format
- `.pcd` - Point Cloud Data
- `.xyz` - XYZ ASCII
- `.pts` - PTS format
- `.las` / `.laz` - LiDAR formats

## Output Files

```
output_option2/
├── reconstructed.ply              # Point cloud (if built from photos)
├── rendered_views/                # Rendered 2D views
│   ├── view_0000.png
│   ├── view_0001.png
│   └── ...
├── annotated_pcd_original_*.ply   # Original point cloud
├── annotated_pcd_colored_*.ply    # With crack highlights
├── annotated_pcd_colored_*.pcd    # PCD format
├── annotated_pcd_cracks_only_*.ply # Crack points only
└── annotated_pcd_report_*.json    # Detection report
```

## Workflow

1. **Load/Build**: Load existing point cloud or build from photos
2. **Render**: Generate multiple 2D views around the point cloud
3. **Detect**: Run YOLOv8 on each rendered view
4. **Back-project**: Map 2D detections back to 3D point coordinates
5. **Export**: Save annotated point cloud with crack markings

## View Types

- **spherical**: Views distributed on a sphere around the object (better coverage)
- **orbit**: Horizontal orbit around the object (good for vertical structures)

## Color Coding
- **Red**: Severe cracks
- **Orange**: Moderate cracks
- **Yellow**: Minor cracks

## Tips

- More views = better coverage but longer processing
- Use `--use-depth` for more accurate back-projection
- `spherical` view type provides better coverage for complex shapes
- Increase `--num-views` to 30-50 for detailed structures
- The rendered views are saved and can be inspected for quality

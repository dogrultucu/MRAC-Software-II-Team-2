"""
Train YOLOv8 Segmentation Model for Crack Detection
====================================================
This script trains a YOLOv8 segmentation model on crack dataset.

Usage:
    python train_segmentation.py

The model will output pixel-level crack masks instead of just bounding boxes.
"""

from ultralytics import YOLO
from pathlib import Path
import torch

def main():
    print("=" * 60)
    print("YOLOV8 SEGMENTATION TRAINING")
    print("=" * 60)

    # Check GPU
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        # Note: RTX 5080 may need CPU fallback
        try:
            _ = torch.zeros(1).cuda()
            device = 0  # Use GPU
            print("CUDA: Available and working")
        except:
            device = 'cpu'
            print("CUDA: Not compatible, using CPU")
    else:
        device = 'cpu'
        print("CUDA: Not available, using CPU")

    # Dataset path
    data_yaml = Path(__file__).parent / "data.yaml"
    print(f"\nDataset: {data_yaml}")

    # Load YOLOv8 segmentation model (pretrained)
    # Options: yolov8n-seg.pt (nano), yolov8s-seg.pt (small), yolov8m-seg.pt (medium)
    print("\nLoading YOLOv8n-seg pretrained model...")
    model = YOLO('yolov8n-seg.pt')  # Nano model - fastest training

    # Training configuration
    print("\n" + "=" * 60)
    print("STARTING TRAINING")
    print("=" * 60)

    results = model.train(
        data=str(data_yaml),
        epochs=100,              # Number of training epochs
        imgsz=640,               # Image size
        batch=16,                # Batch size (reduce if GPU memory error)
        device=device,           # GPU or CPU
        workers=4,               # Data loader workers
        patience=20,             # Early stopping patience
        save=True,               # Save checkpoints
        project='runs/segment',  # Output directory
        name='crack_segmentation',  # Run name
        exist_ok=True,           # Overwrite existing
        pretrained=True,         # Use pretrained weights
        optimizer='Adam',        # Optimizer
        lr0=0.001,               # Initial learning rate
        lrf=0.01,                # Final learning rate factor
        mosaic=1.0,              # Mosaic augmentation
        mixup=0.1,               # Mixup augmentation
        copy_paste=0.1,          # Copy-paste augmentation (good for segmentation)
        degrees=10,              # Rotation augmentation
        translate=0.1,           # Translation augmentation
        scale=0.5,               # Scale augmentation
        fliplr=0.5,              # Horizontal flip probability
        flipud=0.1,              # Vertical flip probability
        hsv_h=0.015,             # HSV hue augmentation
        hsv_s=0.7,               # HSV saturation augmentation
        hsv_v=0.4,               # HSV value augmentation
    )

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)

    # Best model path
    best_model = Path('runs/segment/crack_segmentation/weights/best.pt')
    print(f"\nBest model saved to: {best_model}")

    # Copy to easy location
    import shutil
    output_model = Path(__file__).parent / "best-seg.pt"
    if best_model.exists():
        shutil.copy(best_model, output_model)
        print(f"Copied to: {output_model}")

    print("\n" + "=" * 60)
    print("TO USE THE SEGMENTATION MODEL:")
    print("=" * 60)
    print(f"""
python crack_detector_pcd_render.py \\
    --pcd "your_pointcloud.pcd" \\
    -m "{output_model}" \\
    --visualize
    """)


if __name__ == "__main__":
    main()

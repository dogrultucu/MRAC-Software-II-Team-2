@echo off
echo ============================================
echo Option 2: PCD Render Crack Detection Setup
echo ============================================
echo.

REM Check if conda is available
where conda >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Conda not found. Please install Anaconda or Miniconda first.
    echo Download from: https://docs.conda.io/en/latest/miniconda.html
    pause
    exit /b 1
)

echo Creating conda environment 'crack_detect_pcd'...
echo This may take several minutes...
echo.

conda env create -f environment.yml

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo Environment creation failed. Trying alternative approach...
    echo.

    conda create -n crack_detect_pcd python=3.10 -y
    conda activate crack_detect_pcd

    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
    pip install ultralytics open3d opencv-python numpy scipy matplotlib tqdm plyfile laspy[lazrs] pycolmap transforms3d trimesh pyglet
)

echo.
echo ============================================
echo Setup Complete!
echo ============================================
echo.
echo To use:
echo   1. Open Anaconda Prompt
echo   2. Run: conda activate crack_detect_pcd
echo   3. Run: python crack_detector_pcd_render.py --pcd ./your_scan.ply --model ./best.pt
echo.
echo Or from photos:
echo   python crack_detector_pcd_render.py --photos ./your_photos --model ./best.pt
echo.
pause

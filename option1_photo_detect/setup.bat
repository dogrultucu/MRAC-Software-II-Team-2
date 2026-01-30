@echo off
echo ============================================
echo Option 1: Photo-First Crack Detection Setup
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

echo Creating conda environment 'crack_detect_photo'...
echo This may take several minutes...
echo.

conda env create -f environment.yml

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo Environment creation failed. Trying alternative approach...
    echo.

    conda create -n crack_detect_photo python=3.10 -y
    conda activate crack_detect_photo

    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
    pip install ultralytics open3d opencv-python numpy scipy matplotlib tqdm plyfile laspy[lazrs] pycolmap transforms3d
)

echo.
echo ============================================
echo Setup Complete!
echo ============================================
echo.
echo To use:
echo   1. Open Anaconda Prompt
echo   2. Run: conda activate crack_detect_photo
echo   3. Run: python crack_detector_photo_to_3d.py --photos ./your_photos --model ./best.pt
echo.
echo Optional: Install COLMAP for better reconstruction
echo   Download from: https://github.com/colmap/colmap/releases
echo.
pause

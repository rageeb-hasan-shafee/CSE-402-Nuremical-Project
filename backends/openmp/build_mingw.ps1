# Build script for PowerShell on Windows
$cxx = "C:\Program Files\CodeBlocks\MinGW\bin\g++.exe"
if (-not (Test-Path $cxx)) {
    $cxx = "g++"
}

if (-not (Test-Path "build")) {
    New-Item -ItemType Directory -Path "build" | Out-Null
}

Write-Host "Compiling nbody_serial..." -ForegroundColor Cyan
& $cxx -O3 -std=c++17 -I../../common/cpp nbody_omp.cpp -o build/nbody_serial.exe
if ($LASTEXITCODE -ne 0) {
    Write-Error "Error compiling nbody_serial.exe"
    exit 1
}

Write-Host "Compiling nbody_omp..." -ForegroundColor Cyan
& $cxx -O3 -fopenmp -std=c++17 -I../../common/cpp nbody_omp.cpp -o build/nbody_omp.exe
if ($LASTEXITCODE -ne 0) {
    Write-Error "Error compiling nbody_omp.exe"
    exit 1
}

Write-Host "Build succeeded! Executables are in backends/openmp/build/" -ForegroundColor Green

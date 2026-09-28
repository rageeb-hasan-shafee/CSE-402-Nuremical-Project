@echo off
rem Build script for MinGW GCC on Windows
set CXX="C:\Program Files\CodeBlocks\MinGW\bin\g++.exe"
if not exist %CXX% set CXX=g++

if not exist build mkdir build

echo Compiling nbody_serial...
%CXX% -O3 -std=c++17 -I../../common/cpp nbody_omp.cpp -o build/nbody_serial.exe
if errorlevel 1 (
    echo Error compiling nbody_serial.exe
    exit /b 1
)

echo Compiling nbody_omp...
%CXX% -O3 -fopenmp -std=c++17 -I../../common/cpp nbody_omp.cpp -o build/nbody_omp.exe
if errorlevel 1 (
    echo Error compiling nbody_omp.exe
    exit /b 1
)

echo Build succeeded! Executables are in backends/openmp/build/

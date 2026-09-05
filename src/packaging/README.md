# MedEye3D - Windows Standalone Packaging System

This directory contains the production-ready build and packaging infrastructure to transform **MedEye3D.jl** into an installable, standalone Microsoft Windows software application (`MedEye3D_Setup.exe`).

---

## 📁 Packaging Subsystem Structure

| File | Purpose |
| :--- | :--- |
| [`AppMain.jl`](file:///d:/MedEye3d.jl/src/packaging/AppMain.jl) | Standalone application wrapper, GUI initialization, stdio redirection to `%APPDATA%\MedEye3D\logs` to prevent Windows GUI subsystem fd -2 crashes, and CLI argument dispatch (`--demo`, `--help`, image files). |
| [`precompile_app.jl`](file:///d:/MedEye3d.jl/src/packaging/precompile_app.jl) | Workload execution tracer passed to `PackageCompiler.jl`. Exercises all types, `TextureSpec`, array transformations, resamplings, and OpenGL structures during build to eliminate JIT latency. |
| [`build_app.jl`](file:///d:/MedEye3d.jl/src/packaging/build_app.jl) | Julia build script executing `PackageCompiler.create_app()` with transitive dependency inclusion, artifact bundling, and asset management. |
| [`MedEye3D_Installer.iss`](file:///d:/MedEye3d.jl/src/packaging/MedEye3D_Installer.iss) | Inno Setup 6 script configured for LZMA2 ultra-compression, Start Menu/Desktop shortcuts, Windows Add/Remove programs registry integration, and medical file associations (`.nii`, `.nii.gz`, `.mha`, `.h5`). |
| [`build_installer.ps1`](file:///d:/MedEye3d.jl/src/packaging/build_installer.ps1) | One-command end-to-end PowerShell build pipeline with Inno Setup auto-detection and integrity verification. |
| [`generate_icon.ps1`](file:///d:/MedEye3d.jl/src/packaging/generate_icon.ps1) | Generates the multi-resolution `.ico` icon file (16x16 to 256x256) embedded in the installer and executable. |
| [`app_icon.ico`](file:///d:/MedEye3d.jl/src/packaging/app_icon.ico) | Multi-resolution icon for MedEye3D. |

---

## 🚀 Quick Start: Building the Installer

### Prerequisites
1. **Julia 1.9+** (x86_64) installed and accessible on PATH.
2. **Inno Setup 6** (can be automatically installed by the script via `winget`).
3. Standard Windows PowerShell 5.1+ or PowerShell 7+.

### One-Command Build
Run the automated pipeline in PowerShell:

```powershell
cd D:\MedEye3d.jl\src\packaging
.\build_installer.ps1 -InstallInno
```

The script will:
1. Validate dependencies and compile the standalone application into `build\MedEye3D_dist`.
2. Locate or install Inno Setup 6.
3. Generate the final single-file setup wizard at:
   ```
   D:\MedEye3d.jl\dist\MedEye3D_v0.5.8_Setup.exe
   ```

---

## 🛠️ CLI Options & Build Flags

### `build_installer.ps1`
- `-SkipCompile`: Skips the Julia Ahead-of-Time compilation step and quickly rebuilds the `.iss` installer if binary files already exist in `build\MedEye3D_dist`.
- `-InstallInno`: Automatically installs Inno Setup via `winget` if missing.
- `-Clean`: Removes previous `build/` and `dist/` directories before compiling.

```powershell
# Quick re-pack of existing build
.\build_installer.ps1 -SkipCompile

# Clean rebuild
.\build_installer.ps1 -Clean
```

---

## 🔍 Standalone Application Features & CLI

The packaged executable (`MedEye3D.exe`) supports:
- **Interactive Multi-planar Quad View** (Axial, Coronal, Sagittal, and 3D Mask views).
- **Direct File Opening**:
  ```cmd
  MedEye3D.exe "C:\path\to\patient_scan.nii.gz"
  ```
- **Synthetic Phantom Demo**:
  ```cmd
  MedEye3D.exe --demo
  ```
- **Help & Version**:
  ```cmd
  MedEye3D.exe --help
  MedEye3D.exe --version
  ```

### Windows Logging & Troubleshooting
When running as a GUI application, stdout/stderr are redirected to avoid Windows descriptor errors:
- Output log: `%APPDATA%\MedEye3D\logs\medeye3d_output.log`
- Error log: `%APPDATA%\MedEye3D\logs\medeye3d_error.log`

---

## 📦 File Associations Installed
When the user installs via `MedEye3D_Setup.exe`, optional shell integration allows double-clicking medical image files directly:
- `.nii` / `.nii.gz` (NIfTI Medical Image)
- `.mha` / `.mhd` (MetaImage Medical Volume)
- `.h5` / `.hdf5` (HDF5 Medical Dataset)

"""
Build tiny-qpu as a standalone Windows executable.

Usage (from E:\\QPU\\tiny-qpu):
    pip install pyinstaller
    python build_exe.py

This creates: dist/tiny-qpu.exe (single file, ~30-50 MB)
Double-click to launch the Interactive Quantum Lab.
"""

import os
import sys
import shutil
import subprocess

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
DASHBOARD_DIR = os.path.join(SRC_DIR, "tiny_qpu", "dashboard")
TEMPLATES_DIR = os.path.join(DASHBOARD_DIR, "templates")
STATIC_DIR = os.path.join(DASHBOARD_DIR, "static")
LAUNCHER = os.path.join(PROJECT_ROOT, "tiny_qpu_launcher.py")
ICON = os.path.join(PROJECT_ROOT, "assets", "tiny-qpu-logo.ico")


def check_prerequisites():
    """Check that PyInstaller and required packages are available."""
    try:
        import PyInstaller
        print(f"  PyInstaller {PyInstaller.__version__} found")
    except ImportError:
        print("ERROR: PyInstaller not found. Install it:")
        print("  pip install pyinstaller")
        sys.exit(1)

    try:
        import flask
        print(f"  Flask {flask.__version__} found")
    except ImportError:
        print("ERROR: Flask not found. Install it:")
        print("  pip install flask")
        sys.exit(1)

    try:
        import numpy
        print(f"  NumPy {numpy.__version__} found")
    except ImportError:
        print("ERROR: NumPy not found. Install it:")
        print("  pip install numpy")
        sys.exit(1)

    if not os.path.exists(LAUNCHER):
        print(f"ERROR: Launcher not found: {LAUNCHER}")
        print("  Make sure tiny_qpu_launcher.py is in the project root.")
        sys.exit(1)

    if not os.path.exists(os.path.join(TEMPLATES_DIR, "index.html")):
        print(f"ERROR: Dashboard template not found: {TEMPLATES_DIR}/index.html")
        print("  Make sure the dashboard is installed.")
        sys.exit(1)

    print("  All prerequisites OK\n")


def build():
    """Build the executable."""
    print("=" * 56)
    print("  tiny-qpu Executable Builder")
    print("=" * 56)
    print()
    print("Checking prerequisites...")
    check_prerequisites()

    # Build data files list
    # We need to include the HTML template and static files
    datas = [
        (TEMPLATES_DIR, os.path.join("tiny_qpu", "dashboard", "templates")),
    ]

    # Include static dir if it exists and has files
    if os.path.exists(STATIC_DIR) and os.listdir(STATIC_DIR):
        datas.append((STATIC_DIR, os.path.join("tiny_qpu", "dashboard", "static")))

    # Build hidden imports (modules that PyInstaller might miss)
    hidden_imports = [
        "tiny_qpu",
        "tiny_qpu.circuit",
        "tiny_qpu.backends",
        "tiny_qpu.backends.statevector",
        "tiny_qpu.backends.density_matrix",
        "tiny_qpu.qasm",
        "tiny_qpu.qasm.parser",
        "tiny_qpu.dashboard",
        "tiny_qpu.dashboard.server",
        "flask",
        "jinja2",
        "werkzeug",
        "markupsafe",
        "numpy",
    ]

    # Build PyInstaller command
    cmd = [
        sys.executable, "-m", "PyInstaller",
        "--onefile",
        "--name", "tiny-qpu",
        "--paths", SRC_DIR,
        "--console",  # Keep console so user sees the server URL
        "--clean",
    ]

    # Add icon if available
    if os.path.exists(ICON):
        cmd.extend(["--icon", ICON])
        print(f"Using icon: {ICON}")

    # Add data files
    for src, dst in datas:
        sep = ";" if sys.platform == "win32" else ":"
        cmd.extend(["--add-data", f"{src}{sep}{dst}"])

    # Add hidden imports
    for imp in hidden_imports:
        cmd.extend(["--hidden-import", imp])

    # Add the launcher script
    cmd.append(LAUNCHER)

    print("Building executable...")
    print(f"  Command: {' '.join(cmd[:6])}...")
    print()

    result = subprocess.run(cmd, cwd=PROJECT_ROOT)

    if result.returncode != 0:
        print("\nBuild FAILED. Check the output above for errors.")
        sys.exit(1)

    # Check output
    exe_name = "tiny-qpu.exe" if sys.platform == "win32" else "tiny-qpu"
    exe_path = os.path.join(PROJECT_ROOT, "dist", exe_name)

    if os.path.exists(exe_path):
        size_mb = os.path.getsize(exe_path) / (1024 * 1024)
        print()
        print("=" * 56)
        print(f"  BUILD SUCCESSFUL!")
        print(f"  Output: {exe_path}")
        print(f"  Size: {size_mb:.1f} MB")
        print("=" * 56)
        print()
        print("To run:")
        print(f"  .\\dist\\{exe_name}")
        print()
        print("Or double-click the file in File Explorer.")
        print()
        print("Command-line options:")
        print(f"  .\\dist\\{exe_name} --port 9999")
        print(f"  .\\dist\\{exe_name} --no-browser")
    else:
        print(f"\nWARNING: Expected output not found at {exe_path}")
        print("Check the dist/ directory for the built file.")


if __name__ == "__main__":
    build()

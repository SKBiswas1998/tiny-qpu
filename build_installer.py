"""
╔══════════════════════════════════════════════════════╗
║   tiny-qpu — Full Windows Installer Build Pipeline  ║
╚══════════════════════════════════════════════════════╝

This script builds everything from source to final installer:

  Step 1: Generate logo assets (ICO, PNG, BMP)
  Step 2: Rebuild tiny-qpu.exe with embedded icon
  Step 3: Generate PDF user manual (if reportlab installed)
  Step 4: Compile Windows installer (if Inno Setup installed)

Usage:
    cd E:\\QPU\\tiny-qpu
    python build_installer.py

Prerequisites:
    pip install pyinstaller pillow

Optional (for PDF manual):
    pip install reportlab

Optional (for Windows installer — .exe setup):
    Download Inno Setup from https://jrsoftware.org/isdl.php
    Install it (default location is fine)
"""

import os
import sys
import subprocess
import shutil

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
ASSETS_DIR = os.path.join(PROJECT_ROOT, "assets")
DIST_DIR = os.path.join(PROJECT_ROOT, "dist")
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
DASHBOARD_DIR = os.path.join(SRC_DIR, "tiny_qpu", "dashboard")
TEMPLATES_DIR = os.path.join(DASHBOARD_DIR, "templates")
LAUNCHER = os.path.join(PROJECT_ROOT, "tiny_qpu_launcher.py")
ISS_FILE = os.path.join(PROJECT_ROOT, "tiny-qpu-setup.iss")
ICON_FILE = os.path.join(ASSETS_DIR, "tiny-qpu-logo.ico")
MANUAL_FILE = os.path.join(PROJECT_ROOT, "tiny_qpu_manual.pdf")

# ANSI colors for terminal output
class C:
    CYAN = "\033[96m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    END = "\033[0m"

def header(text):
    print(f"\n{C.CYAN}{C.BOLD}{'═' * 56}")
    print(f"  {text}")
    print(f"{'═' * 56}{C.END}\n")

def step(num, text):
    print(f"{C.BOLD}{C.CYAN}[Step {num}]{C.END} {text}")

def success(text):
    print(f"  {C.GREEN}✓{C.END} {text}")

def warn(text):
    print(f"  {C.YELLOW}⚠{C.END} {text}")

def error(text):
    print(f"  {C.RED}✗{C.END} {text}")

def skip(text):
    print(f"  {C.DIM}→ {text}{C.END}")


def check_prerequisites():
    """Check required and optional tools."""
    header("Checking Prerequisites")

    # Required: Python
    print(f"  Python: {sys.version.split()[0]}")

    # Required: PyInstaller
    try:
        import PyInstaller
        success(f"PyInstaller {PyInstaller.__version__}")
    except ImportError:
        error("PyInstaller not found. Run: pip install pyinstaller")
        sys.exit(1)

    # Required: Pillow
    try:
        from PIL import Image
        success(f"Pillow (PIL) available")
    except ImportError:
        error("Pillow not found. Run: pip install pillow")
        sys.exit(1)

    # Required: Flask
    try:
        import flask
        success("Flask available")
    except ImportError:
        error("Flask not found. Run: pip install flask")
        sys.exit(1)

    # Required: NumPy
    try:
        import numpy
        success("NumPy available")
    except ImportError:
        error("NumPy not found. Run: pip install numpy")
        sys.exit(1)

    # Required: Launcher script
    if os.path.exists(LAUNCHER):
        success("tiny_qpu_launcher.py found")
    else:
        error(f"tiny_qpu_launcher.py not found at {LAUNCHER}")
        sys.exit(1)

    # Required: Dashboard
    html_path = os.path.join(TEMPLATES_DIR, "index.html")
    if os.path.exists(html_path):
        success("Dashboard HTML found")
    else:
        error(f"Dashboard not installed: {html_path}")
        sys.exit(1)

    # Optional: reportlab (for PDF manual)
    has_reportlab = False
    try:
        import reportlab
        success("reportlab available (PDF manual will be generated)")
        has_reportlab = True
    except ImportError:
        warn("reportlab not found — skipping PDF manual (pip install reportlab)")

    # Optional: Inno Setup
    has_inno = False
    inno_path = find_inno_setup()
    if inno_path:
        success(f"Inno Setup found: {inno_path}")
        has_inno = True
    else:
        warn("Inno Setup not found — will skip installer compilation")
        warn("Download from: https://jrsoftware.org/isdl.php")

    return has_reportlab, has_inno, inno_path


def find_inno_setup():
    """Find Inno Setup compiler (iscc.exe) on Windows."""
    if sys.platform != "win32":
        return None

    # Check common install locations
    candidates = [
        os.path.join(os.environ.get("ProgramFiles(x86)", ""), "Inno Setup 6", "ISCC.exe"),
        os.path.join(os.environ.get("ProgramFiles", ""), "Inno Setup 6", "ISCC.exe"),
        os.path.join(os.environ.get("ProgramFiles(x86)", ""), "Inno Setup 5", "ISCC.exe"),
        os.path.join(os.environ.get("ProgramFiles", ""), "Inno Setup 5", "ISCC.exe"),
    ]

    for path in candidates:
        if path and os.path.exists(path):
            return path

    # Check PATH
    result = shutil.which("iscc")
    if result:
        return result

    return None


def step1_generate_icons():
    """Generate all logo/icon assets."""
    step(1, "Generating logo assets")

    create_icon_script = os.path.join(PROJECT_ROOT, "create_icon.py")
    if not os.path.exists(create_icon_script):
        error(f"create_icon.py not found at {create_icon_script}")
        return False

    result = subprocess.run([sys.executable, create_icon_script], cwd=PROJECT_ROOT)
    if result.returncode != 0:
        error("Icon generation failed")
        return False

    if os.path.exists(ICON_FILE):
        size = os.path.getsize(ICON_FILE)
        success(f"Icon: {ICON_FILE} ({size:,} bytes)")
        return True
    else:
        error(f"Icon file not created: {ICON_FILE}")
        return False


def step2_build_exe():
    """Build the standalone executable with embedded icon."""
    step(2, "Building tiny-qpu.exe with embedded icon")

    sep = ";" if sys.platform == "win32" else ":"

    cmd = [
        sys.executable, "-m", "PyInstaller",
        "--onefile",
        "--name", "tiny-qpu",
        "--paths", SRC_DIR,
        "--console",
        "--clean",
        "--noconfirm",
    ]

    # Add icon
    if os.path.exists(ICON_FILE):
        cmd.extend(["--icon", ICON_FILE])
        success(f"Using icon: {ICON_FILE}")

    # Add dashboard templates
    cmd.extend(["--add-data", f"{TEMPLATES_DIR}{sep}tiny_qpu{os.sep}dashboard{os.sep}templates"])

    # Add static dir if exists
    static_dir = os.path.join(DASHBOARD_DIR, "static")
    if os.path.exists(static_dir) and os.listdir(static_dir):
        cmd.extend(["--add-data", f"{static_dir}{sep}tiny_qpu{os.sep}dashboard{os.sep}static"])

    # Hidden imports
    hidden = [
        "tiny_qpu", "tiny_qpu.circuit", "tiny_qpu.backends",
        "tiny_qpu.backends.statevector", "tiny_qpu.backends.density_matrix",
        "tiny_qpu.qasm", "tiny_qpu.qasm.parser",
        "tiny_qpu.dashboard", "tiny_qpu.dashboard.server",
        "flask", "jinja2", "werkzeug", "markupsafe", "numpy",
    ]
    for h in hidden:
        cmd.extend(["--hidden-import", h])

    cmd.append(LAUNCHER)

    print(f"  Running PyInstaller...")
    result = subprocess.run(cmd, cwd=PROJECT_ROOT)

    exe_name = "tiny-qpu.exe" if sys.platform == "win32" else "tiny-qpu"
    exe_path = os.path.join(DIST_DIR, exe_name)

    if result.returncode == 0 and os.path.exists(exe_path):
        size_mb = os.path.getsize(exe_path) / (1024 * 1024)
        success(f"Executable: {exe_path} ({size_mb:.1f} MB)")
        return True
    else:
        error("PyInstaller build failed")
        return False


def step3_generate_manual(has_reportlab):
    """Generate PDF user manual."""
    step(3, "Generating PDF user manual")

    if not has_reportlab:
        skip("Skipping — reportlab not installed")
        return True

    gen_script = os.path.join(PROJECT_ROOT, "generate_manual.py")
    if not os.path.exists(gen_script):
        warn("generate_manual.py not found — skipping PDF")
        return True

    result = subprocess.run(
        [sys.executable, gen_script],
        cwd=PROJECT_ROOT,
    )

    if result.returncode == 0 and os.path.exists(MANUAL_FILE):
        size = os.path.getsize(MANUAL_FILE)
        success(f"Manual: {MANUAL_FILE} ({size:,} bytes)")
        return True
    else:
        warn("PDF generation failed — continuing without manual")
        return True


def step4_compile_installer(has_inno, inno_path):
    """Compile the Inno Setup installer."""
    step(4, "Compiling Windows installer")

    if not has_inno:
        skip("Skipping — Inno Setup not installed")
        skip("Download from: https://jrsoftware.org/isdl.php")
        skip("After installing, re-run this script to create the installer")
        return False

    if not os.path.exists(ISS_FILE):
        error(f"Inno Setup script not found: {ISS_FILE}")
        return False

    # Create output directory
    output_dir = os.path.join(PROJECT_ROOT, "installer_output")
    os.makedirs(output_dir, exist_ok=True)

    cmd = [inno_path, ISS_FILE]
    print(f"  Running Inno Setup compiler...")
    result = subprocess.run(cmd, cwd=PROJECT_ROOT)

    installer_path = os.path.join(output_dir, "tiny-qpu-setup.exe")

    if result.returncode == 0 and os.path.exists(installer_path):
        size_mb = os.path.getsize(installer_path) / (1024 * 1024)
        success(f"Installer: {installer_path} ({size_mb:.1f} MB)")
        return True
    else:
        error("Inno Setup compilation failed")
        return False


def main():
    header("tiny-qpu Windows Installer Builder")

    has_reportlab, has_inno, inno_path = check_prerequisites()

    # Step 1: Icons
    if not step1_generate_icons():
        error("Cannot continue without icons")
        sys.exit(1)

    # Step 2: EXE
    if not step2_build_exe():
        error("Cannot continue without executable")
        sys.exit(1)

    # Step 3: Manual (optional)
    step3_generate_manual(has_reportlab)

    # Step 4: Installer (optional)
    installer_built = step4_compile_installer(has_inno, inno_path)

    # ─── Summary ───
    header("Build Summary")

    exe_path = os.path.join(DIST_DIR, "tiny-qpu.exe" if sys.platform == "win32" else "tiny-qpu")
    installer_path = os.path.join(PROJECT_ROOT, "installer_output", "tiny-qpu-setup.exe")

    if os.path.exists(exe_path):
        size_mb = os.path.getsize(exe_path) / (1024 * 1024)
        success(f"Standalone EXE:  dist/tiny-qpu.exe ({size_mb:.1f} MB)")
    if os.path.exists(MANUAL_FILE):
        success(f"PDF Manual:      tiny_qpu_manual.pdf")
    if os.path.exists(ICON_FILE):
        success(f"Logo icon:       assets/tiny-qpu-logo.ico")

    if installer_built and os.path.exists(installer_path):
        size_mb = os.path.getsize(installer_path) / (1024 * 1024)
        success(f"Installer:       installer_output/tiny-qpu-setup.exe ({size_mb:.1f} MB)")
        print(f"\n  {C.BOLD}Distribute:{C.END} Share 'installer_output/tiny-qpu-setup.exe'")
        print(f"  Users double-click → Next → Next → Finish → Quantum Lab opens!")
    else:
        print(f"\n  {C.BOLD}Without Inno Setup:{C.END}")
        print(f"  Distribute 'dist/tiny-qpu.exe' directly — it's fully standalone.")
        print(f"\n  {C.BOLD}To create a proper installer:{C.END}")
        print(f"  1. Download Inno Setup: https://jrsoftware.org/isdl.php")
        print(f"  2. Install it (takes 1 minute)")
        print(f"  3. Re-run: python build_installer.py")

    print()


if __name__ == "__main__":
    main()

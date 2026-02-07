"""
Generate tiny-qpu logo as .ico, .png, and installer bitmaps.
Uses only Pillow — no SVG/cairo needed.

Run: python create_icon.py
Creates:
  assets/tiny-qpu-logo.ico   (multi-size Windows icon)
  assets/tiny-qpu-logo.png   (256x256 PNG)
  assets/installer_sidebar.bmp (164x314 Inno Setup wizard image)
  assets/installer_small.bmp  (55x58 Inno Setup small image)
"""

import math
import os
import sys

try:
    from PIL import Image, ImageDraw, ImageFont, ImageFilter
except ImportError:
    print("ERROR: Pillow is required.  pip install Pillow")
    sys.exit(1)


# ─── Colors ───
VOID   = (6, 10, 20)
DEEP   = (10, 14, 26)
PANEL  = (15, 20, 36)
CYAN   = (0, 212, 255)
PURPLE = (132, 94, 247)
AMBER  = (250, 176, 5)
GREEN  = (81, 207, 102)
RED    = (255, 107, 107)
BORDER = (30, 40, 72)
TEXT   = (232, 236, 244)
TEXTSEC = (107, 115, 148)
CYAN_DIM = (0, 100, 140)


def draw_logo(size=256):
    """Draw the tiny-qpu logo at the given size."""
    img = Image.new('RGBA', (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    cx, cy = size / 2, size / 2
    scale = size / 512

    # ─── Background rounded rectangle ───
    pad = int(12 * scale)
    radius = int(80 * scale)
    draw.rounded_rectangle([pad, pad, size - pad, size - pad],
                           radius=radius, fill=VOID)

    # ─── Draw orbit ellipses ───
    def draw_orbit(angle_deg, color, width, alpha=160):
        """Draw a tilted ellipse orbit ring."""
        # Create a temporary image, draw ellipse, rotate, composite
        tmp = Image.new('RGBA', (size, size), (0, 0, 0, 0))
        tmp_draw = ImageDraw.Draw(tmp)
        rx = int(155 * scale)
        ry = int(55 * scale)
        bbox = [cx - rx, cy - ry, cx + rx, cy + ry]
        col = color + (alpha,)
        tmp_draw.ellipse(bbox, outline=col, width=max(1, int(2 * scale)))
        rotated = tmp.rotate(angle_deg, center=(cx, cy), resample=Image.BICUBIC)
        img.paste(Image.alpha_composite(
            Image.new('RGBA', (size, size), (0, 0, 0, 0)), rotated),
            (0, 0), rotated)

    # Background orbits
    draw_orbit(-30, BORDER, 1, 100)
    draw_orbit(30, BORDER, 1, 100)
    draw_orbit(90, BORDER, 1, 100)

    # Glowing orbits (drawn over)
    draw_orbit(-30, CYAN, 2, 140)
    draw_orbit(30, PURPLE, 2, 120)
    draw_orbit(90, CYAN, 2, 100)

    # ─── Electron dots on orbits ───
    def orbit_point(angle_deg, orbit_angle_deg, rx, ry):
        """Get a point on a tilted ellipse."""
        t = math.radians(angle_deg)
        ox = rx * math.cos(t)
        oy = ry * math.sin(t)
        rot = math.radians(orbit_angle_deg)
        px = ox * math.cos(rot) - oy * math.sin(rot) + cx
        py = ox * math.sin(rot) + oy * math.cos(rot) + cy
        return px, py

    rx, ry = 155 * scale, 55 * scale
    dots = [
        (orbit_point(-35, -30, rx, ry), CYAN, int(8 * scale)),
        (orbit_point(200, 30, rx, ry), PURPLE, int(7 * scale)),
        (orbit_point(90, 90, rx, ry), GREEN, int(7 * scale)),
    ]

    # Re-get draw for composited image
    # Actually, let's draw dots after all orbits
    draw = ImageDraw.Draw(img)

    for (px, py), color, r in dots:
        # Glow
        for gr in range(r + int(6 * scale), r, -1):
            alpha = int(40 * (1 - (gr - r) / (6 * scale)))
            draw.ellipse([px - gr, py - gr, px + gr, py + gr],
                         fill=color + (alpha,))
        # Solid dot
        draw.ellipse([px - r, py - r, px + r, py + r], fill=color + (255,))

    # ─── Central Bloch sphere ───
    sphere_r = int(62 * scale)

    # Sphere body (dark fill)
    draw.ellipse([cx - sphere_r, cy - sphere_r, cx + sphere_r, cy + sphere_r],
                 fill=DEEP + (255,), outline=CYAN + (200,),
                 width=max(1, int(2.5 * scale)))

    # Equator ellipse
    eq_ry = int(18 * scale)
    draw.ellipse([cx - sphere_r, cy - eq_ry, cx + sphere_r, cy + eq_ry],
                 outline=CYAN + (80,), width=max(1, int(1 * scale)))

    # Meridian ellipse
    mer_rx = int(18 * scale)
    draw.ellipse([cx - mer_rx, cy - sphere_r, cx + mer_rx, cy + sphere_r],
                 outline=CYAN + (60,), width=max(1, int(1 * scale)))

    # |0⟩ pole (top) and |1⟩ pole (bottom)
    pole_r = int(4 * scale)
    draw.ellipse([cx - pole_r, cy - sphere_r + int(2 * scale) - pole_r,
                  cx + pole_r, cy - sphere_r + int(2 * scale) + pole_r],
                 fill=CYAN + (200,))
    draw.ellipse([cx - pole_r, cy + sphere_r - int(2 * scale) - pole_r,
                  cx + pole_r, cy + sphere_r - int(2 * scale) + pole_r],
                 fill=RED + (200,))

    # State vector arrow (pointing to superposition)
    arrow_x = cx + int(34 * scale)
    arrow_y = cy - int(46 * scale)

    # Arrow line
    draw.line([cx, cy, arrow_x, arrow_y],
              fill=CYAN + (220,), width=max(1, int(3 * scale)))

    # Arrow tip glow
    tip_r = int(6 * scale)
    for gr in range(tip_r + int(8 * scale), tip_r, -1):
        alpha = int(50 * (1 - (gr - tip_r) / (8 * scale)))
        draw.ellipse([arrow_x - gr, arrow_y - gr, arrow_x + gr, arrow_y + gr],
                     fill=CYAN + (alpha,))
    draw.ellipse([arrow_x - tip_r, arrow_y - tip_r,
                  arrow_x + tip_r, arrow_y + tip_r],
                 fill=CYAN + (255,))

    # ψ character in center (skip for very small icons)
    if size >= 32:
        font_size = max(8, int(28 * scale))
        try:
            font = ImageFont.truetype("arial.ttf", font_size)
        except (OSError, IOError):
            try:
                font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSerif-Italic.ttf", font_size)
            except (OSError, IOError):
                font = ImageFont.load_default()

        psi = "ψ"
        bbox = draw.textbbox((0, 0), psi, font=font)
        tw = bbox[2] - bbox[0]
        th = bbox[3] - bbox[1]
        draw.text((cx - tw / 2, cy - th / 2 + int(4 * scale)), psi,
                  fill=CYAN + (200,), font=font)

    # ─── Brand text (for larger sizes) ───
    if size >= 128:
        try:
            brand_size = max(8, int(32 * scale))
            brand_font = ImageFont.truetype("cour.ttf", brand_size)
        except (OSError, IOError):
            try:
                brand_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf", brand_size)
            except (OSError, IOError):
                brand_font = ImageFont.load_default()

        brand = "tiny-qpu"
        bbox = draw.textbbox((0, 0), brand, font=brand_font)
        tw = bbox[2] - bbox[0]
        brand_y = cy + int(140 * scale)
        draw.text((cx - tw / 2, brand_y), brand, fill=TEXT + (230,), font=brand_font)

        # Subtitle
        try:
            sub_size = max(8, int(12 * scale))
            sub_font = ImageFont.truetype("arial.ttf", sub_size)
        except (OSError, IOError):
            try:
                sub_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", sub_size)
            except (OSError, IOError):
                sub_font = ImageFont.load_default()

        sub = "QUANTUM LAB"
        bbox = draw.textbbox((0, 0), sub, font=sub_font)
        tw = bbox[2] - bbox[0]
        draw.text((cx - tw / 2, brand_y + int(30 * scale)), sub,
                  fill=TEXTSEC + (200,), font=sub_font)

    return img


def main():
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")
    os.makedirs(out_dir, exist_ok=True)

    print("Generating tiny-qpu logo assets...")

    # ─── Multi-size ICO ───
    ico_sizes = [16, 32, 48, 64, 128, 256]
    ico_images = []
    for s in ico_sizes:
        img = draw_logo(s)
        ico_images.append(img)
        print(f"  Generated {s}x{s} icon")

    ico_path = os.path.join(out_dir, "tiny-qpu-logo.ico")
    ico_images[-1].save(ico_path, format='ICO',
                        sizes=[(s, s) for s in ico_sizes],
                        append_images=ico_images[:-1])
    print(f"  ✓ {ico_path} ({os.path.getsize(ico_path):,} bytes)")

    # ─── 256px PNG ───
    png_path = os.path.join(out_dir, "tiny-qpu-logo.png")
    ico_images[-1].save(png_path, format='PNG')
    print(f"  ✓ {png_path} ({os.path.getsize(png_path):,} bytes)")

    # ─── 512px PNG ───
    logo_512 = draw_logo(512)
    png512_path = os.path.join(out_dir, "tiny-qpu-logo-512.png")
    logo_512.save(png512_path, format='PNG')
    print(f"  ✓ {png512_path} ({os.path.getsize(png512_path):,} bytes)")

    # ─── Inno Setup wizard sidebar (164x314) ───
    sidebar = Image.new('RGB', (164, 314), VOID)
    logo_small = draw_logo(140)
    # Convert RGBA to RGB for BMP
    bg = Image.new('RGBA', logo_small.size, VOID + (255,))
    composited = Image.alpha_composite(bg, logo_small)
    sidebar.paste(composited.convert('RGB'), (12, 60))
    sidebar_path = os.path.join(out_dir, "installer_sidebar.bmp")
    sidebar.save(sidebar_path, format='BMP')
    print(f"  ✓ {sidebar_path} ({os.path.getsize(sidebar_path):,} bytes)")

    # ─── Inno Setup small image (55x58) ───
    small_logo = draw_logo(48)
    small = Image.new('RGB', (55, 58), VOID)
    bg_s = Image.new('RGBA', small_logo.size, VOID + (255,))
    comp_s = Image.alpha_composite(bg_s, small_logo)
    small.paste(comp_s.convert('RGB'), (4, 5))
    small_path = os.path.join(out_dir, "installer_small.bmp")
    small.save(small_path, format='BMP')
    print(f"  ✓ {small_path} ({os.path.getsize(small_path):,} bytes)")

    print(f"\nAll assets saved to {out_dir}/")


if __name__ == "__main__":
    main()

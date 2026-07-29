"""
Convert a portrait image into a clean, monochrome ASCII-art SVG that "types"
itself in like a terminal, then holds.

Monochrome is deliberate -- per-character rainbow color is what makes ASCII
portraits look noisy. One fill color + a good density ramp + high contrast (so a
busy background washes out to blank) reads as neat and legible.

GitHub renders SVGs embedded via <img> and runs their SMIL animations there (JS
does not run). Each row is revealed with a left-to-right clip wipe plus a small
block cursor riding the wipe edge, staggered top -> bottom, so the whole
portrait prints once and freezes.
"""
import html
import os
import sys
from dataclasses import dataclass

from PIL import Image, ImageEnhance, ImageFilter, ImageStat

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_SRC = os.path.join(HERE, "..", "source-prepped.png")
FALLBACK_SRC = os.path.join(HERE, "..", "profile.jpeg")
SRC = sys.argv[1] if len(sys.argv) > 1 else (DEFAULT_SRC if os.path.exists(DEFAULT_SRC) else FALLBACK_SRC)
OUT = sys.argv[2] if len(sys.argv) > 2 else os.path.join(HERE, "..", "techin-ascii.svg")

COLS = 100
ROWS = 53
CELL_W = 8
CELL_H = 15
RAMP = " .'`^\",:;Il!i~+_-?][}{1)(|\\/tfjrxnuvczXYUJCLQ0OZmwqpdbkhao*#MW&8%B@$"
RESAMPLE = getattr(Image, "Resampling", Image).LANCZOS

PAD = 20
TITLEBAR_H = 30
STATUS_H = 30
ART_W = COLS * CELL_W
ART_H = ROWS * CELL_H
CANVAS_W = ART_W + PAD * 2
CANVAS_H = TITLEBAR_H + ART_H + STATUS_H + PAD

BG = "#0d1117"
BG2 = "#111722"
FRAME = "#30363d"
TITLE_TEXT = "#7d8590"
INK = "#10B981"
CURSOR = "#10B981"
HEADER = "techin@github: ~$ ./portrait.sh"
PROMPT = "techin@github:~$ whoami "
WHOAMI = "Techin Jetsribumrung"

# ---- reveal timing (one-shot; a cursor rasters top -> bottom) -------------
ROW_DUR = 0.11
STAGGER = 0.11       # == ROW_DUR -> a single cursor sweeping down


@dataclass
class Tuning:
    contrast: float
    gamma: float
    white_floor: float
    sharpen: bool


def clamp(value, low, high):
    return max(low, min(high, value))


def static_mode():
    return os.environ.get("STATIC", "").lower() not in ("", "0", "false", "no")


def has_useful_alpha(image):
    if "A" not in image.getbands():
        return False
    alpha = image.getchannel("A")
    transparent = sum(alpha.point(lambda px: 255 if px < 245 else 0).histogram()[1:]) / (alpha.width * alpha.height)
    return transparent > 0.02


def looks_like_clean_art(image):
    """Detect avatars/illustrations so crisp edges are not over-sharpened."""
    if has_useful_alpha(image):
        return True

    small = image.convert("RGB").resize((96, 96), RESAMPLE)
    color_count = len(small.quantize(colors=64).getcolors(maxcolors=96 * 96))
    gray = small.convert("L")
    edge_mean = ImageStat.Stat(gray.filter(ImageFilter.FIND_EDGES)).mean[0] / 255.0
    luma_std = ImageStat.Stat(gray).stddev[0]
    return color_count < 54 and (edge_mean > 0.055 or luma_std < 54)


def composite_to_luminance(image):
    if has_useful_alpha(image):
        base = Image.new("RGBA", image.size, (255, 255, 255, 255))
        image = Image.alpha_composite(base, image.convert("RGBA"))
    return image.convert("L")


def trim_white_border(image):
    """Crop only blank white margins introduced by preprocessing, not content."""
    threshold = 248
    mask = image.point(lambda px: 255 if px < threshold else 0)
    bbox = mask.getbbox()
    if not bbox:
        return image

    left, top, right, bottom = bbox
    width, height = image.size
    pad_x = max(2, round((right - left) * 0.04))
    pad_y = max(2, round((bottom - top) * 0.04))
    left = max(0, left - pad_x)
    top = max(0, top - pad_y)
    right = min(width, right + pad_x)
    bottom = min(height, bottom + pad_y)
    return image.crop((left, top, right, bottom))


def fit_to_cell_canvas(image):
    """Thumbnail onto a centered canvas while accounting for tall terminal cells."""
    src_w, src_h = image.size
    desired_grid_aspect = (src_w / src_h) * (CELL_H / CELL_W)
    available_grid_aspect = COLS / ROWS

    if desired_grid_aspect > available_grid_aspect:
        thumb_w = COLS
        thumb_h = max(1, round(COLS / desired_grid_aspect))
    else:
        thumb_h = ROWS
        thumb_w = max(1, round(ROWS * desired_grid_aspect))

    thumb = image.resize((thumb_w, thumb_h), RESAMPLE)
    canvas = Image.new("L", (COLS, ROWS), 255)
    canvas.paste(thumb, ((COLS - thumb_w) // 2, (ROWS - thumb_h) // 2))
    return canvas


def percentile_from_hist(hist, percentile):
    total = sum(hist)
    if total <= 0:
        return 255
    target = total * percentile
    running = 0
    for value, count in enumerate(hist):
        running += count
        if running >= target:
            return value
    return 255


def luminance_stats(image):
    hist = image.histogram()
    total = image.width * image.height
    active_hist = hist[:246]
    if sum(active_hist) < total * 0.05:
        active_hist = hist

    active_total = sum(active_hist)
    mean = sum(value * count for value, count in enumerate(active_hist)) / max(1, active_total)
    variance = sum(((value - mean) ** 2) * count for value, count in enumerate(active_hist)) / max(1, active_total)
    return {
        "std": variance ** 0.5,
        "p50": percentile_from_hist(active_hist, 0.50),
        "p90": percentile_from_hist(active_hist, 0.90),
        "bright_ratio": sum(hist[236:]) / total,
        "edge_mean": ImageStat.Stat(image.filter(ImageFilter.FIND_EDGES)).mean[0] / 255.0,
    }


def choose_tuning(image, clean_art):
    stats = luminance_stats(image)
    std_norm = stats["std"] / 255.0
    median = stats["p50"] / 255.0

    if clean_art:
        contrast = clamp(1.02 + max(0.0, 0.20 - std_norm) * 0.75, 0.96, 1.16)
        gamma = clamp(0.97 + (median - 0.56) * 0.35, 0.86, 1.08)
        white_floor = 0.92 if stats["bright_ratio"] > 0.25 else clamp((stats["p90"] / 255.0) + 0.04, 0.86, 0.94)
        sharpen = False
    else:
        contrast = clamp(1.08 + max(0.0, 0.25 - std_norm) * 1.55, 0.98, 1.42)
        gamma = clamp(0.92 + (median - 0.54) * 0.85, 0.72, 1.18)
        white_floor = 0.90 if stats["bright_ratio"] > 0.25 else clamp((stats["p90"] / 255.0) + 0.035, 0.82, 0.93)
        sharpen = stats["edge_mean"] < 0.075 or std_norm < 0.23

    return Tuning(contrast=contrast, gamma=gamma, white_floor=white_floor, sharpen=sharpen)


def apply_tuning(image, tuning):
    if tuning.sharpen:
        image = image.filter(ImageFilter.UnsharpMask(radius=1.2, percent=115, threshold=3))
    image = ImageEnhance.Contrast(image).enhance(tuning.contrast)
    lut = [round(255 * ((value / 255.0) ** tuning.gamma)) for value in range(256)]
    return image.point(lut)


def image_to_ascii_rows(image, white_floor):
    px = image.load()
    rows = []
    for y in range(ROWS):
        chars = []
        for x in range(COLS):
            lum = px[x, y] / 255.0
            if lum >= white_floor:
                chars.append(" ")
                continue
            idx = round((1.0 - lum) * (len(RAMP) - 1))
            chars.append(RAMP[int(clamp(idx, 0, len(RAMP) - 1))])
        rows.append("".join(chars))
    return rows


def row_text(line, y, font_size):
    safe = html.escape(line)
    return (
        f'<text xml:space="preserve" x="{PAD}" y="{y:.1f}" fill="{INK}" '
        f'font-size="{font_size:.1f}" textLength="{ART_W}" lengthAdjust="spacing">{safe}</text>'
    )


def append_animated_row(parts, row_index, line, frozen):
    y = art_top() + row_index * CELL_H + CELL_H * 0.74
    row_y = art_top() + row_index * CELL_H
    delay = row_index * STAGGER
    text = row_text(line, y, CELL_H * 0.86)

    if frozen:
        parts.append(text)
        return

    parts.append(
        f'<clipPath id="r{row_index}"><rect x="{PAD}" y="{row_y:.1f}" height="{CELL_H}" width="0">'
        f'<animate attributeName="width" from="0" to="{ART_W}" begin="{delay:.3f}s" '
        f'dur="{ROW_DUR:.2f}s" fill="freeze"/></rect></clipPath>'
    )
    parts.append(f'<g clip-path="url(#r{row_index})">{text}</g>')
    parts.append(
        f'<rect y="{row_y+1:.1f}" width="{CELL_W}" height="{CELL_H-2}" fill="{CURSOR}" opacity="0">'
        f'<animate attributeName="x" from="{PAD}" to="{PAD+ART_W}" begin="{delay:.3f}s" '
        f'dur="{ROW_DUR:.2f}s" fill="freeze"/>'
        f'<set attributeName="opacity" to="0.85" begin="{delay:.3f}s"/>'
        f'<set attributeName="opacity" to="0" begin="{delay+ROW_DUR:.3f}s"/></rect>'
    )


def art_top():
    return TITLEBAR_H + PAD * 0.35


def text_width(text, font_size):
    return len(text) * font_size * 0.62


def build_svg(rows, frozen):
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{CANVAS_W}" height="{CANVAS_H}" '
        f'viewBox="0 0 {CANVAS_W} {CANVAS_H}" font-family="ui-monospace, SFMono-Regular, '
        f'Menlo, Consolas, monospace">',
        '<defs>'
        f'<linearGradient id="bg" x1="0" y1="0" x2="0" y2="1">'
        f'<stop offset="0" stop-color="{BG2}"/><stop offset="1" stop-color="{BG}"/>'
        f'</linearGradient></defs>',
        f'<rect width="{CANVAS_W}" height="{CANVAS_H}" rx="12" fill="url(#bg)"/>',
        f'<rect x="0.5" y="0.5" width="{CANVAS_W-1}" height="{CANVAS_H-1}" rx="12" '
        f'fill="none" stroke="{FRAME}" stroke-width="1"/>',
        f'<line x1="0" y1="{TITLEBAR_H}" x2="{CANVAS_W}" y2="{TITLEBAR_H}" stroke="{FRAME}"/>',
    ]

    for i, dotcol in enumerate(["#ff5f56", "#ffbd2e", "#27c93f"]):
        parts.append(f'<circle cx="{PAD + i*16}" cy="{TITLEBAR_H/2}" r="5" fill="{dotcol}"/>')
    parts.append(
        f'<text x="{CANVAS_W/2}" y="{TITLEBAR_H/2 + 4}" fill="{TITLE_TEXT}" font-size="12" '
        f'text-anchor="middle">{html.escape(HEADER)}</text>'
    )

    for row_index, line in enumerate(rows):
        append_animated_row(parts, row_index, line, frozen)

    status_line_y = TITLEBAR_H + ART_H + PAD * 0.35
    status_y = status_line_y + 19
    footer_text = PROMPT + WHOAMI
    cursor_x = PAD + text_width(footer_text, 13) + 4
    parts.append(f'<line x1="0" y1="{status_line_y:.1f}" x2="{CANVAS_W}" y2="{status_line_y:.1f}" stroke="{FRAME}"/>')
    parts.append(
        f'<text x="{PAD}" y="{status_y:.1f}" fill="{TITLE_TEXT}" font-size="13">'
        f'{html.escape(PROMPT)}<tspan fill="{INK}">{html.escape(WHOAMI)}</tspan></text>'
    )
    parts.append(
        f'<rect x="{cursor_x:.1f}" y="{status_y-12:.1f}" width="8" height="14" fill="{CURSOR}">'
        f'<animate attributeName="opacity" values="1;1;0;0" keyTimes="0;0.5;0.51;1" '
        f'dur="1s" repeatCount="indefinite"/></rect>'
    )
    parts.append("</svg>")
    return "".join(parts)


source = Image.open(SRC)
clean_art = looks_like_clean_art(source)
luminance = trim_white_border(composite_to_luminance(source))
canvas = fit_to_cell_canvas(luminance)
tuning = choose_tuning(canvas, clean_art)
tuned = apply_tuning(canvas, tuning)
rows_txt = image_to_ascii_rows(tuned, tuning.white_floor)
svg = build_svg(rows_txt, static_mode())

with open(OUT, "w", encoding="utf-8") as f:
    f.write(svg)

kind = "clean-art" if clean_art else "photo"
print(
    "wrote",
    OUT,
    len(svg),
    "bytes;",
    f"{CANVAS_W}x{CANVAS_H};",
    kind,
    f"contrast={tuning.contrast:.2f}",
    f"gamma={tuning.gamma:.2f}",
    f"white_floor={tuning.white_floor:.2f}",
    f"sharpen={tuning.sharpen}",
)

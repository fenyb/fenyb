"""
Prepare an image for ASCII conversion by removing the background first.

The output stays the same size as the input and is saved as an RGBA PNG:
foreground pixels keep useful character detail, while the background becomes
transparent. make_ascii_svg.py can then composite that transparency onto white
and sample only the subject.

    python scripts/prep_photo.py input.png output.png
"""
import os
import sys

import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

HERE = os.path.dirname(os.path.abspath(__file__))
INP = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "..", "source-photo.jpg")
OUT = sys.argv[2] if len(sys.argv) > 2 else os.path.join(HERE, "..", "source-prepped.png")


def load_rgba(path):
    return Image.open(path).convert("RGBA")


def has_useful_alpha(image):
    alpha = np.array(image.getchannel("A"))
    transparent_ratio = np.count_nonzero(alpha < 245) / alpha.size
    return transparent_ratio > 0.02


def alpha_is_usable(alpha):
    visible_ratio = np.count_nonzero(alpha > 16) / alpha.size
    return 0.03 < visible_ratio < 0.98 and Image.fromarray(alpha).getbbox() is not None


def image_stats(rgb):
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 80, 180)
    edge_density = np.count_nonzero(edges) / edges.size
    sample = rgb.reshape(-1, 3)
    sample = sample[:: max(1, len(sample) // 20000)]
    quantized = (sample // 24).astype(np.uint8)
    unique_ratio = len(np.unique(quantized, axis=0)) / max(1, len(quantized))
    return {
        "edge_density": edge_density,
        "unique_ratio": unique_ratio,
        "luma_std": float(gray.std()),
    }


def looks_like_clean_art(rgb, alpha_source):
    if alpha_source:
        return True
    stats = image_stats(rgb)
    flat_colors = stats["unique_ratio"] < 0.14
    crisp_edges = stats["edge_density"] > 0.075
    low_texture = stats["luma_std"] < 62
    return flat_colors and (crisp_edges or low_texture)


def rembg_alpha(image):
    """Return a U2Net/rembg alpha mask, or None when the model is unavailable."""
    try:
        os.environ.setdefault("NUMBA_CACHE_DIR", os.path.join("/tmp", "numba-cache"))
        from rembg import remove

        cut = remove(image)
        alpha = np.array(cut.getchannel("A"))
        return alpha if alpha_is_usable(alpha) else None
    except Exception as exc:
        print("warning: rembg unavailable:", exc)
        return None


def checkerboard_background(rgb):
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]
    height, width = saturation.shape
    border = np.zeros((height, width), dtype=bool)
    band = max(8, round(min(height, width) * 0.05))
    border[:band, :] = True
    border[-band:, :] = True
    border[:, :band] = True
    border[:, -band:] = True

    samples = rgb[border & (saturation < 35) & (value > 170)]
    if len(samples) < width * 0.3:
        return None

    samples = samples.astype(np.float32)
    _, _, centers = cv2.kmeans(
        samples,
        2,
        None,
        (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 1.0),
        3,
        cv2.KMEANS_PP_CENTERS,
    )
    centers = centers.astype(np.float32)
    center_gap = np.linalg.norm(centers[0] - centers[1])
    if center_gap < 18:
        return None

    return centers


def checkerboard_masks(rgb, centers):
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]
    dist = np.min(
        np.sqrt(((rgb.astype(np.float32)[:, :, None, :] - centers[None, None, :, :]) ** 2).sum(axis=3)),
        axis=2,
    )
    checker_like = (dist < 24) & (saturation < 42) & (value > 165)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(checker_like.astype(np.uint8), 8)
    height, width = checker_like.shape
    edge_connected = np.zeros_like(checker_like)

    for label in range(1, count):
        x, y, w, h, _ = stats[label]
        touches_border = x == 0 or y == 0 or x + w >= width or y + h >= height
        if touches_border:
            edge_connected[labels == label] = True

    return checker_like, edge_connected


def grabcut_alpha(rgb):
    """Fallback segmentation when rembg cannot run."""
    height, width = rgb.shape[:2]
    margin_x = max(2, round(width * 0.04))
    margin_y = max(2, round(height * 0.04))
    rect = (margin_x, margin_y, width - margin_x * 2, height - margin_y * 2)
    mask = np.zeros((height, width), np.uint8)
    bg_model = np.zeros((1, 65), np.float64)
    fg_model = np.zeros((1, 65), np.float64)

    try:
        cv2.grabCut(rgb, mask, rect, bg_model, fg_model, 5, cv2.GC_INIT_WITH_RECT)
    except cv2.error as exc:
        print("warning: grabCut fallback failed:", exc)
        return None

    foreground = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
    return foreground if alpha_is_usable(foreground) else None


def anime_color_priors(rgb, alpha):
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    hue = hsv[:, :, 0]
    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]

    red_or_orange_hair = (
        (((hue <= 15) | (hue >= 168)) & (saturation > 45) & (value > 35))
        | ((hue <= 24) & (saturation > 65) & (value > 55))
    )
    skin = (hue <= 24) & (saturation > 25) & (saturation < 145) & (value > 65)
    light_clothing = (saturation < 75) & (value > 112) & (alpha > 35)
    return red_or_orange_hair, skin, light_clothing


def clean_art_grabcut_alpha(rgb, alpha):
    """Use rembg as a seed, then let GrabCut recover anime hair/outline."""
    hair, skin, clothing = anime_color_priors(rgb, alpha)
    confident_foreground = (alpha > 220) | hair | skin | clothing

    mask = np.full(alpha.shape, cv2.GC_BGD, dtype=np.uint8)
    mask[(alpha > 160) | hair | skin | clothing] = cv2.GC_PR_FGD
    mask[confident_foreground] = cv2.GC_FGD

    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]
    neutral_background = (saturation < 75) & (value < 190) & ~(hair | skin | clothing)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(neutral_background.astype(np.uint8), 8)
    height, width = alpha.shape

    for label in range(1, count):
        x, y, w, h, area = stats[label]
        touches_border = x == 0 or y == 0 or x + w >= width or y + h >= height
        if touches_border and area > 200:
            mask[(labels == label) & ~confident_foreground] = cv2.GC_BGD

    bg_model = np.zeros((1, 65), np.float64)
    fg_model = np.zeros((1, 65), np.float64)
    try:
        cv2.grabCut(rgb, mask, None, bg_model, fg_model, 5, cv2.GC_INIT_WITH_MASK)
    except cv2.error as exc:
        print("warning: seeded grabCut failed:", exc)
        return alpha

    foreground = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
    return foreground if alpha_is_usable(foreground) else alpha


def choose_background_fill(rgb):
    border = np.concatenate(
        [
            rgb[0, :, :],
            rgb[-1, :, :],
            rgb[:, 0, :],
            rgb[:, -1, :],
        ],
        axis=0,
    )
    luma = cv2.cvtColor(border.reshape(1, -1, 3), cv2.COLOR_RGB2GRAY).mean()
    return (255, 255, 255, 255) if luma < 128 else (0, 0, 0, 255)


def remove_large_background(rgb, alpha):
    """Drop low-confidence saturated background while keeping the subject mask."""
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]

    cleaned = alpha.copy()
    likely_background = (cleaned < 130) & (saturation > 115) & (gray < 180)
    cleaned[likely_background] = 0

    # Anime screenshots often have low-saturation walls/textures behind the
    # character. If rembg keeps those as one big edge-touching component, remove
    # that neutral region while leaving colorful hair/skin and inner line art.
    neutral = (cleaned > 16) & (saturation < 65) & (value < 195)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(neutral.astype(np.uint8), 8)
    height, width = neutral.shape

    for label in range(1, count):
        x, y, w, h, area = stats[label]
        touches_border = x == 0 or y == 0 or x + w >= width or y + h >= height
        if touches_border and area > 250:
            cleaned[labels == label] = 0

    soft_neutral = (cleaned > 16) & (cleaned < 210) & (saturation < 70) & (value < 155)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(soft_neutral.astype(np.uint8), 8)

    for label in range(1, count):
        _, _, w, h, area = stats[label]
        if area > 90 and w > 8 and h > 8:
            cleaned[labels == label] = 0

    return cleaned


def preserve_line_art(rgb, alpha):
    """Add back nearby black/gray anime strokes without resurrecting the whole bg."""
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    saturation = hsv[:, :, 1]

    base = alpha > 8
    seed = cv2.dilate(
        base.astype(np.uint8),
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25)),
        iterations=1,
    ).astype(bool)

    line_mask = (gray < 125) & (saturation < 90)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(line_mask.astype(np.uint8), 8)
    height, width = base.shape
    keep = np.zeros_like(base)

    for label in range(1, count):
        component = labels == label
        x, _, w, h, area = stats[label]
        if area < 8:
            continue
        touches_border = x == 0 or x + w >= width
        if touches_border and area > 120:
            continue
        touches_subject = np.any(seed[component])
        likely_character_stroke = area > 110 and h > 16 and w < width * 0.35 and x > width * 0.12
        if touches_subject or likely_character_stroke:
            keep |= component

    return np.maximum(alpha, keep.astype(np.uint8) * 255)


def refine_alpha(rgb, alpha, clean_art, skip_grabcut=False):
    if skip_grabcut:
        alpha = cv2.morphologyEx(
            alpha,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
        )
        return cv2.medianBlur(alpha, 3)

    if clean_art:
        if not skip_grabcut:
            alpha = clean_art_grabcut_alpha(rgb, alpha)
        alpha = remove_large_background(rgb, alpha)
        alpha = preserve_line_art(rgb, alpha)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    else:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    alpha = cv2.morphologyEx(alpha, cv2.MORPH_CLOSE, kernel)
    return cv2.medianBlur(alpha, 3)


def normalize_alpha(alpha, clean_art):
    if not clean_art:
        return alpha

    alpha_float = alpha.astype(np.float32) / 255.0
    alpha_float = np.clip((alpha_float - 0.08) / 0.72, 0.0, 1.0)
    alpha_float = np.power(alpha_float, 0.55)
    return np.clip(alpha_float * 255.0, 0, 255).astype(np.uint8)


def tune_foreground(image, clean_art):
    if clean_art:
        image = ImageEnhance.Contrast(image).enhance(1.05)
        return image.filter(ImageFilter.UnsharpMask(radius=0.9, percent=85, threshold=3))

    image = ImageEnhance.Contrast(image).enhance(1.08)
    return image.filter(ImageFilter.UnsharpMask(radius=1.2, percent=105, threshold=3))


def transparent_rgba(image, alpha, clean_art):
    rgb = tune_foreground(image.convert("RGB"), clean_art)
    out = rgb.convert("RGBA")
    out.putalpha(Image.fromarray(alpha, mode="L"))
    return out


source = load_rgba(INP)
source_rgb = np.array(source.convert("RGB"))
alpha_source = has_useful_alpha(source)
clean_art = looks_like_clean_art(source_rgb, alpha_source)
checker_centers = None if alpha_source else checkerboard_background(source_rgb)
checker_like = None

if alpha_source:
    alpha = np.array(source.getchannel("A"))
else:
    alpha = rembg_alpha(source)
    if alpha is None:
        alpha = grabcut_alpha(source_rgb)

if alpha is None:
    fallback = Image.new("RGBA", source.size, choose_background_fill(source_rgb))
    fallback.alpha_composite(source)
    fallback.save(OUT)
    print("wrote", OUT, fallback.size, "fallback-clean-background")
    sys.exit(0)

if checker_centers is not None:
    checker_like, checker_edge = checkerboard_masks(source_rgb, checker_centers)
    alpha[checker_edge] = 0

alpha = normalize_alpha(refine_alpha(source_rgb, alpha, clean_art, checker_centers is not None), clean_art)
if checker_like is not None:
    alpha[checker_edge] = 0
out = transparent_rgba(source, alpha, clean_art)
out.save(OUT)

kind = "clean-art" if clean_art else "photo"
visible = np.count_nonzero(alpha > 16) / alpha.size
print("wrote", OUT, out.size, kind, f"alpha_visible={visible:.2%}")

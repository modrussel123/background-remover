"""Utility functions for image handling and device detection."""

import os
import sys
from pathlib import Path
from typing import List, Tuple, Optional

from PIL import Image


SUPPORTED_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".webp", ".bmp",
    ".tiff", ".tif", ".gif", ".ico"
}


def get_supported_formats() -> List[str]:
    """Return list of supported image file extensions."""
    return sorted(SUPPORTED_EXTENSIONS)


def is_supported_format(file_path: str) -> bool:
    """Check if a file has a supported image extension."""
    return Path(file_path).suffix.lower() in SUPPORTED_EXTENSIONS


def get_image_files(directory: str, recursive: bool = False) -> List[Path]:
    """Get all supported image files from a directory.

    Args:
        directory: Path to the directory to scan.
        recursive: If True, scan subdirectories recursively.

    Returns:
        List of Path objects for supported image files.
    """
    dir_path = Path(directory)
    if not dir_path.is_dir():
        raise NotADirectoryError(f"Not a directory: {directory}")

    pattern = "**/*" if recursive else "*"
    files = []
    for ext in SUPPORTED_EXTENSIONS:
        files.extend(dir_path.glob(f"{pattern}{ext}"))
        files.extend(dir_path.glob(f"{pattern}{ext.upper()}"))

    return sorted(set(files))


def get_device_info() -> dict:
    """Detect available compute devices (CPU/GPU).

    Returns:
        Dictionary with device information.
    """
    info = {
        "cpu": True,
        "cuda_available": False,
        "cuda_device_count": 0,
        "cuda_device_name": None,
        "recommended_device": "cpu",
        "cuda_error": None,
    }

    try:
        import onnxruntime as ort
        providers = ort.get_available_providers()
        info["onnxruntime_providers"] = providers

        if "CUDAExecutionProvider" in providers:
            info["cuda_available"] = True
            info["recommended_device"] = "cuda"
            try:
                import torch
                if torch.cuda.is_available():
                    info["cuda_device_count"] = torch.cuda.device_count()
                    info["cuda_device_name"] = torch.cuda.get_device_name(0)
                else:
                    info["cuda_error"] = "PyTorch CUDA not available (info only; ONNX Runtime CUDA still works)"
            except ImportError:
                info["cuda_error"] = "PyTorch not installed (device name unavailable, but CUDA still usable)"
            except Exception as e:
                info["cuda_error"] = str(e)
        else:
            info["cuda_error"] = "CUDAExecutionProvider not in onnxruntime providers"
    except ImportError:
        info["onnxruntime_providers"] = ["CPUExecutionProvider"]
        info["cuda_error"] = "onnxruntime not installed"

    return info


def get_output_path(
    input_path: str,
    output_path: Optional[str] = None,
    output_dir: Optional[str] = None,
    suffix: str = "_nobg",
) -> Path:
    """Determine the output file path.

    Args:
        input_path: Original input file path.
        output_path: Explicit output file path (takes priority).
        output_dir: Output directory (if specified, keeps original filename).
        suffix: Suffix to add to filename when no output_path given.

    Returns:
        Resolved output Path.
    """
    input_p = Path(input_path)

    if output_path:
        out = Path(output_path)
        if out.is_dir():
            out = out / f"{input_p.stem}{suffix}.png"
        return out

    if output_dir:
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        return out_dir / f"{input_p.stem}{suffix}.png"

    return input_p.parent / f"{input_p.stem}{suffix}.png"


def create_checkerboard(width: int, height: int, box_size: int = 8, theme: str = "dark_small") -> Image.Image:
    """Create a checkerboard pattern image (useful for visualizing transparency).

    Args:
        width: Image width in pixels.
        height: Image height in pixels.
        box_size: Size of each checker square.
        theme: Background theme ('dark_small', 'dark_contrast', 'light', 'solid_dark', 'solid_white').

    Returns:
        RGBA PIL Image with checkerboard pattern.
    """
    if theme in ("dark_small", "dark", "Dark Check (Small)"):
        c1 = (20, 20, 20, 255)
        c2 = (42, 42, 42, 255)
    elif theme in ("dark_contrast", "Dark Check (Contrast)"):
        c1 = (12, 12, 12, 255)
        c2 = (62, 62, 62, 255)
    elif theme in ("solid_dark", "Solid Dark"):
        c1 = (22, 22, 22, 255)
        c2 = (22, 22, 22, 255)
    elif theme in ("solid_white", "Solid White"):
        c1 = (255, 255, 255, 255)
        c2 = (255, 255, 255, 255)
    else:  # light
        c1 = (255, 255, 255, 255)
        c2 = (204, 204, 204, 255)

    img = Image.new("RGBA", (width, height), c1)
    pixels = img.load()

    for y in range(height):
        for x in range(width):
            is_c1 = ((x // box_size) + (y // box_size)) % 2 == 0
            pixels[x, y] = c1 if is_c1 else c2

    return img


def refine_mask_quality(
    image: Image.Image,
    smooth: float = 1.2,
    defringe: bool = True,
    clean_specks: bool = True,
    choke: int = 0,
) -> Image.Image:
    """Refine alpha mask quality with antialiasing, defringing, and speck cleanup.

    Args:
        image: PIL RGBA image to refine.
        smooth: Edge smoothing / antialiasing radius (0 to 5.0).
        defringe: If True, decontaminates color fringing from previous background.
        clean_specks: If True, removes isolated 1-2px noise specks.
        choke: Pixels to contract (positive) or expand (negative) the mask edge.

    Returns:
        Refined PIL Image with smooth, defringed alpha channels.
    """
    import numpy as np
    import cv2

    if image.mode != "RGBA":
        image = image.convert("RGBA")

    arr = np.array(image, copy=True)
    h, w = arr.shape[:2]
    if h == 0 or w == 0:
        return image

    alpha = arr[:, :, 3]

    # 1. Clean small specks
    if clean_specks and np.any(alpha > 20):
        thresh = (alpha > 20).astype(np.uint8)
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(thresh, connectivity=8)
        for i in range(1, num_labels):
            if stats[i, cv2.CC_STAT_AREA] < 12:
                alpha[labels == i] = 0

    # 2. Choke / Expand mask boundary
    if choke != 0:
        ksize = abs(choke) * 2 + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
        if choke > 0:
            alpha = cv2.erode(alpha, kernel)
        else:
            alpha = cv2.dilate(alpha, kernel)

    # 3. Edge antialiasing / bilateral edge smoothing
    if smooth > 0 and np.any(alpha > 0) and np.any(alpha < 255):
        ksize = int(round(smooth * 2)) | 1
        if ksize >= 3:
            edges = cv2.Canny(alpha, 30, 150)
            dilated_edges = cv2.dilate(edges, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
            blurred_alpha = cv2.GaussianBlur(alpha, (ksize, ksize), smooth * 0.8)
            alpha = np.where(dilated_edges > 0, blurred_alpha, alpha)

    arr[:, :, 3] = alpha

    # 4. Defringe / Color decontamination
    if defringe:
        semi = (alpha > 0) & (alpha < 245)
        solid = alpha >= 245
        if np.any(semi) and np.any(solid):
            inpaint_mask = np.uint8(semi * 255)
            rgb = arr[:, :, :3]
            decontam_rgb = cv2.inpaint(rgb, inpaint_mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)
            arr[:, :, :3] = np.where(semi[:, :, None], decontam_rgb, rgb)

    return Image.fromarray(arr, mode="RGBA")



def format_size(size_bytes: int) -> str:
    """Format file size in human-readable format."""
    for unit in ["B", "KB", "MB", "GB"]:
        if size_bytes < 1024:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f} TB"


def get_file_info(file_path: str) -> dict:
    """Get information about an image file."""
    path = Path(file_path)

    info = {
        "path": str(path.absolute()) if path.exists() else str(path),
        "name": path.name,
        "extension": path.suffix.lower(),
        "size": 0,
        "size_formatted": "N/A",
        "width": None,
        "height": None,
        "mode": None,
        "format": None,
    }

    if not path.exists():
        return info

    try:
        stat = path.stat()
        info["size"] = stat.st_size
        info["size_formatted"] = format_size(stat.st_size)
    except OSError:
        pass

    try:
        with Image.open(path) as img:
            info["width"] = img.width
            info["height"] = img.height
            info["mode"] = img.mode
            info["format"] = img.format
    except Exception:
        pass

    return info

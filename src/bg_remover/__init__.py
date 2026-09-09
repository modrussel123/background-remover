"""Image Background Remover - High-quality background removal using AI."""

__version__ = "1.0.0"
__author__ = "BG Remover"

from bg_remover.core import BackgroundRemover
from bg_remover.utils import get_supported_formats, get_device_info

__all__ = ["BackgroundRemover", "get_supported_formats", "get_device_info"]

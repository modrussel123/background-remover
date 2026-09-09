"""Enhanced Background Remover Core - with caching and performance optimizations"""

import os
import sys
import time
import threading
from pathlib import Path
from typing import Optional, List, Callable, Union, Dict
from io import BytesIO

from PIL import Image

from bg_remover.utils import (
    get_image_files,
    get_output_path,
    get_device_info,
    get_supported_formats,
    create_checkerboard,
    format_size,
    get_file_info,
    refine_mask_quality,
)


AVAILABLE_MODELS = [
    "u2net",
    "u2netp",
    "u2net_human_seg",
    "u2net_cloth_seg",
    "silueta",
    "isnet-general-use",
    "isnet-anime",
    "birefnet-general",
    "birefnet-portrait",
    "birefnet-general-use",
    "birefnet-hr",
    "sam",
]

DEFAULT_MODEL = "birefnet-general"
FAST_MODEL = "u2netp"


class BackgroundRemover:
    """High-performance background removal engine with caching and GPU optimization."""

    _model_cache = {}
    _model_load_times = {}

    AVAILABLE_MODELS = AVAILABLE_MODELS
    DEFAULT_MODEL = DEFAULT_MODEL
    FAST_MODEL = FAST_MODEL

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        device: Optional[str] = None,
        alpha_matting: bool = False,
        alpha_matting_foreground_threshold: int = 240,
        alpha_matting_background_threshold: int = 10,
        alpha_matting_erode_size: int = 10,
        post_process_mask: bool = False,
        use_cache: bool = True,
        cache_timeout: int = 300,
        quality: str = "maximum",
        decontaminate: bool = False,
    ):
        """Initialize the background remover with performance optimizations.

        Args:
            model: Model name (e.g., 'birefnet-general', 'u2net').
            device: 'cuda' for GPU, 'cpu' for CPU, or None for auto-detect.
            alpha_matting: Enable alpha matting for better edge quality.
            alpha_matting_foreground_threshold: Foreground threshold for matting.
            alpha_matting_background_threshold: Background threshold for matting.
            alpha_matting_erode_size: Erosion size for matting.
            post_process_mask: Apply post-processing to clean up the mask.
            use_cache: Enable model caching to avoid repeated loading.
            cache_timeout: Cache timeout in seconds (default: 300 seconds/5 minutes).
            quality: Quality level ('maximum', 'high', 'standard').
            decontaminate: Remove background color fringing left on edges.
        """
        self.model_name = model
        self.alpha_matting = alpha_matting
        self.alpha_matting_foreground_threshold = alpha_matting_foreground_threshold
        self.alpha_matting_background_threshold = alpha_matting_background_threshold
        self.alpha_matting_erode_size = alpha_matting_erode_size
        self.post_process_mask = post_process_mask
        self.use_cache = use_cache
        self.cache_timeout = cache_timeout
        self.quality = quality.lower() if quality else "maximum"
        self.decontaminate = decontaminate

        self._device = self._resolve_device(device)
        self._session = None
        self._last_used = 0

    def _resolve_device(self, device: Optional[str]) -> str:
        """Resolve the compute device with auto-detection."""
        if device:
            return device.lower()

        info = get_device_info()
        return info["recommended_device"]

    def _setup_cuda_paths(self):
        """Add CUDA and cuDNN directories to PATH for GPU acceleration."""
        cuda_paths = [
            r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\bin",
            r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.7\bin",
            r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.5\bin",
        ]

        cuda_root = os.environ.get("CUDA_PATH", r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8")
        cuda_bin = os.path.join(cuda_root, "bin")
        if os.path.isdir(cuda_bin):
            cuda_paths.insert(0, cuda_bin)

        for path in cuda_paths:
            if os.path.isdir(path):
                current_path = os.environ.get("PATH", "")
                if path not in current_path:
                    os.environ["PATH"] = path + ";" + current_path

        cudnn_dir = os.path.join(sys.prefix, "Lib", "site-packages", "nvidia", "cudnn", "bin")
        cublas_bin = os.path.join(sys.prefix, "Lib", "site-packages", "nvidia", "cublas", "bin")

        for path in [cudnn_dir, cublas_bin]:
            if os.path.isdir(path):
                current_path = os.environ.get("PATH", "")
                if path not in current_path:
                    os.environ["PATH"] = path + ";" + current_path

    def _get_session(self):
        """Lazy-load the rembg session with caching and GPU support."""
        is_expired = (time.time() - self._last_used) > self.cache_timeout if self._last_used > 0 else False
        if self._session is None or (not self.use_cache) or is_expired:
            import rembg

            if self._device == "cuda":
                self._setup_cuda_paths()

            session_kwargs = {
                "model_name": self.model_name,
                "providers": ["CPUExecutionProvider"],
            }

            if self._device == "cuda":
                try:
                    import onnxruntime as ort
                    providers = ort.get_available_providers()
                    if "CUDAExecutionProvider" in providers:
                        session_kwargs["providers"] = ["CUDAExecutionProvider", "CPUExecutionProvider"]
                    else:
                        print("CUDAExecutionProvider not available, falling back to CPU", file=sys.stderr)
                        self._device = "cpu"
                except Exception as e:
                    print(f"CUDA provider check failed: {e}, falling back to CPU", file=sys.stderr)
                    self._device = "cpu"

            try:
                self._session = rembg.new_session(**session_kwargs)
            except Exception as error:
                if self._device != "cuda":
                    raise
                print(
                    f"CUDA session initialization failed: {error}, "
                    "falling back to CPU",
                    file=sys.stderr,
                )
                self._device = "cpu"
                session_kwargs["providers"] = ["CPUExecutionProvider"]
                self._session = rembg.new_session(**session_kwargs)
            active_providers = self._session.inner_session.get_providers()
            self._device = (
                "cuda"
                if "CUDAExecutionProvider" in active_providers
                else "cpu"
            )
            self._last_used = time.time()

        return self._session

    @property
    def active_device(self) -> str:
        """Return the compute device used by the current model session."""
        return self._device

    def remove_background(
        self,
        input_image: Union[str, Path, Image.Image],
        output_path: Optional[Union[str, Path]] = None,
        output_dir: Optional[Union[str, Path]] = None,
    ) -> Image.Image:
        """Remove background from a single image with performance optimizations.

        Args:
            input_image: File path (str/Path) or PIL Image object.
            output_path: Optional explicit output file path.
            output_dir: Optional output directory.

        Returns:
            PIL Image with transparent background (RGBA).
        """
        import rembg

        if isinstance(input_image, (str, Path)):
            img = Image.open(input_image)
            input_path = str(input_image)
        else:
            img = input_image
            input_path = "image"

        session = self._get_session()

        alpha_matting = self.alpha_matting
        post_process = self.post_process_mask
        erode_size = self.alpha_matting_erode_size
        fg_thresh = self.alpha_matting_foreground_threshold
        bg_thresh = self.alpha_matting_background_threshold

        if self.quality == "maximum":
            alpha_matting = True
            post_process = True
            erode_size = 6

        result = rembg.remove(
            img,
            session=session,
            alpha_matting=alpha_matting,
            alpha_matting_foreground_threshold=fg_thresh,
            alpha_matting_background_threshold=bg_thresh,
            alpha_matting_erode_size=erode_size,
            post_process_mask=post_process,
        )

        if result.mode != "RGBA":
            result = result.convert("RGBA")

        if self.quality == "maximum":
            result = refine_mask_quality(result, smooth=1.2, defringe=True, clean_specks=True)
        elif self.quality == "high":
            result = refine_mask_quality(result, smooth=0.8, defringe=False, clean_specks=True)

        if output_path or output_dir:
            out = get_output_path(
                input_path,
                str(output_path) if output_path else None,
                str(output_dir) if output_dir else None,
            )
            result.save(out, "PNG", optimize=True, compress_level=6)

        return result

    def remove_background_bytes(
        self,
        image_bytes: bytes,
        filename: str = "image",
    ) -> bytes:
        """Remove background from image bytes (useful for web APIs)."""
        import rembg

        img = Image.open(BytesIO(image_bytes))
        session = self._get_session()

        result = rembg.remove(img, session=session)

        if result.mode != "RGBA":
            result = result.convert("RGBA")

        buf = BytesIO()
        result.save(buf, format="PNG")
        return buf.getvalue()

    def batch_remove(
        self,
        input_dir: str,
        output_dir: Optional[str] = None,
        recursive: bool = False,
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
    ) -> List[Path]:
        """Remove background from all images in a directory with progress tracking."""
        if output_dir is None:
            output_dir = str(Path(input_dir) / "bg_removed")

        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        image_files = get_image_files(input_dir, recursive=recursive)
        if not image_files:
            return []

        total = len(image_files)
        results = []

        for idx, img_path in enumerate(image_files, 1):
            if progress_callback:
                progress_callback(idx, total, img_path.name)

            try:
                out_file = out_path / f"{img_path.stem}_nobg.png"
                self.remove_background(str(img_path), output_path=str(out_file))
                results.append(out_file)
            except Exception as e:
                print(f"Error processing {img_path.name}: {e}", file=sys.stderr)

        return results

    def get_model_info(self) -> dict:
        """Get information about the current model and session."""
        device_info = get_device_info()
        return {
            "model": self.model_name,
            "device": self._device,
            "alpha_matting": self.alpha_matting,
            "available_models": self.AVAILABLE_MODELS,
            "device_info": device_info,
        }

    @staticmethod
    def list_models() -> List[str]:
        """List all available models."""
        return AVAILABLE_MODELS.copy()

    def get_model_stats(self) -> dict:
        """Get model performance statistics."""
        return {
            "model": self.model_name,
            "device": self._device,
            "cache_enabled": self.use_cache,
            "cache_timeout": self.cache_timeout,
        }

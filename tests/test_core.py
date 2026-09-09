"""Tests for the background remover core module."""

import tempfile
from io import BytesIO
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from PIL import Image

from bg_remover.core import BackgroundRemover, AVAILABLE_MODELS, DEFAULT_MODEL, FAST_MODEL
from bg_remover.cli import create_parser, cmd_remove, cmd_info, cmd_models
from bg_remover.editing import erase_connected_color, get_drag_sample_points
from bg_remover.utils import (
    get_supported_formats,
    is_supported_format,
    get_image_files,
    get_device_info,
    get_output_path,
    create_checkerboard,
    format_size,
    get_file_info,
    refine_mask_quality,
)


class TestBackgroundRemover:
    """Tests for BackgroundRemover class."""

    def test_default_model(self):
        """Test that default model is set correctly."""
        remover = BackgroundRemover(device="cpu")
        assert remover.model_name == DEFAULT_MODEL

    def test_custom_model(self):
        """Test setting a custom model."""
        remover = BackgroundRemover(model="u2net", device="cpu")
        assert remover.model_name == "u2net"

    def test_device_resolution_auto(self):
        """Test auto device resolution."""
        remover = BackgroundRemover(device=None)
        assert remover._device in ("cpu", "cuda")

    def test_device_resolution_cpu(self):
        """Test explicit CPU device."""
        remover = BackgroundRemover(device="cpu")
        assert remover._device == "cpu"

    def test_list_models(self):
        """Test listing available models."""
        models = BackgroundRemover.list_models()
        assert isinstance(models, list)
        assert len(models) > 0
        assert DEFAULT_MODEL in models

    def test_get_model_info(self):
        """Test getting model information."""
        remover = BackgroundRemover(device="cpu")
        info = remover.get_model_info()
        assert "model" in info
        assert "device" in info
        assert "available_models" in info

    def test_get_model_stats(self):
        """Test getting model stats."""
        remover = BackgroundRemover(device="cpu")
        stats = remover.get_model_stats()
        assert stats["model"] == DEFAULT_MODEL
        assert stats["cache_enabled"] is True

    def test_session_caching_and_reuse(self):
        """Test that session is created once and reused when cache is valid."""
        remover = BackgroundRemover(device="cpu", use_cache=True, cache_timeout=300)
        with patch("rembg.new_session") as mock_new_session:
            mock_session_obj = MagicMock()
            mock_new_session.return_value = mock_session_obj

            s1 = remover._get_session()
            s2 = remover._get_session()

            assert s1 is mock_session_obj
            assert s2 is mock_session_obj
            assert mock_new_session.call_count == 1

    def test_session_cache_disabled(self):
        """Test that session is reloaded when use_cache is False."""
        remover = BackgroundRemover(device="cpu", use_cache=False)
        with patch("rembg.new_session") as mock_new_session:
            mock_new_session.side_effect = [MagicMock(), MagicMock()]

            s1 = remover._get_session()
            s2 = remover._get_session()

            assert mock_new_session.call_count == 2

    def test_remove_background_pil_image(self):
        """Test removing background from PIL Image."""
        remover = BackgroundRemover(device="cpu")
        img = Image.new("RGB", (100, 100), color=(255, 0, 0))

        with patch("rembg.new_session") as mock_session, patch("rembg.remove") as mock_remove:
            mock_session.return_value = MagicMock()
            mock_remove.return_value = Image.new("RGBA", (100, 100), (255, 0, 0, 255))
            result = remover.remove_background(img)

            assert result.mode == "RGBA"
            assert result.size == (100, 100)

    def test_remove_background_bytes(self):
        """Test removing background from image bytes."""
        remover = BackgroundRemover(device="cpu")
        img = Image.new("RGB", (50, 50), color=(0, 255, 0))

        buf = BytesIO()
        img.save(buf, format="PNG")
        img_bytes = buf.getvalue()

        with patch("rembg.new_session") as mock_session, patch("rembg.remove") as mock_remove:
            mock_session.return_value = MagicMock()
            mock_remove.return_value = Image.new("RGBA", (50, 50), (0, 255, 0, 255))
            result = remover.remove_background_bytes(img_bytes)

            assert isinstance(result, bytes)


class TestUtils:
    """Tests for utility functions."""

    def test_supported_formats(self):
        """Test getting supported formats."""
        formats = get_supported_formats()
        assert isinstance(formats, list)
        assert ".png" in formats
        assert ".jpg" in formats
        assert ".webp" in formats
        assert is_supported_format("image.jpg")
        assert is_supported_format("image.PNG")
        assert not is_supported_format("document.pdf")

    def test_device_info(self):
        """Test device detection."""
        info = get_device_info()
        assert "cpu" in info
        assert "cuda_available" in info
        assert info["cpu"] is True

    def test_get_output_path_default(self):
        """Test default output path generation."""
        path = get_output_path("photo.jpg")
        assert path.name == "photo_nobg.png"

    def test_get_output_path_explicit(self):
        """Test explicit output path."""
        path = get_output_path("photo.jpg", output_path="custom_result.png")
        assert path == Path("custom_result.png")

    def test_get_output_path_directory(self):
        """Test output to directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = get_output_path("photo.jpg", output_dir=tmpdir)
            assert path.name == "photo_nobg.png"
            assert Path(tmpdir) == path.parent

    def test_create_checkerboard(self):
        """Test checkerboard creation."""
        img = create_checkerboard(100, 100, box_size=10)
        assert img.mode == "RGBA"
        assert img.size == (100, 100)

    def test_create_checkerboard_dark_themes(self):
        """Test dark small checkerboard patterns for transparency visualization."""
        dark_img = create_checkerboard(50, 50, box_size=6, theme="dark_small")
        assert dark_img.mode == "RGBA"
        assert dark_img.size == (50, 50)
        # Check darkish pixel values (small black and dark gray/white)
        p1 = dark_img.getpixel((0, 0))
        p2 = dark_img.getpixel((6, 0))
        assert p1 != p2
        assert p1[0] < 50 and p2[0] < 50  # Darkish tones

    def test_erase_connected_color_stays_inside_circular_brush(self):
        """Test color erasing respects the circular brush boundary."""
        alpha = np.full((9, 9), 255, dtype=np.uint8)
        rgb = np.full((9, 9, 3), (20, 40, 60), dtype=np.uint8)
        rgb[4, 5] = (25, 45, 65)
        rgb[4, 6] = (26, 46, 66)

        erased = erase_connected_color(alpha, rgb, (4, 4), radius=2, tolerance=5)

        assert erased == 12
        assert alpha[4, 4] == 0
        assert alpha[4, 5] == 0
        assert alpha[4, 6] == 255
        assert alpha[4, 7] == 255

    def test_erase_connected_color_matches_magic_wand_connectivity(self):
        """Test disconnected matching colors are not erased together."""
        alpha = np.full((7, 7), 255, dtype=np.uint8)
        rgb = np.full((7, 7, 3), (200, 30, 20), dtype=np.uint8)
        rgb[:, 3] = (10, 200, 40)

        erased = erase_connected_color(alpha, rgb, (1, 3), radius=10, tolerance=0)

        assert erased == 21
        assert np.all(alpha[:, :3] == 0)
        assert np.all(alpha[:, 3:] == 255)

    def test_erase_connected_color_resamples_visible_edges(self):
        """Test dragging onto a new edge color erases that color too."""
        alpha = np.full((9, 9), 255, dtype=np.uint8)
        rgb = np.full((9, 9, 3), (30, 80, 180), dtype=np.uint8)
        rgb[:, 4] = (245, 245, 245)

        erase_connected_color(alpha, rgb, (2, 4), radius=2, tolerance=0)
        erased_edge = erase_connected_color(alpha, rgb, (4, 4), radius=3, tolerance=0)

        assert alpha[4, 2] == 0
        assert erased_edge == 7
        assert np.all(alpha[1:8, 4] == 0)

    def test_drag_sampling_captures_thin_color_transitions(self):
        """Test a fast drag still samples a one-pixel contour."""
        rgb = np.full((5, 11, 3), (30, 80, 180), dtype=np.uint8)
        rgb[:, 5] = (245, 245, 245)

        points = get_drag_sample_points(
            rgb,
            (0, 2),
            (10, 2),
            tolerance=20,
            spacing=6,
        )

        assert (5, 2) in points
        assert (6, 2) in points
        assert points[-1] == (10, 2)

    def test_erase_connected_color_follows_diagonal_contours(self):
        """Test diagonally connected line pixels erase as one region."""
        alpha = np.full((9, 9), 255, dtype=np.uint8)
        rgb = np.full((9, 9, 3), (30, 80, 180), dtype=np.uint8)
        diagonal = np.arange(9)
        rgb[diagonal, diagonal] = (245, 245, 245)

        erased = erase_connected_color(alpha, rgb, (4, 4), radius=4, tolerance=0)

        assert erased == 5
        assert np.all(alpha[diagonal[2:7], diagonal[2:7]] == 0)
        assert alpha[4, 3] == 255

    def test_refine_mask_quality(self):
        """Test edge smoothing, defringing, and stray noise cleanup."""
        img = Image.new("RGBA", (100, 100), (255, 100, 50, 255))
        # Add isolated 1-pixel speck and semitransparent edge
        pixels = img.load()
        pixels[10, 10] = (255, 255, 255, 255)
        refined = refine_mask_quality(img, smooth=1.2, defringe=True, clean_specks=True)
        assert refined.mode == "RGBA"
        assert refined.size == (100, 100)

    def test_remover_quality_modes(self):
        """Test BackgroundRemover quality modes initialization."""
        remover_max = BackgroundRemover(device="cpu", quality="maximum")
        assert remover_max.quality == "maximum"

        remover_std = BackgroundRemover(device="cpu", quality="standard")
        assert remover_std.quality == "standard"

    def test_format_size_bytes(self):
        """Test size formatting."""
        assert format_size(500) == "500.0 B"
        assert "KB" in format_size(1500)
        assert "MB" in format_size(1500000)

    def test_get_file_info(self):
        """Test getting file information."""
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            temp_name = f.name
            img = Image.new("RGB", (200, 100))
            img.save(temp_name)

        try:
            info = get_file_info(temp_name)
            assert info["width"] == 200
            assert info["height"] == 100
            assert info["extension"] == ".png"
        finally:
            Path(temp_name).unlink(missing_ok=True)

    def test_get_image_files(self):
        """Test scanning directory for images."""
        with tempfile.TemporaryDirectory() as tmpdir:
            p = Path(tmpdir)
            (p / "img1.png").touch()
            (p / "img2.jpg").touch()
            (p / "text.txt").touch()

            subdir = p / "sub"
            subdir.mkdir()
            (subdir / "img3.webp").touch()

            files = get_image_files(tmpdir, recursive=False)
            names = [f.name for f in files]
            assert "img1.png" in names
            assert "img2.jpg" in names
            assert "text.txt" not in names
            assert "img3.webp" not in names

            rec_files = get_image_files(tmpdir, recursive=True)
            rec_names = [f.name for f in rec_files]
            assert "img3.webp" in rec_names


class TestCLI:
    """Tests for CLI parser and commands."""

    def test_parser_creation(self):
        """Test CLI argument parser."""
        parser = create_parser()
        args = parser.parse_args(["remove", "photo.jpg", "-o", "out.png", "-f"])
        assert args.command == "remove"
        assert args.input == ["photo.jpg"]
        assert args.output == "out.png"
        assert args.fast is True

    def test_models_command(self, capsys):
        """Test models command."""
        parser = create_parser()
        args = parser.parse_args(["models"])
        ret = cmd_models(args)
        assert ret == 0
        captured = capsys.readouterr()
        assert "Available Models:" in captured.out
        assert DEFAULT_MODEL in captured.out

    def test_info_command(self, capsys):
        """Test info command."""
        parser = create_parser()
        args = parser.parse_args(["info"])
        ret = cmd_info(args)
        assert ret == 0
        captured = capsys.readouterr()
        assert "Background Remover - System Info" in captured.out

# Image Background Remover

High-quality AI-powered background removal tool with GPU acceleration.

## Features

- **State-of-the-art quality** using BiRefNet models
- **GPU acceleration** via ONNX Runtime CUDA (auto-detected)
- **CPU fallback** for systems without NVIDIA GPUs
- **CLI interface** for scripting and automation
- **GUI application** with drag-and-drop support
- **Batch processing** for entire folders of images
- **Alpha matting** for better edge quality on hair/fur
- **Multiple models** for different use cases

## Installation

### Basic (CPU only)

```bash
pip install -e .
```

### With GPU Support (NVIDIA)

```bash
pip install -e ".[gpu]"
```

### With GUI

```bash
pip install -e ".[gui]"
```

### Everything

```bash
pip install -e ".[all]"
```

## Usage

### CLI

```bash
# Single image
bgremover remove photo.jpg

# With output path
bgremover remove photo.jpg -o result.png

# Batch process a folder
bgremover remove ./photos/ -o ./output/

# Recursive batch processing
bgremover remove ./photos/ -r -o ./output/

# Force specific device
bgremover remove photo.jpg --device cuda
bgremover remove photo.jpg --device cpu

# Use specific model
bgremover remove photo.jpg -m birefnet-portrait

# Enable alpha matting for better edges
bgremover remove photo.jpg -a

# Show system info
bgremover info

# List available models
bgremover models
```

### GUI

```bash
bgremover gui
```

### Python API

```python
from bg_remover import BackgroundRemover

# Initialize with best quality model
remover = BackgroundRemover(model="birefnet-general")

# Remove background from file
result = remover.remove_background("photo.jpg", output_path="result.png")

# Remove background from PIL Image
from PIL import Image
img = Image.open("photo.jpg")
result = remover.remove_background(img)

# Batch process folder
results = remover.batch_remove("./photos/", output_dir="./output/")
```

## Available Models

| Model | Quality | Speed | Best For |
|-------|---------|-------|----------|
| `birefnet-general` | Best | Slower | General purpose |
| `birefnet-portrait` | Best | Slower | People/portraits |
| `birefnet-general-use` | Best | Slower | High-res images |
| `birefnet-hr` | Best | Slowest | Up to 2048x2048 |
| `u2net` | Good | Fast | General purpose |
| `u2netp` | Good | Fastest | Lightweight |
| `isnet-general-use` | Good | Fast | General objects |
| `isnet-anime` | Good | Fast | Anime/illustrations |
| `sam` | Good | Slow | Segmentation |

## Supported Input Formats

- JPEG/JPG
- PNG
- WebP
- BMP
- TIFF/TIF
- GIF
- ICO

## Output

All output is saved as **PNG with transparency** (RGBA). The alpha channel encodes the foreground mask, allowing the background to be transparent.

## Requirements

- Python >= 3.10
- rembg >= 2.0.81
- Pillow >= 10.0
- tqdm >= 4.66

### Optional

- `onnxruntime-gpu` for GPU acceleration
- `tkinterdnd2` for GUI drag-and-drop

## License

MIT

# Command-line interface for the background remover.
import argparse
import sys
import time
from pathlib import Path

from bg_remover.core import BackgroundRemover, AVAILABLE_MODELS, DEFAULT_MODEL, FAST_MODEL
from bg_remover.utils import get_device_info, get_file_info, format_size, get_supported_formats


def create_parser() -> argparse.ArgumentParser:
    """Create the argument parser."""
    parser = argparse.ArgumentParser(
        prog="bgremover",
        description="High-quality AI background remover with GPU acceleration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  bgremover remove photo.jpg                     Single image
  bgremover remove photo.jpg -o result.png       With output path
  bgremover remove ./photos/ -o ./output/        Batch folder
  bgremover remove ./photos/ -r                  Recursive batch
  bgremover remove ./photos/ -r -f               Fast batch processing
  bgremover gui                                  Launch GUI
  bgremover info                                 Show device/model info
  bgremover models                               List available models
        """,
    )
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Remove command
    remove_parser = subparsers.add_parser("remove", help="Remove background from image(s)")
    remove_parser.add_argument(
        "input",
        nargs="+",
        help="Input image file(s) or directory",
    )
    remove_parser.add_argument(
        "-o", "--output",
        help="Output file or directory",
    )
    remove_parser.add_argument(
        "-m", "--model",
        default=DEFAULT_MODEL,
        choices=AVAILABLE_MODELS,
        help=f"Model to use (default: {DEFAULT_MODEL})",
    )
    remove_parser.add_argument(
        "-d", "--device",
        choices=["cpu", "cuda", "auto"],
        default="auto",
        help="Compute device (default: auto-detect)",
    )
    remove_parser.add_argument(
        "-r", "--recursive",
        action="store_true",
        help="Process subdirectories recursively",
    )
    remove_parser.add_argument(
        "-a", "--alpha-matting",
        action="store_true",
        help="Enable alpha matting for better edges",
    )
    remove_parser.add_argument(
        "--fg-threshold",
        type=int,
        default=240,
        help="Alpha matting foreground threshold (default: 240)",
    )
    remove_parser.add_argument(
        "--bg-threshold",
        type=int,
        default=10,
        help="Alpha matting background threshold (default: 10)",
    )
    remove_parser.add_argument(
        "--erode-size",
        type=int,
        default=10,
        help="Alpha matting erode size (default: 10)",
    )
    remove_parser.add_argument(
        "--post-process",
        action="store_true",
        help="Apply post-processing to clean up mask",
    )
    remove_parser.add_argument(
        "-q", "--quiet",
        action="store_true",
        help="Suppress progress output",
    )
    remove_parser.add_argument(
        "-f", "--fast",
        action="store_true",
        help="Use fast model (u2netp) for better performance",
    )

    # GUI command
    subparsers.add_parser("gui", help="Launch graphical user interface")

    # Info command
    subparsers.add_parser("info", help="Show device and model information")

    # Models command
    subparsers.add_parser("models", help="List available models")

    return parser


def cmd_remove(args: argparse.Namespace) -> int:
    """Execute the remove command."""
    device = None if args.device == "auto" else args.device
    model = FAST_MODEL if args.fast else args.model

    remover = BackgroundRemover(
        model=model,
        device=device,
        alpha_matting=args.alpha_matting,
        alpha_matting_foreground_threshold=args.fg_threshold,
        alpha_matting_background_threshold=args.bg_threshold,
        alpha_matting_erode_size=args.erode_size,
        post_process_mask=args.post_process,
    )

    input_paths = []
    for inp in args.input:
        p = Path(inp)
        if p.is_dir():
            from bg_remover.utils import get_image_files
            input_paths.extend(get_image_files(str(p), recursive=args.recursive))
        elif p.is_file():
            input_paths.append(p)
        else:
            print(f"Warning: '{inp}' not found, skipping", file=sys.stderr)

    if not input_paths:
        print("No input images found.", file=sys.stderr)
        return 1

    total = len(input_paths)
    start_time = time.time()

    def progress(current, total, filename):
        if not args.quiet:
            elapsed = time.time() - start_time
            eta = (elapsed / current * (total - current)) if current > 0 else 0
            print(
                f"\r[{current}/{total}] {filename} "
                f"(elapsed: {elapsed:.1f}s, ETA: {eta:.1f}s)",
                end="",
                flush=True,
            )

    if total == 1:
        img_path = input_paths[0]
        out_path = args.output if args.output else str(get_output_path(str(img_path)))

        info = get_file_info(str(img_path))
        if not args.quiet:
            print(f"Processing: {img_path.name} ({info['width']}x{info['height']}, {info['size_formatted']})")

        try:
            result = remover.remove_background(str(img_path), output_path=out_path)
        except Exception as e:
            print(f"\nError processing image: {e}", file=sys.stderr)
            return 1

        elapsed = time.time() - start_time
        if not args.quiet:
            out_info = get_file_info(str(out_path))
            print(f"\nDone! Saved to: {out_path} ({out_info['size_formatted']}) [{elapsed:.2f}s]")

    else:
        output_dir = args.output or None
        results = []
        for idx, img_path in enumerate(input_paths, 1):
            progress(idx, total, img_path.name)
            try:
                if output_dir:
                    out_file = Path(output_dir) / f"{img_path.stem}_nobg.png"
                    Path(output_dir).mkdir(parents=True, exist_ok=True)
                else:
                    out_file = img_path.parent / f"{img_path.stem}_nobg.png"
                remover.remove_background(str(img_path), output_path=str(out_file))
                results.append(out_file)
            except Exception as e:
                print(f"\nError during batch processing {img_path.name}: {e}", file=sys.stderr)

        elapsed = time.time() - start_time
        if not args.quiet:
            print(f"\n\nDone! Processed {len(results)}/{total} images in {elapsed:.2f}s")
            if results:
                print(f"Output directory: {results[0].parent}")

    return 0


def cmd_gui(args: argparse.Namespace) -> int:
    """Execute the gui command."""
    try:
        from bg_remover.gui import run_gui
        run_gui()
        return 0
    except ImportError as e:
        print(f"GUI dependencies missing: {e}", file=sys.stderr)
        print("Install with: pip install tkinterdnd2", file=sys.stderr)
        return 1


def cmd_info(args: argparse.Namespace) -> int:
    """Execute the info command."""
    info = get_device_info()
    remover = BackgroundRemover()
    model_info = remover.get_model_info()

    print("=" * 50)
    print("Background Remover - System Info")
    print("=" * 50)
    print(f"\nModel: {model_info['model']}")
    print(f"Device: {model_info['device']}")
    print(f"Alpha Matting: {model_info['alpha_matting']}")
    print(f"\nDevice Detection:")
    print(f"  CPU: Available")
    print(f"  CUDA Available: {info['cuda_available']}")
    if info['cuda_available']:
        print(f"  CUDA Device Count: {info['cuda_device_count'] if info['cuda_device_count'] else 'N/A (install PyTorch for details)'}")
        if info['cuda_device_name']:
            print(f"  CUDA Device: {info['cuda_device_name']}")
    if info.get('cuda_error'):
        print(f"  CUDA Note: {info['cuda_error']}")
    print(f"\nONNX Runtime Providers: {info.get('onnxruntime_providers', 'N/A')}")
    print(f"\nSupported Formats: {', '.join(sorted(get_supported_formats()))}")

    return 0


def cmd_models(args: argparse.Namespace) -> int:
    """Execute the models command."""
    print("Available Models:")
    print("-" * 40)
    for model in AVAILABLE_MODELS:
        marker = " (default)" if model == DEFAULT_MODEL else ""
        print(f"  - {model}{marker}")

    print(f"\nRecommended: {DEFAULT_MODEL} (best quality)")
    return 0


def main(argv=None) -> int:
    """Main entry point."""
    parser = create_parser()
    args = parser.parse_args(argv)

    if not args.command:
        parser.print_help()
        return 1

    commands = {
        "remove": cmd_remove,
        "gui": cmd_gui,
        "info": cmd_info,
        "models": cmd_models,
    }

    handler = commands.get(args.command)
    if handler:
        return handler(args)

    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
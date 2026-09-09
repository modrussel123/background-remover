"""Desktop GUI for the background remover with drag-and-drop support."""

import os
import sys
import threading
from pathlib import Path
from typing import Optional, List, Tuple

try:
    import tkinter as tk
    from tkinter import ttk, filedialog, messagebox
    from PIL import Image, ImageTk, ImageDraw
    import numpy as np
    import cv2
except ImportError:
    raise ImportError(
        "tkinter is required for the GUI. "
        "Install with: pip install tk (or use your OS package manager)"
    )

try:
    from tkinterdnd2 import DND_FILES, TkinterDnD
    HAS_DND = True
except ImportError:
    HAS_DND = False

from bg_remover.core import BackgroundRemover, AVAILABLE_MODELS, DEFAULT_MODEL
from bg_remover.editing import erase_connected_color, get_drag_sample_points
from bg_remover.utils import (
    get_device_info,
    get_file_info,
    refine_mask_quality,
)


TOOL_ERASER = "eraser"
TOOL_RESTORER = "restorer"
TOOL_WAND = "wand"
TOOL_MAGIC_ERASER = "magic_eraser"
TOOL_COMPARE = "compare"
TOOL_PAN = "pan"

# Model descriptions shown in the info bar
MODEL_DESCRIPTIONS = {
    "u2net":              "U2-Net: Balanced quality/speed. Great general-purpose model.",
    "u2netp":             "U2-Net P: Lightweight fast version. Ideal for quick previews.",
    "u2net_human_seg":    "U2-Net Human: Optimized specifically for people/portraits.",
    "u2net_cloth_seg":    "U2-Net Cloth: Specialized for clothing and fashion images.",
    "silueta":            "Silueta: Lightweight model, good for simple foreground subjects.",
    "isnet-general-use":  "IS-Net General: High accuracy general model with fine edge detail.",
    "isnet-anime":        "IS-Net Anime: Tuned for anime/illustration artwork — great for cartoons.",
    "birefnet-general":   "BiRefNet General: State-of-the-art accuracy. Best for most photos. (Default)",
    "birefnet-portrait":  "BiRefNet Portrait: Fine-tuned for faces/people. Excellent hair detail.",
    "birefnet-general-use": "BiRefNet General-Use: Versatile high-res variant of BiRefNet.",
    "birefnet-hr":        "BiRefNet HR: Ultra high-resolution output. Slower but sharpest edges.",
    "sam":                "SAM (Segment Anything): Meta AI model. Extremely powerful, requires more VRAM.",
}

MIN_ZOOM = 0.05
MAX_ZOOM = 40.0


class BackgroundRemoverGUI:
    """Desktop GUI application for background removal with manual editing tools."""

    def __init__(self):
        if HAS_DND:
            self.root = TkinterDnD.Tk()
        else:
            self.root = tk.Tk()

        self.root.title("Background Remover")
        self.root.geometry("1280x860")
        self.root.minsize(1024, 700)

        self.remover = None
        self._remover_config = None
        self.current_image: Optional[Image.Image] = None
        self.original_image: Optional[Image.Image] = None
        self.ai_result_image: Optional[Image.Image] = None
        self.current_file: Optional[str] = None
        self.processed_files: List[Path] = []
        self.drop_label = None

        # Rendering & performance cache (for 60+ FPS butter-smooth pan and zoom)
        self._render_cache = {
            "orig": {"img_ref": None, "size": None, "photo": None, "item_id": None},
            "result": {"img_ref": None, "size": None, "photo": None, "item_id": None, "version": 0},
        }
        self._result_version = 0
        self._hq_debounce_id = None
        self._compare_cache = {"orig_rgb": None, "res_rgb": None, "size": None, "version": None}

        # Processing state & animated indicator
        self._is_processing = False
        self._spinner_angle = 0
        self._spinner_timer = None

        # Manual editing state
        self.selected_tool = tk.StringVar(value=TOOL_ERASER)
        self.brush_size = tk.IntVar(value=24)
        self.wand_tolerance = tk.IntVar(value=40)
        self._is_drawing = False
        self._last_xy: Optional[tuple] = None
        self._undo_stack: List[Image.Image] = []
        self._max_undo = 20
        self._wand_mode = False  # legacy compat flag
        # Fast in-memory array & redraw throttling for lag-free painting
        self._active_alpha: Optional[np.ndarray] = None
        self._active_rgb: Optional[np.ndarray] = None
        self._active_orig_rgb: Optional[np.ndarray] = None
        self._draw_redraw_pending: bool = False

        # Zoom + pan state (shared between Original + Result so they stay in sync)
        self._zoom_factor: float = 1.0
        self._pan_x: int = 0
        self._pan_y: int = 0
        self._zoom_label_var = tk.StringVar(value="100%")
        self._is_panning = False
        self._pan_start_xy: Optional[Tuple[int, int, int, int]] = None
        self._space_held = False
        self._alt_held = False

        # Compare overlay state
        self._compare_mode = False
        self._compare_handle_id = None
        self._compare_split_x = None
        self._compare_active_drag = False

        # Cursor overlay state (drawn directly on canvas for brush preview)
        self._cursor_canvas_pos: Optional[Tuple[int, int]] = None
        self._cursor_crosshair_ids: List[int] = []

        # Tk canvas images (keep references to prevent GC)
        self._photo_orig = None
        self._photo_result = None
        self._photo_compare = None

        self._setup_styles()
        self._create_widgets()
        self._setup_dnd()
        self._setup_bindings()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------
    def _setup_styles(self):
        style = ttk.Style()
        try:
            style.theme_use("clam")
        except Exception:
            pass

        self.root.configure(bg="#2b2b2b")
        style.configure("TFrame", background="#2b2b2b")
        style.configure("TLabel", background="#2b2b2b", foreground="#ffffff")
        style.configure("TButton", padding=5)
        style.configure("Tool.TButton", padding=(6, 4))
        style.configure(
            "Accent.TButton",
            background="#4a9eff",
            foreground="#ffffff",
            padding=(12, 8),
        )
        style.configure("Header.TLabel", font=("Segoe UI", 16, "bold"))
        style.configure("Status.TLabel", font=("Segoe UI", 9))
        style.configure("Section.TLabelframe.Label",
                        background="#2b2b2b", foreground="#cccccc")
        style.configure("WandActive.TButton",
                        background="#e06b3c", foreground="#ffffff",
                        padding=(8, 4))
        style.configure("Zoom.TButton", padding=(5, 2), width=4)

    def _create_widgets(self):
        main_frame = ttk.Frame(self.root, padding=8)
        main_frame.pack(fill=tk.BOTH, expand=True)

        # --- Header ---
        header = ttk.Label(main_frame, text="Background Remover", style="Header.TLabel")
        header.pack(pady=(0, 8))

        # --- Top toolbar: Model / Device / File actions ---
        top_frame = ttk.Frame(main_frame)
        top_frame.pack(fill=tk.X, pady=(0, 4))

        left_controls = ttk.Frame(top_frame)
        left_controls.pack(side=tk.LEFT, fill=tk.X, expand=True)

        ttk.Label(left_controls, text="Model:").pack(side=tk.LEFT, padx=(0, 5))
        self.model_var = tk.StringVar(value=DEFAULT_MODEL)
        self.model_combo = ttk.Combobox(
            left_controls,
            textvariable=self.model_var,
            values=AVAILABLE_MODELS,
            state="readonly",
            width=22,
        )
        self.model_combo.pack(side=tk.LEFT, padx=(0, 6))
        self.model_combo.bind("<<ComboboxSelected>>", self._on_model_change)

        ttk.Label(left_controls, text="Device:").pack(side=tk.LEFT, padx=(0, 5))
        self.device_var = tk.StringVar(value="auto")
        device_menu = ttk.Combobox(
            left_controls,
            textvariable=self.device_var,
            values=["auto", "cpu", "cuda"],
            state="readonly",
            width=8,
        )
        device_menu.pack(side=tk.LEFT, padx=(0, 6))

        # CUDA status indicator
        self.cuda_status_var = tk.StringVar(value="")
        self.cuda_status_lbl = ttk.Label(
            left_controls, textvariable=self.cuda_status_var,
            foreground="#aaaaaa", font=("Segoe UI", 8),
        )
        self.cuda_status_lbl.pack(side=tk.LEFT, padx=(0, 10))
        self._update_cuda_status()

        ttk.Label(left_controls, text="Quality:").pack(side=tk.LEFT, padx=(0, 4))
        self.quality_var = tk.StringVar(value="Maximum Quality")
        self.quality_combo = ttk.Combobox(
            left_controls,
            textvariable=self.quality_var,
            values=["Maximum Quality", "High Quality", "Standard (Fast)"],
            state="readonly",
            width=16,
        )
        self.quality_combo.pack(side=tk.LEFT, padx=(0, 8))
        self.quality_combo.bind("<<ComboboxSelected>>", self._on_quality_change)

        self.alpha_var = tk.BooleanVar(value=True)
        self.alpha_chk = ttk.Checkbutton(
            left_controls, text="Alpha Matting", variable=self.alpha_var
        )
        self.alpha_chk.pack(side=tk.LEFT, padx=(0, 8))

        self.post_var = tk.BooleanVar(value=True)
        self.post_chk = ttk.Checkbutton(
            left_controls, text="Post-process", variable=self.post_var
        )
        self.post_chk.pack(side=tk.LEFT)

        right_controls = ttk.Frame(top_frame)
        right_controls.pack(side=tk.RIGHT)

        self.btn_open = ttk.Button(
            right_controls, text="Open Image", command=self._open_file
        )
        self.btn_open.pack(side=tk.LEFT, padx=2)

        self.btn_folder = ttk.Button(
            right_controls, text="Open Folder", command=self._open_folder
        )
        self.btn_folder.pack(side=tk.LEFT, padx=2)

        self.btn_process = ttk.Button(
            right_controls,
            text="Process",
            command=self._process_image,
            style="Accent.TButton",
        )
        self.btn_process.pack(side=tk.LEFT, padx=2)

        self.btn_save = ttk.Button(
            right_controls, text="Save", command=self._save_image, state=tk.DISABLED
        )
        self.btn_save.pack(side=tk.LEFT, padx=2)

        # --- Edit Tools row: Tools + Brush + Wand Tolerance ---
        edit_frame = ttk.LabelFrame(main_frame, text=" Edit Tools ",
                                    style="Section.TLabelframe", padding=(6, 3))
        edit_frame.pack(fill=tk.X, pady=(0, 2))

        tools_row = ttk.Frame(edit_frame)
        tools_row.pack(fill=tk.X)

        ttk.Label(tools_row, text="Tool:").pack(side=tk.LEFT, padx=(0, 5))
        self.tool_eraser_btn = ttk.Radiobutton(
            tools_row, text="Eraser [X]", value=TOOL_ERASER,
            variable=self.selected_tool, style="Tool.TButton",
            command=self._on_tool_change,
        )
        self.tool_eraser_btn.pack(side=tk.LEFT, padx=2)
        self.tool_restorer_btn = ttk.Radiobutton(
            tools_row, text="Restorer [O]", value=TOOL_RESTORER,
            variable=self.selected_tool, style="Tool.TButton",
            command=self._on_tool_change,
        )
        self.tool_restorer_btn.pack(side=tk.LEFT, padx=2)
        self.tool_wand_btn = ttk.Radiobutton(
            tools_row, text="Magic Wand [W]", value=TOOL_WAND,
            variable=self.selected_tool, style="Tool.TButton",
            command=self._on_tool_change,
        )
        self.tool_wand_btn.pack(side=tk.LEFT, padx=2)
        self.tool_magic_eraser_btn = ttk.Radiobutton(
            tools_row, text="Magic Eraser [M]", value=TOOL_MAGIC_ERASER,
            variable=self.selected_tool, style="Tool.TButton",
            command=self._on_tool_change,
        )
        self.tool_magic_eraser_btn.pack(side=tk.LEFT, padx=2)
        self.tool_compare_btn = ttk.Radiobutton(
            tools_row, text="Compare [C]", value=TOOL_COMPARE,
            variable=self.selected_tool, style="Tool.TButton",
            command=self._on_tool_change,
        )
        self.tool_compare_btn.pack(side=tk.LEFT, padx=2)
        self.tool_pan_btn = ttk.Radiobutton(
            tools_row, text="Pan [H]", value=TOOL_PAN,
            variable=self.selected_tool, style="Tool.TButton",
            command=self._on_tool_change,
        )
        self.tool_pan_btn.pack(side=tk.LEFT, padx=2)

        ttk.Separator(tools_row, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=8)

        # Brush
        ttk.Label(tools_row, text="Brush:").pack(side=tk.LEFT, padx=(0, 4))
        self.brush_slider = ttk.Scale(
            tools_row, from_=1, to=300, orient=tk.HORIZONTAL,
            variable=self.brush_size, length=130,
            command=lambda _v: (self._on_brush_change(),
                                 self._redraw_cursor_overlay()),
        )
        self.brush_slider.pack(side=tk.LEFT, padx=(0, 3))
        self.brush_label = ttk.Label(tools_row, text="24px", width=5)
        self.brush_label.pack(side=tk.LEFT, padx=(0, 8))

        ttk.Separator(tools_row, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=8)

        # Wand tolerance
        ttk.Label(tools_row, text="Wand Tol:").pack(side=tk.LEFT, padx=(0, 4))
        self.wand_slider = ttk.Scale(
            tools_row, from_=1, to=200, orient=tk.HORIZONTAL,
            variable=self.wand_tolerance, length=100,
            command=lambda _v: self.wand_label.config(
                text=f"{int(self.wand_tolerance.get())}"),
        )
        self.wand_slider.pack(side=tk.LEFT, padx=(0, 3))
        self.wand_label = ttk.Label(tools_row, text="40", width=4)
        self.wand_label.pack(side=tk.LEFT)

        # ── ROW 3: View / Zoom / Bg + Actions ──────────────────────────────
        view_frame = ttk.LabelFrame(main_frame, text=" View & Actions ",
                                    style="Section.TLabelframe", padding=(6, 3))
        view_frame.pack(fill=tk.X, pady=(0, 4))

        view_row = ttk.Frame(view_frame)
        view_row.pack(fill=tk.X)

        # Zoom controls
        ttk.Label(view_row, text="Zoom:").pack(side=tk.LEFT, padx=(0, 4))
        self.btn_zoom_out = ttk.Button(
            view_row, text="−", style="Zoom.TButton",
            command=lambda: self._zoom_at(self._zoom_factor / 1.25, None),
        )
        self.btn_zoom_out.pack(side=tk.LEFT, padx=1)

        self.btn_zoom_label = ttk.Button(
            view_row, textvariable=self._zoom_label_var, width=6,
            command=self._zoom_reset,
        )
        self.btn_zoom_label.pack(side=tk.LEFT, padx=1)

        self.btn_zoom_in = ttk.Button(
            view_row, text="+", style="Zoom.TButton",
            command=lambda: self._zoom_at(self._zoom_factor * 1.25, None),
        )
        self.btn_zoom_in.pack(side=tk.LEFT, padx=1)

        self.btn_fit = ttk.Button(
            view_row, text="Fit", width=4,
            command=lambda: self._zoom_fit(),
        )
        self.btn_fit.pack(side=tk.LEFT, padx=(5, 1))

        self.btn_zoom_100 = ttk.Button(
            view_row, text="1:1", width=4,
            command=lambda: self._zoom_set(1.0),
        )
        self.btn_zoom_100.pack(side=tk.LEFT, padx=1)

        ttk.Separator(view_row, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=7)

        # Background pattern
        ttk.Label(view_row, text="Bg:").pack(side=tk.LEFT, padx=(0, 3))
        self.bg_pattern_var = tk.StringVar(value="Dark Check (Small)")
        self.bg_combo = ttk.Combobox(
            view_row,
            textvariable=self.bg_pattern_var,
            values=[
                "Dark Check (Small)",
                "Dark Check (Contrast)",
                "Light Check",
                "Solid Dark",
                "Solid White",
            ],
            state="readonly",
            width=19,
        )
        self.bg_combo.pack(side=tk.LEFT, padx=(0, 4))
        self.bg_combo.bind("<<ComboboxSelected>>", self._on_bg_pattern_change)

        ttk.Separator(view_row, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=7)

        # Action buttons: Undo / Reset Mask / Quality Edit
        self.btn_undo = ttk.Button(
            view_row, text="↩ Undo", command=self._undo, state=tk.DISABLED
        )
        self.btn_undo.pack(side=tk.LEFT, padx=2)

        self.btn_reset_mask = ttk.Button(
            view_row, text="Reset Mask", command=self._reset_mask,
            state=tk.DISABLED
        )
        self.btn_reset_mask.pack(side=tk.LEFT, padx=2)

        self.btn_quality_edit = ttk.Button(
            view_row,
            text="✨ Quality Edit",
            command=self._open_quality_edit_dialog,
            state=tk.DISABLED,
        )
        self.btn_quality_edit.pack(side=tk.LEFT, padx=2)

        # Hint (right side — only if space allows)
        ttk.Label(view_row,
                  text="Wheel=Zoom  Ctrl+Wheel=Brush  Space+Drag=Pan",
                  foreground="#9aa0a6", font=("Segoe UI", 8),
                  ).pack(side=tk.RIGHT, padx=(0, 4))

        # ── Model description info bar ──────────────────────────────────────
        info_row = ttk.Frame(main_frame)
        info_row.pack(fill=tk.X, pady=(0, 4))
        self.model_desc_var = tk.StringVar(
            value=MODEL_DESCRIPTIONS.get(DEFAULT_MODEL, "")
        )
        model_desc_lbl = ttk.Label(
            info_row, textvariable=self.model_desc_var,
            foreground="#7ab8f5", font=("Segoe UI", 8, "italic"),
        )
        model_desc_lbl.pack(side=tk.LEFT, padx=(4, 0))

        # --- Preview area: side-by-side original + result ---
        preview_frame = ttk.Frame(main_frame)
        preview_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 8))

        self.canvas_orig_frame = ttk.LabelFrame(preview_frame, text=" Original Image ",
                                                 style="Section.TLabelframe")
        self.canvas_orig_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True,
                                     padx=(0, 4))
        self.canvas_orig = tk.Canvas(
            self.canvas_orig_frame, bg="#1e1e1e", highlightthickness=1,
            highlightbackground="#3a3a3a",
        )
        self.canvas_orig.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)

        self.canvas_result_frame = ttk.LabelFrame(preview_frame,
                                                   text=" Result (Editable) ",
                                                   style="Section.TLabelframe")
        self.canvas_result_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True,
                                       padx=(4, 0))
        self.canvas_result = tk.Canvas(
            self.canvas_result_frame, bg="#1e1e1e", highlightthickness=1,
            highlightbackground="#3a3a3a",
        )
        self.canvas_result.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)

        if HAS_DND:
            drop_label = ttk.Label(
                self.canvas_result_frame,
                text="Drag & drop images here, or use Open buttons above",
                foreground="#888888",
                background="#1e1e1e",
            )
            self.drop_label = drop_label
            drop_label.place(relx=0.5, rely=0.5, anchor=tk.CENTER)

        # --- Status bar ---
        status_frame = ttk.Frame(main_frame)
        status_frame.pack(fill=tk.X)

        self.status_var = tk.StringVar(
            value="Ready. Wheel=Zoom / Ctrl+Wheel=Brush / Alt+Drag=Pan. "
                  "ESC cancels Wand or Compare."
        )
        self.status_label = ttk.Label(
            status_frame, textvariable=self.status_var, style="Status.TLabel"
        )
        self.status_label.pack(side=tk.LEFT)

        self.progress_var = tk.DoubleVar(value=0)
        self.progress_bar = ttk.Progressbar(
            status_frame, variable=self.progress_var, maximum=100
        )
        self.progress_bar.pack(side=tk.RIGHT, fill=tk.X, expand=True, padx=(10, 0))

        self._center_window()
        self._update_cursor_and_tool_highlight()
        self._update_zoom_label()

    def _setup_bindings(self):
        # --- Bindings on both canvases (zoom + pan are synchronized) ---
        for c in (self.canvas_orig, self.canvas_result):
            c.bind("<Configure>", self._on_canvas_configure)
            c.bind("<MouseWheel>", self._on_mousewheel)        # Windows/Linux
            c.bind("<Button-4>", lambda e, d=1: self._on_mousewheel(e, _override=d))
            c.bind("<Button-5>", lambda e, d=-1: self._on_mousewheel(e, _override=d))
            # Left click / drag / release
            c.bind("<ButtonPress-1>", self._on_mouse_down, add="+")
            c.bind("<B1-Motion>", self._on_mouse_drag, add="+")
            c.bind("<ButtonRelease-1>", self._on_mouse_up, add="+")
            # Middle mouse button pan
            c.bind("<ButtonPress-2>", self._on_middle_press)
            c.bind("<B2-Motion>", self._on_middle_drag)
            c.bind("<ButtonRelease-2>", self._on_mouse_release_any)

        # Result-only interactions
        cr = self.canvas_result
        cr.bind("<Motion>", self._on_mouse_move)
        cr.bind("<Leave>", self._on_mouse_leave)
        cr.bind("<Button-3>", self._on_right_click)

        self.canvas_orig.bind("<Motion>", lambda e: None)

        # Focus & modifier key bindings (safe against lost keyup on focus change)
        self.root.bind("<FocusOut>", self._on_focus_out)
        self.root.bind("<KeyPress-space>", self._on_space_down)
        self.root.bind("<KeyRelease-space>", self._on_space_up)
        self.root.bind("<Escape>", self._on_escape)
        self.root.bind("<KeyPress-Alt_L>", lambda e: setattr(self, "_alt_held", True))
        self.root.bind("<KeyRelease-Alt_L>", lambda e: setattr(self, "_alt_held", False))
        self.root.bind("<KeyPress-Alt_R>", lambda e: setattr(self, "_alt_held", True))
        self.root.bind("<KeyRelease-Alt_R>", lambda e: setattr(self, "_alt_held", False))

        # Ctrl+0/1/plus/minus zoom shortcuts (Photoshop-style)
        self.root.bind("<Control-Key-0>", lambda e: (self._zoom_fit(), "break"))
        self.root.bind("<Control-Key-1>", lambda e: (self._zoom_set(1.0), "break"))
        self.root.bind("<Control-plus>",
                        lambda e: (self._zoom_at(self._zoom_factor * 1.25, None),
                                    "break"))
        self.root.bind("<Control-equal>",
                        lambda e: (self._zoom_at(self._zoom_factor * 1.25, None),
                                    "break"))
        self.root.bind("<Control-minus>",
                        lambda e: (self._zoom_at(self._zoom_factor / 1.25, None),
                                    "break"))

    def _center_window(self):
        self.root.update_idletasks()
        w = self.root.winfo_width()
        h = self.root.winfo_height()
        x = (self.root.winfo_screenwidth() // 2) - (w // 2)
        y = (self.root.winfo_screenheight() // 2) - (h // 2)
        self.root.geometry(f"+{x}+{y}")

    # ------------------------------------------------------------------
    # Drag & drop
    # ------------------------------------------------------------------
    def _setup_dnd(self):
        if HAS_DND:
            for c in (self.canvas_orig, self.canvas_result):
                c.drop_target_register(DND_FILES)
                c.dnd_bind("<<Drop>>", self._on_drop)

    def _on_drop(self, event):
        files = self.root.tk.splitlist(event.data)
        if files:
            self._load_file(files[0])

    # ------------------------------------------------------------------
    # File dialogs
    # ------------------------------------------------------------------
    def _open_file(self):
        formats = " ".join(f"*{ext}" for ext in [
            ".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff"])
        file_path = filedialog.askopenfilename(
            title="Select Image",
            filetypes=[("Image files", formats), ("All files", "*.*")],
        )
        if file_path:
            self._load_file(file_path)

    def _open_folder(self):
        folder = filedialog.askdirectory(title="Select Folder with Images")
        if folder:
            self._process_folder(folder)

    def _load_file(self, file_path: str):
        try:
            info = get_file_info(file_path)
            self.current_file = file_path
            self.original_image = Image.open(file_path).convert("RGBA")
            self.current_image = None
            self.ai_result_image = None
            self._undo_stack.clear()
            self._update_undo_button()
            self._exit_compare_mode()
            # Reset wand tool back to eraser when loading new image
            if self.selected_tool.get() == TOOL_WAND:
                self.selected_tool.set(TOOL_ERASER)
                self._on_tool_change()
            self.btn_reset_mask.config(state=tk.DISABLED)
            self.btn_quality_edit.config(state=tk.DISABLED)
            self.btn_save.config(state=tk.DISABLED)
            self._result_version += 1
            self._invalidate_render_cache()
            # Fit new image to window
            self._zoom_pending_fit = True
            self._redraw_all()
            self._clear_cursor_overlay()

            if self.drop_label:
                self.drop_label.place_forget()

            self.status_var.set(
                f"Loaded: {info['name']} ({info['width']}x{info['height']}, "
                f"{info['size_formatted']}) — click Process"
            )
            self.btn_process.config(state=tk.NORMAL)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load image:\n{e}")

    _zoom_pending_fit = False

    # ------------------------------------------------------------------
    # AI processing
    # ------------------------------------------------------------------
    def _get_remover(self) -> BackgroundRemover:
        device = self.device_var.get()
        if device == "auto":
            device = None

        quality_val = self.quality_var.get()
        quality_map = {
            "Maximum Quality": "maximum",
            "High Quality": "high",
            "Standard (Fast)": "standard",
            "Standard": "standard",
        }
        quality_mode = quality_map.get(quality_val, "maximum")

        config = (
            self.model_var.get(),
            device,
            self.alpha_var.get(),
            self.post_var.get(),
            quality_mode,
        )

        if self.remover is None or self._remover_config != config:
            self.remover = BackgroundRemover(
                model=config[0],
                device=config[1],
                alpha_matting=config[2],
                post_process_mask=config[3],
                quality=config[4],
            )
            self._remover_config = config
        return self.remover

    def _on_quality_change(self, _event=None):
        """Update settings when quality dropdown is changed."""
        q = self.quality_var.get()
        if "Maximum" in q:
            self.alpha_var.set(True)
            self.post_var.set(True)
            self.status_var.set("Quality: Maximum — Sub-pixel edge antialiasing, defringing & precision alpha matting.")
        elif "High" in q:
            self.alpha_var.set(True)
            self.post_var.set(True)
            self.status_var.set("Quality: High — Standard post-processing & edge refinement enabled.")
        else:
            self.alpha_var.set(False)
            self.post_var.set(False)
            self.status_var.set("Quality: Standard — Fast inference without extra edge refinement.")
        self._remover_config = None

    def _on_bg_pattern_change(self, _event=None):
        """Refresh canvas when transparent checkerboard pattern changes."""
        self._invalidate_render_cache()
        self._redraw_all()
        self.status_var.set(f"Background pattern set to: {self.bg_pattern_var.get()}")

    # ------------------------------------------------------------------
    # CUDA status + model description helpers
    # ------------------------------------------------------------------
    def _update_cuda_status(self):
        """Check CUDA/GPU availability and update the status label."""
        def _check():
            try:
                import onnxruntime as ort
                providers = ort.get_available_providers()
                if "CUDAExecutionProvider" in providers:
                    self.root.after(0, lambda: (
                        self.cuda_status_var.set("GPU: CUDA ready"),
                        self.cuda_status_lbl.config(foreground="#4aff8a"),
                    ))
                else:
                    self.root.after(0, lambda: (
                        self.cuda_status_var.set("GPU: CPU only"),
                        self.cuda_status_lbl.config(foreground="#ffaa44"),
                    ))
            except Exception:
                self.root.after(0, lambda: (
                    self.cuda_status_var.set("GPU: unavailable"),
                    self.cuda_status_lbl.config(foreground="#ff5555"),
                ))
        import threading
        threading.Thread(target=_check, daemon=True).start()

    def _on_model_change(self, _event=None):
        """Update model description label when user picks a different model."""
        model = self.model_var.get()
        desc = MODEL_DESCRIPTIONS.get(model, "")
        self.model_desc_var.set(desc)

    def _start_processing_indicator(self, message: str = "Removing background with AI..."):
        """Show active animated spinner and indeterminate progress bar."""
        self._is_processing = True
        self.btn_process.config(text="⏳ Processing...", state=tk.DISABLED)
        self.btn_save.config(state=tk.DISABLED)
        self.btn_reset_mask.config(state=tk.DISABLED)
        self.btn_quality_edit.config(state=tk.DISABLED)
        self.btn_open.config(state=tk.DISABLED)
        self.btn_folder.config(state=tk.DISABLED)
        self.status_var.set(message)

        self.progress_bar.config(mode="indeterminate")
        self.progress_bar.start(10)

        self.root.config(cursor="watch")
        self.canvas_orig.config(cursor="watch")
        self.canvas_result.config(cursor="watch")

        self._spinner_angle = 0
        self._animate_processing_overlay()

    def _stop_processing_indicator(self, success: bool = True):
        """Stop processing indicator and restore UI controls."""
        self._is_processing = False
        if self._spinner_timer is not None:
            try:
                self.root.after_cancel(self._spinner_timer)
            except Exception:
                pass
            self._spinner_timer = None

        self.canvas_result.delete("loading_overlay")
        self.progress_bar.stop()
        self.progress_bar.config(mode="determinate")
        self.progress_var.set(100 if success else 0)

        self.btn_process.config(text="Process", state=tk.NORMAL)
        self.btn_open.config(state=tk.NORMAL)
        self.btn_folder.config(state=tk.NORMAL)
        if success and self.current_image:
            self.btn_save.config(state=tk.NORMAL)
            self.btn_reset_mask.config(state=tk.NORMAL)
            self.btn_quality_edit.config(state=tk.NORMAL)

        self.root.config(cursor="")
        self._update_cursor_and_tool_highlight()

    def _animate_processing_overlay(self):
        """Draw animated AI processing badge with rotating ring on canvas_result."""
        if not self._is_processing:
            return

        c = self.canvas_result
        c.delete("loading_overlay")

        cw = max(200, c.winfo_width())
        ch = max(150, c.winfo_height())
        cx = cw // 2
        cy = ch // 2

        # Card dimensions
        bw, bh = 340, 130
        bx0, by0 = cx - bw // 2, cy - bh // 2
        bx1, by1 = cx + bw // 2, cy + bh // 2

        # Card shadow / background
        c.create_rectangle(
            bx0, by0, bx1, by1,
            fill="#1e222b", outline="#4a9eff", width=2,
            tags="loading_overlay",
        )

        # Spinner ring
        sr = 18
        scx = cx
        scy = cy - 20
        c.create_oval(
            scx - sr, scy - sr, scx + sr, scy + sr,
            outline="#2f3846", width=4,
            tags="loading_overlay",
        )
        c.create_arc(
            scx - sr, scy - sr, scx + sr, scy + sr,
            start=self._spinner_angle, extent=100,
            outline="#4a9eff", width=4, style="arc",
            tags="loading_overlay",
        )

        # Title text
        c.create_text(
            cx, cy + 18,
            text="⚡ AI Processing Active...",
            fill="#ffffff", font=("Segoe UI", 11, "bold"),
            tags="loading_overlay",
        )

        # Subtitle with model and device
        model_name = self.model_var.get()
        dev = self.device_var.get().upper()
        c.create_text(
            cx, cy + 38,
            text=f"Model: {model_name} • Device: {dev}",
            fill="#8fa0b5", font=("Segoe UI", 9),
            tags="loading_overlay",
        )

        self._spinner_angle = (self._spinner_angle + 18) % 360
        self._spinner_timer = self.root.after(35, self._animate_processing_overlay)

    def _process_image(self):
        if not self.current_file:
            return

        self._exit_compare_mode()
        # If compare tool was active, switch back to eraser
        if self.selected_tool.get() == TOOL_COMPARE:
            self.selected_tool.set(TOOL_ERASER)
            self._on_tool_change()
        self._start_processing_indicator("Processing with AI...")

        def worker():
            try:
                remover = self._get_remover()
                out = remover.remove_background(self.current_file)
                self.current_image = out.convert("RGBA")
                self.ai_result_image = self.current_image.copy()
                self._undo_stack.clear()
                self._result_version += 1
                self.root.after(0, lambda: self._on_process_complete())
            except Exception as e:
                self.root.after(0, lambda: self._on_process_error(str(e)))

        threading.Thread(target=worker, daemon=True).start()

    def _on_process_complete(self):
        self._stop_processing_indicator(success=True)
        self._invalidate_render_cache("result")
        self._redraw_all()
        self.status_var.set(
            "AI done! Use Eraser / Wand to adjust, or click ✨ Quality Edit to refine edges at Maximum Quality."
        )

    def _on_process_error(self, error: str):
        self._stop_processing_indicator(success=False)
        self.status_var.set(f"Error: {error}")
        messagebox.showerror("Processing Error",
                             f"Failed to remove background:\n{error}")

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    def _save_image(self):
        if not self.current_image:
            return

        default_name = (Path(self.current_file).stem + "_nobg.png"
                        if self.current_file else "output.png")

        file_path = filedialog.asksaveasfilename(
            title="Save Processed Image",
            defaultextension=".png",
            initialfile=default_name,
            filetypes=[("PNG files", "*.png"), ("All files", "*.*")],
        )

        if file_path:
            try:
                self.current_image.save(file_path, "PNG", optimize=True, compress_level=6)
                self.status_var.set(f"Saved (Maximum Quality Lossless): {file_path}")
                messagebox.showinfo("Saved", f"Image saved at maximum quality to:\n{file_path}")
            except Exception as e:
                messagebox.showerror("Save Error", f"Failed to save:\n{e}")

    # ------------------------------------------------------------------
    # Folder batch
    # ------------------------------------------------------------------
    def _process_folder(self, folder: str):
        self._start_processing_indicator("Processing folder...")

        def worker():
            try:
                remover = self._get_remover()
                from bg_remover.utils import get_image_files
                image_files = get_image_files(folder)
                total = len(image_files)

                if total == 0:
                    self.root.after(0, lambda: (
                        self._stop_processing_indicator(success=False),
                        messagebox.showinfo("No Images", "No supported images found in the folder.")
                    ))
                    return

                results = []
                for idx, img_path in enumerate(image_files, 1):
                    self.root.after(
                        0,
                        lambda i=idx, n=total, name=img_path.name: (
                            self.status_var.set(f"Processing {i}/{n}: {name}"),
                            self.progress_var.set(i / n * 100),
                        ),
                    )
                    try:
                        out_dir = Path(folder) / "bg_removed"
                        out_dir.mkdir(exist_ok=True)
                        out_path = out_dir / f"{img_path.stem}_nobg.png"
                        remover.remove_background(str(img_path),
                                                   output_path=str(out_path))
                        results.append(out_path)
                    except Exception as e:
                        print(f"Error: {img_path.name}: {e}", file=sys.stderr)

                self.processed_files = results
                self.root.after(0, lambda: self._on_batch_complete(results))
            except Exception as e:
                self.root.after(0, lambda: self._on_process_error(str(e)))

        threading.Thread(target=worker, daemon=True).start()

    def _on_batch_complete(self, results: List[Path]):
        self._stop_processing_indicator(success=True)

        if results:
            out_dir = results[0].parent
            self.status_var.set(f"Done! {len(results)} images saved to: {out_dir}")
            messagebox.showinfo(
                "Batch Complete",
                f"Successfully processed {len(results)} images.\n\nOutput: {out_dir}",
            )
        else:
            self.status_var.set("No images were processed.")

    # ==================================================================
    # ZOOM + PAN ENGINE
    # ==================================================================
    def _update_zoom_label(self):
        pct = int(round(self._zoom_factor * 100))
        self._zoom_label_var.set(f"{pct}%")

    def _clamp_zoom(self, z: float) -> float:
        return max(MIN_ZOOM, min(MAX_ZOOM, float(z)))

    def _canvas_for_event(self, event_source) -> Optional[tk.Canvas]:
        if event_source is self.canvas_orig or event_source is self.canvas_result:
            return event_source
        return self.canvas_result

    def _zoom_set(self, factor: float):
        """Set zoom to an exact factor, centered on the canvas midpoint."""
        self._zoom_at(factor, None)

    def _zoom_at(self, factor: float,
                 canvas_center: Optional[Tuple[tk.Canvas, int, int]]):
        """Zoom to `factor` while keeping (canvas, cx, cy) under the cursor fixed.

        If canvas_center is None, zoom around the center of the result canvas.
        """
        target_img = self.original_image or self.current_image
        if target_img is None:
            # No image loaded yet: just change factor (no anchor logic needed)
            self._zoom_factor = self._clamp_zoom(factor)
            self._update_zoom_label()
            self._redraw_all()
            self._redraw_cursor_overlay()
            return

        if canvas_center is None:
            c = self.canvas_result
            cx = c.winfo_width() // 2
            cy = c.winfo_height() // 2
        else:
            c, cx, cy = canvas_center

        old_z = self._zoom_factor
        new_z = self._clamp_zoom(factor)
        if new_z == old_z:
            return

        # Compute where the cursor was in image pixel space BEFORE zoom,
        # so we can adjust pan to keep the same image pixel under cursor AFTER.
        bbox_before = self._image_to_canvas_bbox_raw(c, target_img.size, old_z)
        if bbox_before is None:
            self._zoom_factor = new_z
            self._update_zoom_label()
            self._redraw_all()
            return
        x0, y0, disp_w, disp_h = bbox_before
        if disp_w > 0 and disp_h > 0 and x0 <= cx < x0 + disp_w and \
                y0 <= cy < y0 + disp_h:
            # Point on source image in pixels (0..w, 0..h)
            iw, ih = target_img.size
            fx = (cx - x0) / disp_w
            fy = (cy - y0) / disp_h
        else:
            fx = 0.5
            fy = 0.5

        self._zoom_factor = new_z
        # Recompute bbox at new zoom (WITHOUT pan applied yet to get centered base)
        bbox_after = self._image_to_canvas_bbox_raw(c, target_img.size, new_z,
                                                     ignore_pan=True)
        if bbox_after is None:
            self._update_zoom_label()
            self._redraw_all()
            return
        nx0, ny0, ndisp_w, ndisp_h = bbox_after
        # We want cursor cx,cy to land at image-relative fx/fy of new bbox → solve
        # cx == nx0 + pan_x + fx * ndisp_w  →  pan_x = cx - nx0 - fx*ndisp_w
        new_pan_x = cx - (nx0 + fx * ndisp_w)
        new_pan_y = cy - (ny0 + fy * ndisp_h)
        self._pan_x = int(round(new_pan_x))
        self._pan_y = int(round(new_pan_y))

        self._update_zoom_label()
        self._redraw_all()
        self._redraw_cursor_overlay()

    def _zoom_fit(self):
        """Fit the image to the Result canvas, reset pan to 0."""
        target_img = self.original_image or self.current_image
        if target_img is None:
            return
        c = self.canvas_result
        canvas_w = max(2, c.winfo_width() - 8)
        canvas_h = max(2, c.winfo_height() - 8)
        iw, ih = target_img.size
        self._zoom_factor = self._clamp_zoom(min(canvas_w / iw, canvas_h / ih))
        self._pan_x = 0
        self._pan_y = 0
        self._update_zoom_label()
        self._redraw_all()
        self._redraw_cursor_overlay()

    def _zoom_reset(self):
        """Clicking the zoom-% button toggles between Fit and 100%."""
        if abs(self._zoom_factor - 1.0) < 0.01:
            self._zoom_fit()
        else:
            self._zoom_set(1.0)

    def _pan_by(self, dx: int, dy: int):
        if dx == 0 and dy == 0:
            return
        self._pan_x += int(dx)
        self._pan_y += int(dy)
        self._redraw_all()
        self._redraw_cursor_overlay()

    # ---- Core: image -> canvas bounding box (with zoom + pan) ----
    def _image_to_canvas_bbox_raw(self, canvas: tk.Canvas,
                                   img_size: Tuple[int, int],
                                   zoom: float,
                                   ignore_pan: bool = False) -> Optional[Tuple]:
        """Return (x0, y0, disp_w, disp_h) for where img_size would render on canvas
        using the current zoom and pan (or ignore_pan=True to get the centered,
        un-panned baseline)."""
        canvas_w = max(2, canvas.winfo_width() - 8)
        canvas_h = max(2, canvas.winfo_height() - 8)
        iw, ih = img_size
        # Fit-based default scale at zoom=1.0 is the ratio that fits the image.
        # We treat `zoom_factor = 1.0` as meaning "fit the canvas".
        # Then zoom=2.0 means "twice as many pixels as fit".
        fit_scale = min(canvas_w / iw, canvas_h / ih)
        final_scale = fit_scale * zoom
        disp_w = int(round(iw * final_scale))
        disp_h = int(round(ih * final_scale))
        # Centered position without pan
        base_x = (canvas.winfo_width() - disp_w) // 2
        base_y = (canvas.winfo_height() - disp_h) // 2
        if ignore_pan:
            return (base_x, base_y, disp_w, disp_h)
        return (base_x + self._pan_x, base_y + self._pan_y, disp_w, disp_h)

    def _image_to_canvas_bbox(self, canvas: tk.Canvas,
                               img: Optional[Image.Image]) -> Optional[Tuple[int,
                                                                             int,
                                                                             int,
                                                                             int]]:
        """Public helper: return (x0, y0, x1, y1) screen bbox of image on canvas."""
        if img is None:
            return None
        raw = self._image_to_canvas_bbox_raw(canvas, img.size, self._zoom_factor)
        if raw is None:
            return None
        x0, y0, dw, dh = raw
        return (x0, y0, x0 + dw, y0 + dh)

    def _canvas_to_image_coords(self, canvas: tk.Canvas, cx: int, cy: int,
                                 img: Optional[Image.Image]) -> Optional[Tuple[int,
                                                                               int]]:
        """Convert canvas (cx,cy) to integer pixel (px,py) on img. None if outside."""
        if img is None:
            return None
        raw = self._image_to_canvas_bbox_raw(canvas, img.size, self._zoom_factor)
        if raw is None:
            return None
        x0, y0, dw, dh = raw
        if cx < x0 or cx >= x0 + dw or cy < y0 or cy >= y0 + dh:
            return None
        iw, ih = img.size
        px = int((cx - x0) / dw * iw)
        py = int((cy - y0) / dh * ih)
        return max(0, min(px, iw - 1)), max(0, min(py, ih - 1))

    def _canvas_brush_radius(self, canvas: tk.Canvas,
                              img: Optional[Image.Image]) -> int:
        """Brush on-screen radius in canvas pixels, honoring zoom.

        brush_size is in IMAGE pixels. To draw the cursor circle at the correct
        on-screen size we multiply by the display scale: disp_w / img_w.
        """
        if img is None:
            return max(2, int(self.brush_size.get() / 2))
        raw = self._image_to_canvas_bbox_raw(canvas, img.size, self._zoom_factor)
        if raw is None:
            return max(2, int(self.brush_size.get() / 2))
        x0, y0, dw, dh = raw
        iw, _ = img.size
        if iw <= 0:
            return max(2, int(self.brush_size.get() / 2))
        scale = dw / iw  # pixels-per-image-pixel on screen
        return max(1, int((self.brush_size.get() / 2) * scale))

    # ------------------------------------------------------------------
    # Canvas configure (auto-fit on first load or resize)
    # ------------------------------------------------------------------
    def _on_canvas_configure(self, event):
        # First load, or explicit auto-fit request → fit image to canvas.
        any_image = self.original_image or self.current_image
        if any_image is not None and self._zoom_pending_fit:
            self._zoom_pending_fit = False
            self._zoom_fit()
            return
        self._redraw_all()
        self._redraw_cursor_overlay()

    # ------------------------------------------------------------------
    # Mouse wheel: Wheel = zoom (touchpad friendly), Ctrl+Wheel = brush size
    # ------------------------------------------------------------------
    def _on_mousewheel(self, event, _override=None):
        """Handle scroll wheel + touchpad gestures.

        - Plain scroll / touchpad two-finger scroll → ZOOM in/out around cursor
        - Ctrl held + scroll / touchpad pinch (Windows sends Ctrl+wheel) →
          change brush size when over the Result canvas (otherwise zoom)
        """
        # Windows: event.delta is +120 per notch up, -120 per notch down.
        # Linux Button-4/5: we pass _override=1 for up, -1 for down.
        if _override is not None:
            steps = float(_override)
        else:
            steps = 1.0 if event.delta > 0 else -1.0

        ctrl = bool(event.state & 0x0004)

        target_canvas = self._canvas_for_event(event.widget)

        if ctrl and target_canvas is self.canvas_result:
            # Ctrl + wheel → brush size (Photoshop standard)
            delta = int(round(steps * 6))
            new_s = max(1, min(300, int(self.brush_size.get()) + delta))
            if new_s != int(self.brush_size.get()):
                self.brush_size.set(new_s)
                self._on_brush_change()
                self._redraw_cursor_overlay()
            return "break"

        # Default: wheel zooms (touchpad friendly)
        factor = 1.12 if steps > 0 else 1.0 / 1.12
        self._zoom_at(self._zoom_factor * factor,
                       (target_canvas, event.x, event.y))
        return "break"

    # ------------------------------------------------------------------
    # Pan helpers (Space + drag, Middle-mouse drag, or Pan tool)
    # ------------------------------------------------------------------
    def _is_pan_active(self, event) -> bool:
        """Check if panning is requested via Spacebar, Pan tool, or Middle mouse."""
        if getattr(event, "num", None) == 2:
            return True
        if self._space_held or self._alt_held:
            return True
        if self.selected_tool.get() == TOOL_PAN:
            return True
        return False

    def _on_focus_out(self, _event=None):
        self._space_held = False
        self._alt_held = False
        self._is_panning = False
        self._is_drawing = False
        self._compare_active_drag = False
        self._update_cursor_and_tool_highlight()

    def _on_space_down(self, _event=None):
        if not self._space_held:
            self._space_held = True
            self._update_cursor_and_tool_highlight()

    def _on_space_up(self, _event=None):
        self._space_held = False
        self._update_cursor_and_tool_highlight()

    def _on_middle_press(self, event):
        self._begin_pan(event)

    def _on_middle_drag(self, event):
        self._continue_pan(event)

    def _on_mouse_release_any(self, _event):
        self._is_panning = False
        self._pan_start_xy = None

    def _begin_pan(self, event):
        if (self.original_image or self.current_image) is None:
            return
        self._is_panning = True
        self._pan_start_xy = (event.x, event.y, self._pan_x, self._pan_y)

    def _continue_pan(self, event):
        if not self._is_panning or self._pan_start_xy is None:
            return
        sx, sy, spx, spy = self._pan_start_xy
        self._pan_x = spx + (event.x - sx)
        self._pan_y = spy + (event.y - sy)
        self._redraw_all()
        self._redraw_cursor_overlay()

    # ------------------------------------------------------------------
    # Canvas drawing helpers (with zoom & instant pan repositioning)
    # ------------------------------------------------------------------
    _tile_cache = {}

    @classmethod
    def _get_checkerboard_tile(cls, box_size: int = 6, theme: str = "Dark Check (Small)") -> Image.Image:
        key = (box_size, theme)
        if key in cls._tile_cache:
            return cls._tile_cache[key]

        if theme in ("Dark Check (Small)", "dark_small", "dark"):
            box_size = 6
            c1 = (20, 20, 20, 255)
            c2 = (42, 42, 42, 255)
        elif theme in ("Dark Check (Contrast)", "dark_contrast"):
            box_size = 6
            c1 = (12, 12, 12, 255)
            c2 = (62, 62, 62, 255)
        elif theme in ("Solid Dark", "solid_dark"):
            box_size = 6
            c1 = (22, 22, 22, 255)
            c2 = (22, 22, 22, 255)
        elif theme in ("Solid White", "solid_white"):
            box_size = 6
            c1 = (255, 255, 255, 255)
            c2 = (255, 255, 255, 255)
        else:  # Light Check
            box_size = 8
            c1 = (255, 255, 255, 255)
            c2 = (204, 204, 204, 255)

        tile_size = box_size * 2
        img = Image.new("RGBA", (tile_size, tile_size), c1)
        sub = Image.new("RGBA", (box_size, box_size), c2)
        img.paste(sub, (0, 0))
        img.paste(sub, (box_size, box_size))
        cls._tile_cache[key] = img
        return img

    def _create_checkerboard(self, width: int, height: int, box_size: Optional[int] = None, theme: Optional[str] = None) -> Image.Image:
        """Fast exponential doubling checkerboard generator (runs in < 0.1ms)."""
        if theme is None:
            theme_var = getattr(self, "bg_pattern_var", None)
            theme = theme_var.get() if theme_var else "Dark Check (Small)"
        if box_size is None:
            box_size = 6 if ("Small" in theme or "Contrast" in theme) else 8

        tile = self._get_checkerboard_tile(box_size, theme)
        cur = tile
        while cur.width < width or cur.height < height:
            nw = min(width, cur.width * 2) if cur.width < width else cur.width
            nh = min(height, cur.height * 2) if cur.height < height else cur.height
            doubled = Image.new("RGBA", (nw, nh))
            doubled.paste(cur, (0, 0))
            if cur.width < width:
                doubled.paste(cur, (cur.width, 0))
            if cur.height < height:
                doubled.paste(cur, (0, cur.height))
            if cur.width < width and cur.height < height:
                doubled.paste(cur, (cur.width, cur.height))
            cur = doubled
        if cur.width != width or cur.height != height:
            return cur.crop((0, 0, width, height))
        return cur

    def _invalidate_render_cache(self, which: Optional[str] = None):
        """Invalidate rendered image caches."""
        if which is None:
            self._render_cache["orig"] = {"img_ref": None, "size": None, "photo": None, "item_id": None}
            self._render_cache["result"] = {"img_ref": None, "size": None, "photo": None, "item_id": None, "version": 0}
            self._compare_cache = {"orig_rgb": None, "res_rgb": None, "size": None, "version": None}
        elif which in self._render_cache:
            self._render_cache[which]["img_ref"] = None

    def _schedule_hq_render(self):
        """Debounce a high-quality LANCZOS render after fast user interactions stop."""
        if self._hq_debounce_id is not None:
            try:
                self.root.after_cancel(self._hq_debounce_id)
            except Exception:
                pass
        self._hq_debounce_id = self.root.after(140, self._render_hq)

    def _render_hq(self):
        self._hq_debounce_id = None
        self._invalidate_render_cache()
        self._redraw_all(force_hq=True)
        self._redraw_cursor_overlay()

    def _display_image_on_canvas(self, canvas: tk.Canvas,
                                  img: Optional[Image.Image],
                                  cache_key: str,
                                  force_hq: bool = False):
        """Draw img on canvas with checkerboard background if RGBA, honoring
        zoom + pan with instant coordinate repositioning when panning."""
        cache = self._render_cache[cache_key]

        if img is None:
            canvas.delete("image_item")
            cache["img_ref"] = None
            cache["photo"] = None
            cache["item_id"] = None
            return

        raw = self._image_to_canvas_bbox_raw(canvas, img.size, self._zoom_factor)
        if raw is None:
            return
        x0, y0, disp_w, disp_h = raw
        if disp_w <= 0 or disp_h <= 0:
            disp_w = max(2, disp_w)
            disp_h = max(2, disp_h)

        current_version = self._result_version if cache_key == "result" else 0

        # Fast-path: Check if we can reuse the already rendered PhotoImage and just move its coordinates
        if (cache["img_ref"] is img and
                cache["size"] == (disp_w, disp_h) and
                cache.get("version", 0) == current_version and
                cache["photo"] is not None and
                cache["item_id"] is not None):
            try:
                canvas.coords(cache["item_id"], x0, y0)
                canvas.tag_lower("image_item")
                return
            except Exception:
                pass

        # Choose resampling filter: BILINEAR during fast interaction, LANCZOS when idle or forced HQ
        resample = Image.Resampling.LANCZOS if force_hq else Image.Resampling.BILINEAR

        img_copy = img.resize((disp_w, disp_h), resample)

        if img_copy.mode == "RGBA":
            checker = self._create_checkerboard(disp_w, disp_h)
            checker.paste(img_copy, (0, 0), img_copy)
            display_img = checker.convert("RGB")
        else:
            display_img = img_copy.convert("RGB")

        photo = ImageTk.PhotoImage(display_img)
        cache["photo"] = photo
        cache["img_ref"] = img
        cache["size"] = (disp_w, disp_h)
        cache["version"] = current_version

        if cache_key == "orig":
            self._photo_orig = photo
        else:
            self._photo_result = photo

        if cache["item_id"] is not None:
            try:
                canvas.itemconfig(cache["item_id"], image=photo)
                canvas.coords(cache["item_id"], x0, y0)
            except Exception:
                canvas.delete("image_item")
                item_id = canvas.create_image(x0, y0, image=photo, anchor=tk.NW, tags="image_item")
                cache["item_id"] = item_id
        else:
            canvas.delete("image_item")
            item_id = canvas.create_image(x0, y0, image=photo, anchor=tk.NW, tags="image_item")
            cache["item_id"] = item_id

        canvas.tag_lower("image_item")
        if not force_hq:
            self._schedule_hq_render()

    def _redraw_all(self, force_hq: bool = False):
        self._display_image_on_canvas(self.canvas_orig,
                                       self.original_image, "orig", force_hq=force_hq)
        if self._compare_mode and self.original_image and self.current_image:
            self._draw_compare()
        else:
            self._display_image_on_canvas(self.canvas_result,
                                           self.current_image, "result", force_hq=force_hq)

    # ------------------------------------------------------------------
    # Cursor overlay (circle brush preview) + crosshair (for wand)
    # ------------------------------------------------------------------
    def _clear_cursor_overlay(self):
        c = self.canvas_result
        for i in self._cursor_crosshair_ids:
            try:
                c.delete(i)
            except Exception:
                pass
        self._cursor_crosshair_ids = []

    def _redraw_cursor_overlay(self):
        """Draw the brush preview circle / wand crosshair on the result canvas."""
        self._clear_cursor_overlay()
        c = self.canvas_result
        if self._cursor_canvas_pos is None:
            return
        cx, cy = self._cursor_canvas_pos

        img = self.current_image if self.current_image is not None else \
            self.original_image

        bbox = self._image_to_canvas_bbox(c, img) if img is not None else None
        if bbox is not None:
            x0, y0, x1, y1 = bbox
            if not (x0 <= cx < x1 and y0 <= cy < y1):
                return
        else:
            x0, y0 = 0, 0
            x1, y1 = c.winfo_width(), c.winfo_height()
        tool = self.selected_tool.get()

        if tool == TOOL_WAND:
            color = "#ff6a3d"
            r = 10
            ids = [
                c.create_line(cx - 16, cy, cx - r, cy, fill=color, width=2),
                c.create_line(cx + r, cy, cx + 16, cy, fill=color, width=2),
                c.create_line(cx, cy - 16, cx, cy - r, fill=color, width=2),
                c.create_line(cx, cy + r, cx, cy + 16, fill=color, width=2),
                c.create_oval(cx - r, cy - r, cx + r, cy + r, outline=color,
                               width=2),
            ]
            self._cursor_crosshair_ids = ids
            return

        if tool == TOOL_COMPARE:
            color = "#4a9eff"
            line_id = c.create_line(cx, y0, cx, y1, fill=color, width=1,
                                     dash=(4, 4))
            self._cursor_crosshair_ids = [line_id]
            return

        if tool in (TOOL_ERASER, TOOL_RESTORER):
            r = self._canvas_brush_radius(c, img)
            color = "#4aff8a" if tool == TOOL_RESTORER else "#ff4a4a"
            circ_id = c.create_oval(
                cx - r, cy - r, cx + r, cy + r, outline=color, width=2,
            )
            ch = 8
            ids = [
                circ_id,
                c.create_line(cx - ch, cy, cx - 2, cy, fill=color, width=2),
                c.create_line(cx + 2, cy, cx + ch, cy, fill=color, width=2),
                c.create_line(cx, cy - ch, cx, cy - 2, fill=color, width=2),
                c.create_line(cx, cy + 2, cx, cy + ch, fill=color, width=2),
            ]
            self._cursor_crosshair_ids = ids
            return

        if tool == TOOL_MAGIC_ERASER:
            # Orange circle (brush size) + wand crosshair = combined cursor
            r = self._canvas_brush_radius(c, img)
            color = "#ffaa00"
            circ_id = c.create_oval(
                cx - r, cy - r, cx + r, cy + r, outline=color, width=2,
            )
            wr = max(6, r // 3)
            ids = [
                circ_id,
                c.create_line(cx - r - 6, cy, cx - r, cy, fill=color, width=2),
                c.create_line(cx + r, cy, cx + r + 6, cy, fill=color, width=2),
                c.create_line(cx, cy - r - 6, cx, cy - r, fill=color, width=2),
                c.create_line(cx, cy + r, cx, cy + r + 6, fill=color, width=2),
            ]
            self._cursor_crosshair_ids = ids
            return

        if tool == TOOL_PAN or self._space_held or self._alt_held:
            # Show a small hand icon via canvas text (unicode hand)
            hand_id = c.create_text(
                cx, cy, text="✋", font=("Segoe UI", 14),
                fill="#ffffff", tags="cursor_overlay",
            )
            self._cursor_crosshair_ids = [hand_id]
            return

    # ------------------------------------------------------------------
    # Tool change + stability
    # ------------------------------------------------------------------
    def _on_tool_change(self):
        tool = self.selected_tool.get()
        if tool != TOOL_COMPARE:
            self._exit_compare_mode()
        self._update_cursor_and_tool_highlight()
        self._redraw_cursor_overlay()
        if tool == TOOL_WAND:
            self.status_var.set(
                f"Magic Wand: click a color to flood-fill erase it (Tol={int(self.wand_tolerance.get())}). Click again to remove another."
            )
        elif tool == TOOL_ERASER:
            self.status_var.set("Eraser: paint to erase pixels. Ctrl+Wheel=adjust brush size.")
        elif tool == TOOL_RESTORER:
            self.status_var.set("Restorer: paint to restore pixels from the original image.")
        elif tool == TOOL_MAGIC_ERASER:
            self.status_var.set(
                f"Magic Eraser: drag over an unwanted color to erase its connected pixels "
                f"inside the brush (Tol={int(self.wand_tolerance.get())})."
            )
        elif tool == TOOL_COMPARE:
            self.status_var.set("Compare: drag on Result to slide before/after comparison.")
        elif tool == TOOL_PAN:
            self.status_var.set("Pan: drag to move the view. Also: Space+drag or middle-mouse drag.")

    def _update_cursor_and_tool_highlight(self):
        c = self.canvas_result
        if self._space_held:
            c.config(cursor="fleur")
            self.canvas_orig.config(cursor="fleur")
            return

        tool = self.selected_tool.get()
        if tool == TOOL_PAN:
            c.config(cursor="fleur")
            self.canvas_orig.config(cursor="fleur")
        elif tool == TOOL_COMPARE:
            c.config(cursor="sb_h_double_arrow")
            self.canvas_orig.config(cursor="fleur")
        elif tool == TOOL_WAND:
            c.config(cursor="crosshair")
            self.canvas_orig.config(cursor="fleur")
        elif tool == TOOL_MAGIC_ERASER:
            c.config(cursor="crosshair")
            self.canvas_orig.config(cursor="fleur")
        elif tool == TOOL_ERASER:
            c.config(cursor="crosshair")
            self.canvas_orig.config(cursor="fleur")
        elif tool == TOOL_RESTORER:
            c.config(cursor="target")
            self.canvas_orig.config(cursor="fleur")
        else:
            c.config(cursor="arrow")
            self.canvas_orig.config(cursor="fleur")

    def _on_brush_change(self):
        self.brush_label.config(text=f"{int(self.brush_size.get())}px")

    # ------------------------------------------------------------------
    # ESC / right-click = cancel current tool
    # ------------------------------------------------------------------
    def _on_escape(self, _event=None):
        if self._is_panning:
            self._is_panning = False
            self._pan_start_xy = None
        if self._compare_mode:
            self._exit_compare_mode()
            return "break"
        if self._is_drawing:
            self._is_drawing = False
            self._last_xy = None
            self._draw_redraw_pending = False
            if self._active_alpha is not None:
                if self._active_rgb is not None:
                    rgb_img = Image.fromarray(self._active_rgb).convert("RGBA")
                    rgb_img.putalpha(Image.fromarray(self._active_alpha))
                    self.current_image = rgb_img
                else:
                    self.current_image.putalpha(Image.fromarray(self._active_alpha))
                self._active_alpha = None
                self._active_rgb = None
                self._active_orig_rgb = None
        return "break"

    def _on_right_click(self, _event=None):
        if self._compare_mode:
            self._exit_compare_mode()
            return "break"
        return None

    # ------------------------------------------------------------------
    # Undo / reset
    # ------------------------------------------------------------------
    def _push_undo(self):
        if self.current_image is None:
            return
        self._undo_stack.append(self.current_image.copy())
        if len(self._undo_stack) > self._max_undo:
            self._undo_stack.pop(0)
        self._update_undo_button()

    def _update_undo_button(self):
        self.btn_undo.config(state=tk.NORMAL if self._undo_stack else tk.DISABLED)

    def _undo(self):
        if not self._undo_stack:
            return
        self.current_image = self._undo_stack.pop()
        self._result_version += 1
        self._invalidate_render_cache("result")
        self._update_undo_button()
        self._redraw_all()
        self.btn_save.config(state=tk.NORMAL if self.current_image else tk.DISABLED)
        self.btn_quality_edit.config(state=tk.NORMAL if self.current_image else tk.DISABLED)
        self.status_var.set("Undo complete.")

    def _reset_mask(self):
        if self.ai_result_image is None:
            return
        if not messagebox.askyesno(
                "Reset Mask",
                "Discard all manual edits and restore the AI result?"):
            return
        self._push_undo()
        self.current_image = self.ai_result_image.copy()
        self._result_version += 1
        self._invalidate_render_cache("result")
        self._redraw_all()
        self.btn_quality_edit.config(state=tk.NORMAL if self.current_image else tk.DISABLED)
        self.status_var.set("Mask reset to AI result.")

    def _open_quality_edit_dialog(self):
        """Open a sleek Quality Edit dialog to fine-tune edge antialiasing, defringing, and mask quality."""
        if self.current_image is None:
            messagebox.showinfo("Quality Edit", "Please open and process an image first.")
            return

        dialog = tk.Toplevel(self.root)
        dialog.title("✨ Quality Edit & Edge Refinement")
        dialog.geometry("460x390")
        dialog.resizable(False, False)
        dialog.configure(bg="#242424")
        dialog.transient(self.root)
        dialog.grab_set()

        pad = 14
        frame = ttk.Frame(dialog, padding=pad)
        frame.pack(fill=tk.BOTH, expand=True)

        # Header
        ttk.Label(
            frame,
            text="✨ Mask Quality & Edge Refinement",
            font=("Segoe UI", 12, "bold"),
            foreground="#ffffff",
        ).pack(anchor=tk.W, pady=(0, 2))

        ttk.Label(
            frame,
            text="Antialias jagged edges, remove background halos, and clean stray pixels.",
            font=("Segoe UI", 9),
            foreground="#aaaaaa",
        ).pack(anchor=tk.W, pady=(0, 12))

        # Smoothness slider
        smooth_frame = ttk.Frame(frame)
        smooth_frame.pack(fill=tk.X, pady=(4, 0))
        ttk.Label(smooth_frame, text="Edge Antialiasing / Smoothness:").pack(side=tk.LEFT)
        smooth_val_lbl = ttk.Label(smooth_frame, text="1.2", width=4)
        smooth_val_lbl.pack(side=tk.RIGHT)
        smooth_slider = ttk.Scale(
            frame, from_=0.0, to=3.0, orient=tk.HORIZONTAL, value=1.2,
            command=lambda v: smooth_val_lbl.config(text=f"{float(v):.1f}")
        )
        smooth_slider.pack(fill=tk.X, pady=(2, 8))

        # Choke / Feather slider
        choke_frame = ttk.Frame(frame)
        choke_frame.pack(fill=tk.X, pady=(4, 0))
        ttk.Label(choke_frame, text="Edge Shift (Choke - / Expand +):").pack(side=tk.LEFT)
        choke_val_lbl = ttk.Label(choke_frame, text="0 px", width=6)
        choke_val_lbl.pack(side=tk.RIGHT)
        choke_slider = ttk.Scale(
            frame, from_=-4, to=4, orient=tk.HORIZONTAL, value=0,
            command=lambda v: choke_val_lbl.config(text=f"{int(round(float(v)))} px")
        )
        choke_slider.pack(fill=tk.X, pady=(2, 8))

        # Checkboxes
        defringe_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            frame,
            text="Defringe / Color Decontaminate (eliminates background color halos)",
            variable=defringe_var,
        ).pack(anchor=tk.W, pady=3)

        specks_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            frame,
            text="Remove Stray Pixel Noise & Artifacts",
            variable=specks_var,
        ).pack(anchor=tk.W, pady=3)

        def apply_refinement(smooth_override=None, defringe_override=None, specks_override=None, choke_override=None):
            s = smooth_slider.get() if smooth_override is None else smooth_override
            df = defringe_var.get() if defringe_override is None else defringe_override
            sp = specks_var.get() if specks_override is None else specks_override
            ch = int(round(choke_slider.get())) if choke_override is None else choke_override

            self._push_undo()
            refined = refine_mask_quality(
                self.current_image,
                smooth=float(s),
                defringe=bool(df),
                clean_specks=bool(sp),
                choke=int(ch),
            )
            self.current_image = refined
            self._result_version += 1
            self._invalidate_render_cache("result")
            self._redraw_all()
            self.btn_save.config(state=tk.NORMAL)
            self.status_var.set(f"Quality Edit applied: Smooth={float(s):.1f}, Defringe={df}, Specks={sp}, Choke={ch}px.")

        btn_box = ttk.Frame(frame)
        btn_box.pack(fill=tk.X, pady=(16, 0))

        ttk.Button(
            btn_box,
            text="⚡ Maximum Quality Preset",
            style="Accent.TButton",
            command=lambda: (apply_refinement(smooth_override=1.4, defringe_override=True, specks_override=True, choke_override=0), dialog.destroy()),
        ).pack(side=tk.LEFT, padx=(0, 6))

        ttk.Button(
            btn_box,
            text="Apply Custom",
            command=lambda: apply_refinement(),
        ).pack(side=tk.LEFT, padx=4)

        ttk.Button(
            btn_box,
            text="Undo",
            command=lambda: self._undo(),
        ).pack(side=tk.LEFT, padx=4)

        ttk.Button(
            btn_box,
            text="Close",
            command=dialog.destroy,
        ).pack(side=tk.RIGHT)

    # ------------------------------------------------------------------
    # Mouse handlers (with pan / draw / wand / compare)
    # ------------------------------------------------------------------
    def _on_mouse_move(self, event):
        self._cursor_canvas_pos = (event.x, event.y)
        self._redraw_cursor_overlay()

    def _on_mouse_leave(self, _event):
        self._clear_cursor_overlay()

    def _on_mouse_down(self, event):
        # 1. Middle mouse button always pans
        if getattr(event, "num", None) == 2:
            self._begin_pan(event)
            return

        # 2. Original canvas left-click pans (also sync pan state reset)
        if event.widget is self.canvas_orig:
            self._is_drawing = False      # make sure no draw bleeds from result
            self._begin_pan(event)
            return

        # 3. Only process result canvas beyond this point
        if event.widget is not self.canvas_result:
            return

        # 4. If Space is held, Alt is held, or Pan tool is selected: pan
        if self._is_pan_active(event):
            self._is_drawing = False
            self._begin_pan(event)
            return

        # Not panning → make sure panning state is cleared
        self._is_panning = False
        self._pan_start_xy = None

        # 5. Result canvas edit tools:
        if self.current_image is None and self.original_image is None:
            return

        tool = self.selected_tool.get()

        if tool == TOOL_WAND:
            self._is_drawing = False
            self._wand_click(event)
            return

        if tool == TOOL_COMPARE:
            self._is_drawing = False
            self._compare_start(event)
            return

        if tool in (TOOL_ERASER, TOOL_RESTORER, TOOL_MAGIC_ERASER):
            if self.current_image is None:
                return
            self._is_panning = False
            self._is_drawing = True
            self._push_undo()
            self._last_xy = None

            # Fast in-memory 2D contiguous arrays for zero-lag drawing
            self._active_alpha = np.array(self.current_image.getchannel("A"), copy=True)
            self._active_rgb = np.array(self.current_image.convert("RGB"), copy=True)
            if self.original_image is not None and self.original_image.size == self.current_image.size:
                self._active_orig_rgb = np.array(self.original_image.convert("RGB"), copy=True)
            else:
                self._active_orig_rgb = None

            self._draw_stroke(event, single=True)
            self._cursor_canvas_pos = (event.x, event.y)
            self._redraw_cursor_overlay()
            return

    def _on_mouse_drag(self, event):
        # Pan drag: only valid when _is_panning AND the initiating widget
        # (canvas_orig or result with pan-active) is the current event source.
        # We never pan if the user has a drawing tool active (_is_drawing).
        if self._is_panning and not self._is_drawing:
            self._continue_pan(event)
            return

        # Drawing / compare drag: only on result canvas
        if event.widget is self.canvas_result:
            self._cursor_canvas_pos = (event.x, event.y)
            self._redraw_cursor_overlay()

            tool = self.selected_tool.get()
            if tool == TOOL_COMPARE and self._compare_active_drag:
                self._compare_move(event)
                return

            if self._is_drawing and self.current_image is not None:
                self._draw_stroke(event, single=False)
                return

    def _on_mouse_up(self, event):
        # Always stop panning on any button-up
        was_panning = self._is_panning
        self._is_panning = False
        self._pan_start_xy = None
        if was_panning and not self._is_drawing:
            return

        if event.widget is not self.canvas_result:
            # Even if this isn't the result canvas, clear drawing state
            self._is_drawing = False
            return

        if self._compare_active_drag:
            self._compare_active_drag = False
            self._exit_compare_mode()
            self._update_cursor_and_tool_highlight()
            return

        if self._is_drawing:
            self._is_drawing = False
            self._last_xy = None
            if self._draw_redraw_pending:
                self._draw_redraw_pending = False
            if self._active_alpha is not None:
                if self._active_rgb is not None:
                    rgb_img = Image.fromarray(self._active_rgb).convert("RGBA")
                    rgb_img.putalpha(Image.fromarray(self._active_alpha))
                    self.current_image = rgb_img
                else:
                    self.current_image.putalpha(Image.fromarray(self._active_alpha))
                self._active_alpha = None
                self._active_rgb = None
                self._active_orig_rgb = None
            self._result_version += 1
            self._invalidate_render_cache("result")
            self._update_undo_button()
            self._redraw_all(force_hq=True)
            self._redraw_cursor_overlay()
            if self.current_image is not None:
                self.btn_save.config(state=tk.NORMAL)
                tool = self.selected_tool.get()
                if tool == TOOL_ERASER:
                    action = "Erased"
                elif tool == TOOL_MAGIC_ERASER:
                    action = "Magic Erased (Color-Aware)"
                else:
                    action = "Restored"
                self.status_var.set(f"{action} — save when done. (Undo to revert)")

    # ------------------------------------------------------------------
    # Zero-lag brush stroke engine & Color-Aware Magic Eraser
    # ------------------------------------------------------------------
    def _request_draw_redraw(self):
        """Throttle screen updates during active drawing to ~60 FPS."""
        if not self._draw_redraw_pending:
            self._draw_redraw_pending = True
            self.root.after(16, self._perform_draw_redraw)

    def _perform_draw_redraw(self):
        self._draw_redraw_pending = False
        if self._active_alpha is None:
            return
        if self._active_rgb is not None:
            rgb_img = Image.fromarray(self._active_rgb).convert("RGBA")
            rgb_img.putalpha(Image.fromarray(self._active_alpha))
            self.current_image = rgb_img
        else:
            self.current_image.putalpha(Image.fromarray(self._active_alpha))
        self._result_version += 1
        # Update result canvas only — do not waste CPU redrawing canvas_orig during strokes
        self._display_image_on_canvas(self.canvas_result, self.current_image, "result", force_hq=False)
        self._redraw_cursor_overlay()

    def _draw_stroke(self, event, single: bool):
        tool = self.selected_tool.get()
        if tool not in (TOOL_ERASER, TOOL_RESTORER, TOOL_MAGIC_ERASER):
            return

        size = int(self.brush_size.get())
        p0 = self._canvas_to_image_coords(self.canvas_result, event.x, event.y,
                                           self.current_image)
        if p0 is None:
            self._last_xy = None
            return

        if self._active_alpha is None:
            self._active_alpha = np.array(self.current_image.getchannel("A"), copy=True)
            self._active_rgb = np.array(self.current_image.convert("RGB"), copy=True)

        if not single and self._last_xy is not None:
            self._apply_brush_line(self._last_xy, p0, size, tool)
        else:
            self._apply_brush_dot(p0, size, tool)

        self._last_xy = p0
        if single:
            self._perform_draw_redraw()
        else:
            self._request_draw_redraw()

    def _apply_brush_line(self, p0, p1, size, tool):
        if self._active_alpha is None:
            return
        if tool == TOOL_ERASER:
            cv2.line(self._active_alpha, p0, p1, 0, thickness=size)
        elif tool == TOOL_RESTORER:
            cv2.line(self._active_alpha, p0, p1, 255, thickness=size)
            if self._active_orig_rgb is not None and self._active_rgb is not None:
                h, w = self._active_alpha.shape
                line_mask = np.zeros((h, w), dtype=np.uint8)
                cv2.line(line_mask, p0, p1, 255, thickness=size)
                where_m = (line_mask > 0)
                self._active_rgb[where_m] = self._active_orig_rgb[where_m]
        elif tool == TOOL_MAGIC_ERASER:
            color_src = (
                self._active_orig_rgb
                if self._active_orig_rgb is not None
                else self._active_rgb
            )
            if color_src is None:
                color_src = np.array(
                    self.current_image.convert("RGB"),
                    dtype=np.uint8,
                )
                self._active_rgb = color_src

            points = get_drag_sample_points(
                color_src,
                p0,
                p1,
                int(self.wand_tolerance.get()),
                max(2, size // 4),
            )
            for point in points:
                self._apply_magic_eraser_dot(point, size)

    def _apply_brush_dot(self, pt, size, tool):
        if self._active_alpha is None:
            return
        px, py = pt
        r = max(1, size // 2)
        if tool == TOOL_ERASER:
            cv2.circle(self._active_alpha, (px, py), r, 0, -1)
        elif tool == TOOL_RESTORER:
            cv2.circle(self._active_alpha, (px, py), r, 255, -1)
            if self._active_orig_rgb is not None and self._active_rgb is not None:
                h, w = self._active_alpha.shape
                x0, y0 = max(0, px - r), max(0, py - r)
                x1, y1 = min(w, px + r + 1), min(h, py + r + 1)
                yy, xx = np.ogrid[:y1 - y0, :x1 - x0]
                c_mask = ((xx - (px - x0)) ** 2 + (yy - (py - y0)) ** 2 <= r ** 2)
                self._active_rgb[y0:y1, x0:x1][c_mask] = self._active_orig_rgb[y0:y1, x0:x1][c_mask]
        elif tool == TOOL_MAGIC_ERASER:
            self._apply_magic_eraser_dot(pt, size)

    def _apply_magic_eraser_dot(self, pt, size):
        """Erase the connected color sampled beneath the brush."""
        if self._active_alpha is None:
            return

        color_src = (
            self._active_orig_rgb
            if self._active_orig_rgb is not None
            else self._active_rgb
        )
        if color_src is None:
            color_src = np.array(self.current_image.convert("RGB"), dtype=np.uint8)
            self._active_rgb = color_src

        erase_connected_color(
            self._active_alpha,
            color_src,
            pt,
            max(2, size // 2),
            int(self.wand_tolerance.get()),
        )

    # ------------------------------------------------------------------
    # Legacy wand-mode helpers (wand is now a radio-button tool; these
    # are kept as stubs for backward compat with any calls that remain)
    # ------------------------------------------------------------------
    def _exit_wand_mode(self, silent=False):
        """No-op stub – wand state is handled entirely by selected_tool."""
        self._wand_mode = False

    def _wand_click(self, event):
        if (self.original_image or self.current_image) is None:
            self._exit_wand_mode()
            return
        target = self.current_image if self.current_image is not None else self.original_image
        pt = self._canvas_to_image_coords(self.canvas_result, event.x, event.y, target)
        if pt is None:
            return
        if self.current_image is None:
            self.status_var.set("Run Process to generate a Result image before using the Wand.")
            return

        w, h = self.current_image.size
        px, py = pt
        if px < 0 or px >= w or py < 0 or py >= h:
            return

        tolerance = int(self.wand_tolerance.get())
        rgba = np.array(self.current_image, dtype=np.uint8)

        if rgba[py, px, 3] < 5:
            self.status_var.set(
                "Magic Wand still on — that pixel is already transparent. "
                "Click a non-transparent pixel, or ESC/right-click to cancel."
            )
            return

        self._push_undo()

        # Ultra-fast OpenCV flood-fill
        rgb = rgba[:, :, :3].copy()
        mask = np.zeros((h + 2, w + 2), dtype=np.uint8)
        mask[1:-1, 1:-1] = (rgba[:, :, 3] == 0).astype(np.uint8)

        diff = int(tolerance)
        cv2.floodFill(
            rgb, mask, (px, py), (0, 0, 0),
            loDiff=(diff, diff, diff),
            upDiff=(diff, diff, diff),
            flags=4 | cv2.FLOODFILL_MASK_ONLY | (255 << 8)
        )
        filled = (mask[1:-1, 1:-1] == 255)
        rgba[filled, 3] = 0
        matches = int(np.sum(filled))

        self.current_image = Image.fromarray(rgba)
        self._result_version += 1
        self._invalidate_render_cache("result")
        self._update_undo_button()
        self._redraw_all()
        self.btn_save.config(state=tk.NORMAL)
        self.status_var.set(
            f"Wand erased {matches:,} pixels (tolerance {tolerance}). "
            f"Staying in Wand mode — click another color, or press ESC / "
            f"right-click / press Wand again to exit."
        )

    # ------------------------------------------------------------------
    # Compare mode (stable: exits after drag or ESC or right-click)
    # ------------------------------------------------------------------
    def _exit_compare_mode(self):
        was_on = self._compare_mode
        self._compare_mode = False
        self._compare_handle_id = None
        self._compare_active_drag = False
        self._compare_split_x = None
        # Always purge any lingering compare canvas items (the "BEFORE"/"AFTER"
        # labels and the split line) – these have tag "compare_item".
        try:
            self.canvas_result.delete("compare_item")
        except Exception:
            pass
        if was_on:
            self._redraw_all()

    def _on_compare_enter(self):
        if not (self.original_image and self.current_image):
            self.status_var.set(
                "Load an image and run Process before using Compare."
            )
            self.selected_tool.set(TOOL_ERASER)
            self._update_cursor_and_tool_highlight()
            return False
        self._compare_mode = True
        self._compare_split_x = None
        self.status_var.set(
            "Compare: DRAG on Result pane to slide before/after. Release, ESC, "
            "or right-click when done."
        )
        self._draw_compare()
        return True

    def _compare_start(self, event):
        if not self._compare_mode:
            if not self._on_compare_enter():
                return
        bbox = self._image_to_canvas_bbox(self.canvas_result,
                                            self.current_image)
        if bbox is None:
            return
        x0, y0, x1, y1 = bbox
        if event.x < x0 or event.x > x1:
            return
        self._compare_split_x = event.x
        self._compare_handle_id = event.x
        self._compare_active_drag = True
        self._draw_compare()

    def _compare_move(self, event):
        bbox = self._image_to_canvas_bbox(self.canvas_result,
                                            self.current_image)
        if bbox is None:
            return
        x0, _, x1, _ = bbox
        self._compare_split_x = min(max(event.x, x0), x1)
        self._draw_compare()

    def _draw_compare(self):
        if not (self.original_image and self.current_image):
            return

        # Build display-ready thumbnails at current zoom/pan
        img = self.current_image
        raw = self._image_to_canvas_bbox_raw(self.canvas_result, img.size,
                                               self._zoom_factor)
        if raw is None:
            return
        x0, y0, disp_w, disp_h = raw
        if disp_w <= 0 or disp_h <= 0:
            return

        # Check compare cache to avoid expensive resizing on drag
        cc = self._compare_cache
        if (cc["size"] != (disp_w, disp_h) or
                cc["version"] != self._result_version or
                cc["orig_rgb"] is None or
                cc["res_rgb"] is None):
            orig_thumb = self.original_image.resize((disp_w, disp_h),
                                                      Image.Resampling.BILINEAR)
            res_thumb = self.current_image.resize((disp_w, disp_h),
                                                   Image.Resampling.BILINEAR)

            checker = self._create_checkerboard(disp_w, disp_h)
            checker.paste(res_thumb, (0, 0), res_thumb)
            res_rgb = checker.convert("RGB")

            if orig_thumb.mode == "RGBA":
                checker_orig = self._create_checkerboard(disp_w, disp_h)
                checker_orig.paste(orig_thumb, (0, 0), orig_thumb)
                orig_rgb = checker_orig.convert("RGB")
            else:
                orig_rgb = orig_thumb.convert("RGB")

            cc["orig_rgb"] = orig_rgb
            cc["res_rgb"] = res_rgb
            cc["size"] = (disp_w, disp_h)
            cc["version"] = self._result_version
        else:
            orig_rgb = cc["orig_rgb"]
            res_rgb = cc["res_rgb"]

        if self._compare_split_x is None:
            self._compare_split_x = (x0 + x0 + disp_w) // 2

        split_local = max(0, min(disp_w, self._compare_split_x - x0))
        combined = Image.new("RGB", (disp_w, disp_h))
        combined.paste(orig_rgb.crop((0, 0, split_local, disp_h)), (0, 0))
        combined.paste(res_rgb.crop((split_local, 0, disp_w, disp_h)),
                        (split_local, 0))

        photo = ImageTk.PhotoImage(combined)
        self._photo_compare = photo

        self.canvas_result.delete("compare_item")
        self.canvas_result.create_image(x0, y0, image=photo, anchor=tk.NW, tags="compare_item")

        split_on_canvas = x0 + split_local
        self.canvas_result.create_line(
            split_on_canvas, y0, split_on_canvas, y0 + disp_h,
            fill="#4a9eff", width=2, tags="compare_item"
        )
        my = y0 + disp_h // 2
        self.canvas_result.create_rectangle(
            split_on_canvas - 10, my - 16, split_on_canvas + 10, my + 16,
            fill="#4a9eff", outline="#ffffff", tags="compare_item"
        )
        self.canvas_result.create_text(
            split_on_canvas, my, text="◀▶", fill="#ffffff",
            font=("Segoe UI", 10, "bold"), tags="compare_item"
        )
        self.canvas_result.create_text(
            x0 + 6, y0 + 6, anchor="nw", text="BEFORE", fill="#000000",
            font=("Segoe UI", 9, "bold"), tags="compare_item"
        )
        self.canvas_result.create_text(
            x0 + disp_w - 6, y0 + 6, anchor="ne", text="AFTER", fill="#ffffff",
            font=("Segoe UI", 9, "bold"), tags="compare_item"
        )

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------
    def run(self):
        self.root.mainloop()


def run_gui():
    """Launch the GUI application."""
    app = BackgroundRemoverGUI()
    app.run()


if __name__ == "__main__":
    run_gui()

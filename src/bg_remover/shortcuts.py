"""Keyboard shortcut resolution for the desktop interface."""

from typing import Optional


TOOL_SHORTCUTS = {
    "x": "eraser",
    "o": "restorer",
    "w": "wand",
    "m": "magic_eraser",
    "c": "compare",
    "h": "pan",
}


def resolve_shortcut(
    keysym: str,
    *,
    control: bool,
    shift: bool,
    alt: bool,
    editable: bool,
) -> Optional[str]:
    """Return the action for a keyboard event, if one is assigned."""
    key = keysym.lower()

    if control:
        if key == "o":
            return "open_folder" if shift else "open_image"
        if key == "s":
            return "save"
        if key == "z":
            return "undo"
        if key in ("return", "kp_enter"):
            return "process"
        if key == "0":
            return "zoom_fit"
        if key == "1":
            return "zoom_100"
        if key in ("plus", "equal", "kp_add"):
            return "zoom_in"
        if key in ("minus", "kp_subtract"):
            return "zoom_out"
        return None

    if alt or editable:
        return None

    if key == "f5":
        return "process"
    if key == "bracketleft":
        return "brush_smaller"
    if key == "bracketright":
        return "brush_larger"
    if key in TOOL_SHORTCUTS:
        return f"tool:{TOOL_SHORTCUTS[key]}"
    return None

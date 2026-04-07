from .video import extract_frames_from_video

__all__ = [
    "get_device",
    "print_device_info",
    "parse_normalization_values",
    "build_transform",
    "extract_state_dict",
    "remap_state_dict",
    "load_model_from_checkpoint",
    "extract_frames_from_video",
]


def __getattr__(name: str):
    if name in ("get_device", "print_device_info"):
        from .hardware import get_device, print_device_info
        return get_device if name == "get_device" else print_device_info
    if name in ("parse_normalization_values", "build_transform"):
        from .data_utils import parse_normalization_values, build_transform
        return parse_normalization_values if name == "parse_normalization_values" else build_transform
    if name in ("extract_state_dict", "remap_state_dict", "load_model_from_checkpoint"):
        from .model_utils import extract_state_dict, remap_state_dict, load_model_from_checkpoint
        return {"extract_state_dict": extract_state_dict, "remap_state_dict": remap_state_dict,
                "load_model_from_checkpoint": load_model_from_checkpoint}[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

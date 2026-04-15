from .splits import (
    DEFAULT_MANIFEST_NAME,
    ensure_split_manifest,
    frames_directory_has_images,
    manifest_path_for_frames_dir,
    write_split_manifest,
)

__all__ = [
    "VideoDataset",
    "build_dataloaders",
    "DEFAULT_MANIFEST_NAME",
    "ensure_split_manifest",
    "frames_directory_has_images",
    "manifest_path_for_frames_dir",
    "write_split_manifest",
]


def __getattr__(name: str):
    if name == "VideoDataset":
        from .dataset import VideoDataset
        return VideoDataset
    if name == "build_dataloaders":
        from .dataset import build_dataloaders
        return build_dataloaders
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

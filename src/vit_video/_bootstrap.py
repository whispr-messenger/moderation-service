import sys
from pathlib import Path


def setup():
    pkg_dir = Path(__file__).resolve().parent
    src_dir = pkg_dir.parent
    if src_dir.name == "src" and str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))
    elif pkg_dir.name == "vit_video" and str(pkg_dir.parent) not in sys.path:
        sys.path.insert(0, str(pkg_dir.parent))

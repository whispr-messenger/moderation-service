"""
Moderation Service Package
"""

# Import submodules for easy access
from .efficientnet_lite import classify_image

__all__ = [
    'classify_image'
]
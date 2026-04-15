"""
Canonical mapping from fine-grained food classes to the 3-tier health verdict.

Single source of truth for `{healthy, unhealthy, not_food}` rollup. Used by:
  - `app_mockup_demo/app.py` (demo UI)
  - `src/mobilenet_v3_small/validation/` (Keras model validators)
  - `src/efficientnet_lite_gpu/validation/` (Keras model validators)
  - `src/vit_video/` notebook (training & evaluation)

Classes follow DATASET.md §3a (legacy 16-class taxonomy).
Edit here to change the rollup globally -- do not fork into model directories.
"""

from __future__ import annotations


HEALTH_LABELS: dict[str, str] = {
    # Healthy (8)
    "fruits": "healthy",
    "vegetables": "healthy",
    "salads": "healthy",
    "seafood": "healthy",
    "grilled_meat": "healthy",
    "grain_bowls": "healthy",
    "soups": "healthy",
    "smoothies": "healthy",
    # Unhealthy (7)
    "burgers": "unhealthy",
    "pizza": "unhealthy",
    "fried_food": "unhealthy",
    "desserts": "unhealthy",
    "candy_sweets": "unhealthy",
    "salty_snacks": "unhealthy",
    "sugary_drinks": "unhealthy",
    # Non-food (1)
    "not_food": "not_food",
}

UNHEALTHY_CLASSES: frozenset[str] = frozenset(
    k for k, v in HEALTH_LABELS.items() if v == "unhealthy"
)
HEALTHY_CLASSES: frozenset[str] = frozenset(
    k for k, v in HEALTH_LABELS.items() if v == "healthy"
)

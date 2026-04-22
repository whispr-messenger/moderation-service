import sys
from pathlib import Path

# Put fastapi-mvp on sys.path so `from src...` imports work regardless of
# the pytest rootdir convention.
ROOT = Path(__file__).parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

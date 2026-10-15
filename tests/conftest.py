# tests/conftest.py
from pathlib import Path
import sys

# Ensure project root is always in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import get_settings
# ... keep the rest of conftest.py unchanged
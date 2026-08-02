"""Pozwala uruchomić `pytest` bez instalacji pakietu — dokłada src/ do ścieżki."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

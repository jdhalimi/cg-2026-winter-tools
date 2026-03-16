"""Shared fixtures for the public toolkit test suite."""

import sys
from pathlib import Path

# Ensure the public/ root is on sys.path so simulator.* imports work
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

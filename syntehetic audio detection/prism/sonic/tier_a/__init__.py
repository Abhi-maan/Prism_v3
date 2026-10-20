"""
PRISM — Tier A: Spectral Artifact Extractor
============================================
Zero-shot, heuristic-based audio authenticity detector.

Exports:
    analyze_file(file_path) -> dict
    analyze_dataset(folder_path, output_csv) -> None
"""

from .analyzer import analyze_file, analyze_dataset

__all__ = ["analyze_file", "analyze_dataset"]
__version__ = "1.0.0"

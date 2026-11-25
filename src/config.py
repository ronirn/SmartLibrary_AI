"""Lokasi file terpusat untuk SmartLibrary AI."""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA = PROJECT_ROOT / "dataset" / "raw" / "books_with_categories.csv"
CLEAN_DATA = PROJECT_ROOT / "dataset" / "processed" / "books_clean.csv"
INDONESIA_DATA = PROJECT_ROOT / "dataset" / "processed" / "books_indonesia.csv"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"

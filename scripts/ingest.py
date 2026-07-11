"""
Run document ingestion pipeline.

Usage:
    python scripts/ingest.py

Downloads needed first:
    - EU AI Act PDF → data/eu_ai_act.pdf
    - GDPR PDF      → data/gdpr.pdf
    - CSVs          → data/ai_act_timeline.csv, risk_classification.csv, penalty_structure.csv
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

# Add src to path so imports work
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

logging.basicConfig(level=logging.INFO, format="%(levelname)s — %(message)s")

from regulation_advisor.ingestion.pipeline import run_ingestion

if __name__ == "__main__":
    run_ingestion(
        data_dir=Path("data"),
        index_dir=Path("data/index"),
    )

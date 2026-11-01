"""Thin wrapper so `python scripts/ingest_portfolio.py` works like `portfolio-ai-ingest`."""

import sys

from portfolio_ai.ingest_cli import main

if __name__ == "__main__":
    sys.exit(main())

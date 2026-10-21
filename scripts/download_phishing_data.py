"""Verify a local phishing URL dataset.

This script does not crawl or visit suspicious links. It only checks a CSV you
obtained yourself.

Expected schema:
    url,label,source,timestamp,url_id(optional)
    label: 1 = phishing/malicious, 0 = benign
    source: dataset or feed name
    timestamp: ISO-8601

Possible sources (check licence and terms before use):
  - PhishTank (registration/API key, verification status)
  - OpenPhish (public feed; URLs are live malicious links -- do not visit)
  - URLhaus by abuse.ch (terms apply)
  - Tranco or a curated benign list for negatives

Never commit the downloaded data.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--path", type=Path, default=Path("data/raw/phishing_urls.csv"))
    args = parser.parse_args()
    if not args.path.exists():
        print(f"Dataset not found: {args.path}", file=sys.stderr)
        print(__doc__)
        return 2
    frame = pd.read_csv(args.path)
    required = {"url", "label", "source", "timestamp"}
    missing = required - set(frame.columns)
    if missing:
        print(f"Missing columns: {sorted(missing)}", file=sys.stderr)
        return 1
    print(f"Rows: {len(frame):,}")
    print(f"Columns: {list(frame.columns)}")
    print(f"Label counts:\n{frame['label'].value_counts(dropna=False).to_string()}")
    print(f"Sources:\n{frame['source'].value_counts().to_string()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

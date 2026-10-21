"""Write the synthetic DEMO URL dataset."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import DEMO_DATA_DIR, DEMO_N_URLS, RANDOM_STATE  # noqa: E402
from src.data_loader import audit_dataset, write_demo_dataset  # noqa: E402
from src.demo_data import generate_demo_urls  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-urls", type=int, default=DEMO_N_URLS)
    parser.add_argument("--seed", type=int, default=RANDOM_STATE)
    parser.add_argument("--output", type=Path, default=DEMO_DATA_DIR)
    args = parser.parse_args()
    frame = generate_demo_urls(n_urls=args.n_urls, seed=args.seed)
    path = write_demo_dataset(frame, args.output)
    audit = audit_dataset(frame)
    print(f"Wrote {path}")
    print(f"Rows: {audit['n_rows']:,} | positives: {audit['positive_rate']:.2%} | DEMO synthetic")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

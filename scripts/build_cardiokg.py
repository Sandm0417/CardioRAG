"""Build CardioKG from guideline PDFs.

Usage:
    python scripts/build_cardiokg.py --disease cad --pdf-dir data/raw_guidelines --output data/cardiokg
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from cardiorag.cardiokg_builder import build_cardiokg


def main() -> None:
    parser = argparse.ArgumentParser(description="Build CardioKG from guideline PDFs")
    parser.add_argument(
        "--disease",
        type=str,
        required=True,
        choices=["cad", "hf", "af", "all"],
        help="Disease to process (cad/hf/af/all)",
    )
    parser.add_argument(
        "--pdf-dir",
        type=Path,
        default=Path("data/raw_guidelines"),
        help="Directory containing guideline PDFs",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/cardiokg"),
        help="Output directory for JSONL files",
    )
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)

    diseases = ["cad", "hf", "af"] if args.disease == "all" else [args.disease]

    for disease in diseases:
        print(f"\n{'='*60}")
        print(f"Building CardioKG for: {disease.upper()}")
        print(f"{'='*60}")

        result = build_cardiokg(
            pdf_dir=args.pdf_dir,
            output_dir=args.output,
            disease=disease,
        )

        print(f"  Nodes: {result['node_count']}")
        print(f"  Relations: {result['relation_count']}")
        print(f"  Sources: {', '.join(result['guideline_sources'])}")
        print(f"  Output: {result['output_path']}")

    print("\n✅ CardioKG build complete.")


if __name__ == "__main__":
    main()

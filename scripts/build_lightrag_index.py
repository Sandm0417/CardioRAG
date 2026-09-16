"""Build LightRAG index from CardioKG JSONL files.

Usage:
    python scripts/build_lightrag_index.py --cardiokg-dir data/cardiokg --output lightrag_index/
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from cardiorag.lightrag_adapter import LightRAGAdapter


def main() -> None:
    parser = argparse.ArgumentParser(description="Build LightRAG index from CardioKG")
    parser.add_argument(
        "--cardiokg-dir",
        type=Path,
        default=Path("data/cardiokg"),
        help="Directory containing cad.jsonl, hf.jsonl, af.jsonl",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("lightrag_index"),
        help="Output directory for LightRAG index",
    )
    args = parser.parse_args()

    # Check prerequisites
    for disease in ["cad", "hf", "af"]:
        jsonl_path = args.cardiokg_dir / f"{disease}.jsonl"
        if not jsonl_path.exists():
            print(f"⚠️  Missing: {jsonl_path} — run build_cardiokg.py first")
            sys.exit(1)

    # Initialize LightRAG
    print("Initializing LightRAG adapter...")
    rag = LightRAGAdapter(working_dir=args.output)
    rag.initialize()

    # Build index
    print("Building LightRAG index from CardioKG...")
    stats = rag.build_index(args.cardiokg_dir)

    print(f"✅ LightRAG index built: {stats}")
    print(f"   Nodes: {stats['nodes']}, Relations: {stats['relations']}, Chunks: {stats['chunks']}")


if __name__ == "__main__":
    main()

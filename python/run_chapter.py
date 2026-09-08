"""Portable entry point: python run_chapter.py 5."""
from pathlib import Path
import argparse
import json
import runpy

parser = argparse.ArgumentParser()
parser.add_argument("chapter", type=int, choices=range(1, 21))
args = parser.parse_args()
path = Path(__file__).resolve().parent / "snippets"
namespace = runpy.run_path(str(path / f"ch{args.chapter:02d}.py"))
print(json.dumps(namespace["RESULT"], ensure_ascii=False, indent=2))

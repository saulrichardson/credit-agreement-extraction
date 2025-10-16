#!/usr/bin/env python3
"""Convert HTML segments into plain-text files with Markdown tables.

Usage
-----
    python scripts/export_markdown_text.py \
        --input-dir /path/to/html_segments \
        --output-dir /path/to/markdown_segments
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd
from bs4 import BeautifulSoup


def html_to_markdown_text(html_path: Path) -> str:
    soup = BeautifulSoup(html_path.read_text(encoding="utf-8", errors="ignore"), "lxml")

    for table in soup.find_all("table"):
        try:
            df_list = pd.read_html(str(table))
            markdown = df_list[0].to_markdown(index=False)
        except ValueError:
            markdown = table.get_text(separator="\n").strip()
        table.replace_with(soup.new_string(f"\n{markdown}\n"))

    text = soup.get_text(separator="\n")
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert HTML segment files into plain text with Markdown tables."
    )
    parser.add_argument("--input-dir", required=True, type=Path, help="Directory of HTML (.txt) segments")
    parser.add_argument("--output-dir", required=True, type=Path, help="Destination directory for Markdown text files")
    args = parser.parse_args()

    input_dir: Path = args.input_dir
    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    for html_path in sorted(input_dir.glob("*.txt")):
        markdown_text = html_to_markdown_text(html_path)
        out_path = output_dir / html_path.name
        out_path.write_text(markdown_text, encoding="utf-8")


if __name__ == "__main__":
    main()

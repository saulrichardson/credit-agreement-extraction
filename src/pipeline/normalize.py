from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Tuple
import re

from bs4 import BeautifulSoup

from .config import Paths, record_manifest
from .utils import manifest_items

# Bullet detection (for normalization and anchor splitting)
_bullet_re = re.compile(r"^\s*(?:[-•]|\([a-zA-Z0-9ivxIVX]+\))\s+")

# Abbreviations to avoid sentence splits (finance/legal heavy)
_abbr_tokens = {
    "mr", "ms", "mrs", "dr", "inc", "ltd", "corp", "co", "no",
    "art", "sec", "ex", "fig", "st", "u.s", "u.s.a", "llc", "lp", "l.p", "l.l.p",
    "llp", "plc", "n.a", "na", "cf.", "cf", "vs", "al.", "eq.", "dept", "div",
    "assoc", "approx", "appx", "dept.", "div.", "gov.", "adj.", "adm.", "agt.",
}

# Conservative sentence splitter: split on punctuation only if preceding token is >=2 chars and not an abbreviation.
# Accepts either upper or lower case after the punctuation.
_sent_splitter = re.compile(r"(?<=[A-Za-z0-9]{2}[.!?])\s+(?=[A-Za-z])")


def _sentence_split(paragraph: str) -> List[str]:
    parts = []
    start = 0
    for match in _sent_splitter.finditer(paragraph):
        end = match.start()
        segment = paragraph[start:end]
        token = segment.rstrip().split()[-1].rstrip(".!?").lower() if segment.strip() else ""
        if token in _abbr_tokens or len(token) <= 1:
            continue  # skip split; keep going
        parts.append(paragraph[start:end])
        start = match.end()
    parts.append(paragraph[start:])  # tail
    parts = [p for p in parts if p.strip()]

    # Fallback: if no splits and block is long (>400 chars), force a split on first safe punctuation+space
    if len(parts) == 1 and len(parts[0]) > 400:
        m = re.search(r"[.!?]\s+", parts[0])
        if m:
            idx = m.end()
            first, second = parts[0][:idx].strip(), parts[0][idx:].strip()
            parts = [p for p in (first, second) if p]

    # Merge very short fragments into previous sentence to avoid tiny anchors
    merged: List[str] = []
    for seg in parts:
        if merged and len(seg.strip()) < 10:
            merged[-1] = merged[-1].rstrip() + " " + seg.lstrip()
        else:
            merged.append(seg)

    # Pairing: do not split inside tight parentheses/quotes if we accidentally did
    paired: List[str] = []
    for seg in merged:
        # If segment looks like "(a)" or "(a) something" alone, glue with next if exists
        if re.fullmatch(r"\([a-zA-Z0-9]\)", seg.strip()) and paired:
            paired[-1] = paired[-1].rstrip() + " " + seg.lstrip()
        else:
            paired.append(seg)
    return paired


def _normalize_non_table_text(text: str) -> str:
    text = text.replace("\r", "")
    text = re.sub(r"\n{3,}", "\n\n", text)

    paras = text.split("\n\n")
    norm_paras = []
    for para in paras:
        lines = para.split("\n")
        # If any line is a bullet, keep line breaks to preserve list structure
        if any(_bullet_re.match(ln.lstrip()) for ln in lines):
            kept_lines = []
            for ln in lines:
                ln_norm = re.sub(r"[ \t]{2,}", " ", ln)
                kept_lines.append(ln_norm.strip())
            norm_paras.append("\n".join(kept_lines).strip())
        else:
            # join lines with spaces
            joined = " ".join(ln.strip() for ln in lines if ln.strip())
            joined = re.sub(r"[ \t]{2,}", " ", joined)
            norm_paras.append(joined)
    return "\n\n".join([p for p in norm_paras if p])


def _canonicalize_html(html_path: Path) -> str:
    raw = html_path.read_text(errors="ignore")
    soup = BeautifulSoup(raw, "lxml")

    # Convert tables to lightweight Markdown; wrap with plain markers.
    for table in soup.find_all("table"):
        rows = []
        for tr in table.find_all("tr"):
            cells = [c.get_text(" ", strip=True) for c in tr.find_all(["td", "th"])]
            if cells:
                rows.append(cells)
        if not rows:
            table.replace_with("")
            continue
        ncols = max(len(r) for r in rows)
        normalized_rows = [r + [""] * (ncols - len(r)) for r in rows]
        lines = []
        header = normalized_rows[0]
        lines.append("| " + " | ".join(header) + " |")
        lines.append("| " + " | ".join(["---"] * ncols) + " |")
        for r in normalized_rows[1:]:
            lines.append("| " + " | ".join(r) + " |")
        table_md = "\n".join(lines)
        table.replace_with(soup.new_string(f"\n[[TABLE]]\n{table_md}\n[[/TABLE]]\n"))

    text = soup.get_text("\n")

    # Split out tables, normalize non-table segments separately to fix whitespace
    parts = []
    last = 0
    for match in re.finditer(r"\[\[TABLE\]\]\s*(.*?)\s*\[\[/TABLE\]\]", text, flags=re.DOTALL):
        start, end = match.span()
        pre = text[last:start]
        if pre.strip():
            parts.append(_normalize_non_table_text(pre))
        parts.append(text[start:end])  # keep table block as-is
        last = end
    if last < len(text):
        tail = text[last:]
        if tail.strip():
            parts.append(_normalize_non_table_text(tail))

    text = "\n\n".join([p for p in parts if p])

    # Normalize bullets to a single marker "- " at line starts where appropriate
    norm_lines = []
    for line in text.splitlines():
        stripped = line.lstrip()
        if _bullet_re.match(stripped):
            after = _bullet_re.sub("", stripped, count=1).lstrip()
            norm_lines.append(f"- {after}")
        else:
            # collapse multiple internal spaces to a single space
            norm_lines.append(re.sub(r"[ \t]{2,}", " ", line))
    return "\n".join(norm_lines)


def _table_spans(text: str) -> List[Tuple[int, int]]:
    return [match.span() for match in re.finditer(r"\[\[TABLE\]\]\s*(.*?)\s*\[\[/TABLE\]\]", text, flags=re.DOTALL)]


def _split_blocks(text: str) -> List[Tuple[int, int, str, str]]:
    """Split text into anchors: whole tables, bullets, sentences."""

    anchors: List[Tuple[int, int, str, str]] = []
    anchor_idx = 1

    cursor = 0
    for table_start, table_end in _table_spans(text):
        pre_segment = text[cursor:table_start]
        anchors, anchor_idx = _split_non_table(pre_segment, anchors, anchor_idx, base_offset=cursor)

        anchors.append((table_start, table_end, "table", f"A{anchor_idx:04d}"))
        anchor_idx += 1
        cursor = table_end

    trailing = text[cursor:]
    if trailing:
        anchors, anchor_idx = _split_non_table(trailing, anchors, anchor_idx, base_offset=cursor)

    return anchors


def _split_non_table(segment: str, anchors: List[Tuple[int, int, str, str]], anchor_idx: int, base_offset: int) -> Tuple[List[Tuple[int, int, str, str]], int]:
    parts = segment.split("\n\n")
    offset = base_offset
    for part in parts:
        block = part.strip("\n")
        if not block.strip():
            offset += len(part) + 2
            continue
        start = segment.find(block, offset - base_offset) + base_offset
        end = start + len(block)

        lines = block.split("\n")
        bullet_flags = [bool(_bullet_re.match(ln)) for ln in lines]
        if sum(bullet_flags) >= 2 and sum(bullet_flags) / max(1, len(lines)) > 0.5:
            current: List[str] = []
            saved_start = None
            line_cursor = start
            for ln in lines:
                ln_pos = segment.find(ln, line_cursor - base_offset) + base_offset
                if _bullet_re.match(ln):
                    if current:
                        item_text = "\n".join(current)
                        item_start = saved_start if saved_start is not None else start
                        item_end = item_start + len(item_text)
                        anchors.append((item_start, item_end, "bullet", f"A{anchor_idx:04d}"))
                        anchor_idx += 1
                        current = []
                    saved_start = ln_pos
                current.append(ln)
                line_cursor = ln_pos + len(ln) + 1
            if current:
                item_text = "\n".join(current)
                item_start = saved_start if saved_start is not None else start
                item_end = item_start + len(item_text)
                anchors.append((item_start, item_end, "bullet", f"A{anchor_idx:04d}"))
                anchor_idx += 1
        else:
            sentences = _sentence_split(block)
            sent_offset = start
            for sent in sentences:
                sent_start = segment.find(sent, sent_offset - base_offset) + base_offset
                sent_end = sent_start + len(sent)
                anchors.append((sent_start, sent_end, "sentence", f"A{anchor_idx:04d}"))
                anchor_idx += 1
                sent_offset = sent_end
        offset = end
    return anchors, anchor_idx


def build_prompt_views(paths: Paths, manifest: Dict) -> None:
    pv_root = paths.normalized_dir
    pv_root.mkdir(parents=True, exist_ok=True)

    items = manifest_items(manifest)
    prompt_view_paths = {}
    for item in items:
        item_id = item.get("item_id")
        html_file = Path(item.get("path"))
        if not html_file.exists():
            raise FileNotFoundError(f"Expected HTML missing: {html_file}")
        out_dir = pv_root / item_id
        out_dir.mkdir(parents=True, exist_ok=True)

        canonical_text = _canonicalize_html(html_file)
        (out_dir / "canonical.txt").write_text(canonical_text)
        (out_dir / "prompt_view.txt").write_text(canonical_text)
        anchors = _split_blocks(canonical_text)
        with (out_dir / "anchors.tsv").open("w") as f:
            f.write("anchor_id\tanchor_type\tstart\tend\tlabel\n")
            for start, end, label, aid in anchors:
                f.write(f"{aid}\t{label}\t{start}\t{end}\t{label}\n")
        annotated_lines = []
        for start, end, label, aid in anchors:
            block = canonical_text[start:end]
            annotated_lines.append(f"[[{aid}]]\n{block}\n")
        (out_dir / "prompt_view_annotated.txt").write_text("\n".join(annotated_lines))
        prompt_view_paths[item_id] = str(out_dir / "prompt_view.txt")

    manifest_path = paths.manifest_path
    manifest["normalized"] = prompt_view_paths
    # keep legacy key for backward compatibility
    manifest["prompt_views"] = prompt_view_paths
    record_manifest(manifest_path, manifest)

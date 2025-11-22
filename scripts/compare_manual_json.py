#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
import pandas as pd
import math

def norm_percent(val):
    if pd.isna(val):
        return None
    s = str(val).strip().replace('%','')
    try:
        x = float(s)
    except ValueError:
        return None
    # assume already percent units; round to 4 decimal places
    return round(x, 4)

def collect_manual(manual_path):
    df = pd.read_excel(manual_path)
    vals = {}
    for acc, grp in df.groupby('accession'):
        nums = [norm_percent(v) for v in grp['value_percent']]
        nums = [v for v in nums if v is not None]
        vals[acc] = nums
    return vals

def collect_json(json_dir: Path):
    vals = {}
    for path in json_dir.glob('*.txt'):
        acc = path.stem.replace('_snippets','')
        try:
            data = json.loads(path.read_text())
        except Exception:
            continue
        nums = []
        for fac in data.get('facilities', []):
            for rate in fac.get('rates', []):
                for bl in rate.get('by_level', []):
                    if 'bps' in bl:
                        nums.append(round(bl['bps']/100.0, 4))
                    elif 'amount' in bl:
                        # skip currency amounts
                        continue
        vals[acc] = nums
    return vals

def compare(manual_vals, json_vals):
    rows = []
    for acc, m_vals in manual_vals.items():
        j_vals = json_vals.get(acc, [])
        m_set = set(m_vals)
        j_set = set(j_vals)
        missing = sorted(m_set - j_set)
        extra = sorted(j_set - m_set)
        rows.append({
            'accession': acc,
            'manual_count': len(m_vals),
            'json_count': len(j_vals),
            'missing_values': missing,
            'extra_values': extra
        })
    return rows

def main():
    ap = argparse.ArgumentParser(description="Compare manual pricing percents vs JSON outputs")
    ap.add_argument('--manual', required=True, help='Path to manual Excel file')
    ap.add_argument('--json-dir', required=True, help='Directory with structured JSON outputs')
    ap.add_argument('--report', required=True, help='Path to write report txt')
    args = ap.parse_args()

    manual_vals = collect_manual(args.manual)
    json_vals = collect_json(Path(args.json_dir))
    rows = compare(manual_vals, json_vals)

    lines = []
    mismatches = 0
    for r in rows:
        if r['missing_values'] or r['extra_values']:
            mismatches += 1
        lines.append(
            f"{r['accession']}: manual={r['manual_count']} json={r['json_count']} "
            f"missing={r['missing_values']} extra={r['extra_values']}"
        )
    Path(args.report).write_text("\n".join(lines)+"\n", encoding='utf-8')
    print(f"Wrote report to {args.report}; mismatches={mismatches}/{len(rows)}")
    if mismatches:
        exit(1)

if __name__ == '__main__':
    main()

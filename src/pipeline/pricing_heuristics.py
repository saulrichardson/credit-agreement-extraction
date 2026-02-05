from __future__ import annotations


def looks_like_pricing_anchor(text: str) -> bool:
    hay = (text or "").lower()
    keywords = [
        "applicable margin",
        "applicable rate",
        "default rate",
        "fronting fee",
        "letter of credit fee",
        "letter of credit fees",
        "l/c fee rate",
        "commitment fee",
        "facility fee rate",
        "unused fee",
        "utilization",
        "basis point",
        "bps",
        "pricing grid",
        "sustainability grid",
        "sustainability metric grid",
    ]
    # Do NOT treat '%' alone as a pricing signal; that explodes context size on long exhibits.
    # We only include anchors when explicit pricing keywords are present.
    return any(k in hay for k in keywords)


def looks_like_definition_anchor(text: str) -> bool:
    # Conservative: include only classic definitions we want nearby for pricing.
    hay = (text or "").strip()
    if not hay:
        return False
    if " means " not in hay and " shall mean " not in hay:
        return False
    targets = [
        '"Base Rate"',
        '"ABR"',
        '"Eurocurrency Rate"',
        '"SOFR"',
        '"Term SOFR"',
        '"Adjusted Term SOFR Rate"',
        '"Adjusted CD Rate"',
        '"CD Base Rate"',
        '"London Interbank Offered Rate"',
        '"Applicable Margin"',
        '"Applicable Rate"',
        '"Facility Fee"',
        '"Commitment Fee"',
        '"L/C Fee',
    ]
    return any(t in hay for t in targets)


def looks_like_regime_intro(text: str) -> bool:
    """Heuristic: lines that introduce a distinct pricing applicability regime."""

    hay = (text or "").strip().lower()
    if not hay:
        return False

    # Very common in amendments/agreements: bullet-like regime intros.
    if hay.startswith("-"):
        if any(k in hay for k in ["prior to", "on or after", "from and after", "until", "while", "during"]):
            return True
        if hay.startswith("- if "):
            return True

    # Common "alternate grid" intros.
    if "notwithstanding" in hay and any(k in hay for k in ["if ", "while ", "from and after", "on and after"]):
        return True

    return False


def looks_like_adjustment_anchor(text: str) -> bool:
    """Heuristic: footnotes/clauses that adjust a grid by +/- X bps."""

    hay = (text or "").lower()
    if not hay:
        return False
    has_bps = any(k in hay for k in ["basis point", "basis points", " bps", " bp"])
    if not has_bps:
        return False
    return any(k in hay for k in ["reduced", "increased", "additional", "decreased", "increase", "reduce"])


"""Chemistry helpers for building and querying the molecule index.

Query-token classification (is this a formula, a SMILES, or a name?) and the
RDKit descriptor derivation used at build time. Kept below the search backends
so both the builder and the facade can use them.
"""

from __future__ import annotations

import re
from typing import Any

from rpfr_visualizer.io.search.models import RowFilter, SizeFilter

# Hill-style molecular formulas, e.g. C3H7NO2, C6H6, H2O
_FORMULA_RE = re.compile(r"^(?:[A-Z][a-z]?\d*)+$")
_ELEMENT_TOKEN = re.compile(r"([A-Z][a-z]?)(\d*)")
_HILL_ORDER = ("C", "H")  # then alphabetical for the rest

#: Subscript and superscript digits, as they arrive when a formula is pasted
#: out of a paper or a rendered page.
#:
#: Translating them is a correctness fix, not a courtesy. _ELEMENT_TOKEN only
#: matches ASCII digits, so "C\u2086H\u2081\u2084O" parsed as C, H, O with every count
#: dropped -- and normalize_formula returned "CHO" with no sign that anything
#: had been lost. A search for hexanol quietly became a search for a different
#: molecule entirely.
_UNICODE_DIGITS = str.maketrans(
    "\u2080\u2081\u2082\u2083\u2084\u2085\u2086\u2087\u2088\u2089\u2070\u00b9\u00b2\u00b3\u2074\u2075\u2076\u2077\u2078\u2079",
    "01234567890123456789",
)


def clean_formula_text(text: str) -> str:
    """Normalise the characters of a formula before anything parses it.

    Removes whitespace -- ``C6 H14 O`` is how a formula is often written out
    loud -- and maps unicode digits to ASCII. Case is deliberately left alone:
    a formula is case-sensitive (``CO`` is carbon monoxide, ``Co`` is cobalt),
    and lowercasing would also collide with aromatic SMILES like ``c1ccccc1``.
    """
    return "".join((text or "").split()).translate(_UNICODE_DIGITS)


#: Size tokens: ``C:5-9``, ``C:7``, ``nC>3``, ``heavy<=12``, ``atoms=20``.
#: The optional leading ``n`` reads as "number of", so ``nC>3`` and ``C>3``
#: mean the same thing. Anchored, so it can never fire on part of a SMILES.
_SIZE_RE = re.compile(
    r"^n?(?P<field>[A-Za-z][A-Za-z]*)\s*(?:"
    r":\s*(?P<lo>\d+)(?:\s*-\s*(?P<hi>\d+))?"
    r"|(?P<op><=|>=|<|>|=)\s*(?P<val>\d+)"
    r")$"
)

#: Accepted spellings of a stored QC verdict, as ``qc:<word>``. Anything else
#: after ``qc:`` is not a verdict and falls through to normal classification
#: rather than being silently treated as one.
_QC_VALUES = {"pass": True, "passed": True, "ok": True, "fail": False, "failed": False}

#: Keyword fields that map to a ``molecules`` column rather than an element.
_SIZE_COLUMNS = {"heavy": "n_heavy", "atoms": "n_atoms"}

#: Ranges are stored inclusive; this stands in for "no upper bound".
_SIZE_MAX = 1_000_000


def parse_size_token(token: str) -> SizeFilter | None:
    """Parse one size token, or return None when *token* is not one.

    ``C:5-9`` and ``nC>3`` constrain an element count; ``heavy<=12`` and
    ``atoms=20`` constrain a stored column. Anything else — a name, a
    formula, a SMILES — returns None so the caller falls through to its
    normal classification ladder.
    """
    text = (token or "").strip()
    match = _SIZE_RE.match(text)
    if match is None:
        return None

    field = match.group("field")
    column = _SIZE_COLUMNS.get(field.lower())
    if column is None and not re.fullmatch(r"[A-Z][a-z]?", field):
        # Not a keyword and not an element symbol: e.g. "version:2".
        return None

    if match.group("op"):
        value = int(match.group("val"))
        low, high = {
            "<": (0, value - 1),
            "<=": (0, value),
            ">": (value + 1, _SIZE_MAX),
            ">=": (value, _SIZE_MAX),
            "=": (value, value),
        }[match.group("op")]
    else:
        low = int(match.group("lo"))
        high = int(match.group("hi")) if match.group("hi") is not None else low

    if high < low:
        return None
    if column is not None:
        return SizeFilter(column_counts=((column, low, high),))
    return SizeFilter(element_counts=((field, low, high),))


#: Attributes that together decide a stored QC verdict.
_QC_FLAGS = ("freq_qc_ok", "rpfr_qc_ok", "tr_converged", "rdkit_ok", "smiles_order_match")


def stored_qc_pass(group: Any) -> bool | None:
    """The QC verdict an HDF5 molecule group records, or None if it records none.

    One definition, used by both the SQLite index builder and the in-memory
    one. If they derived this differently the two backends would disagree about
    ``qc:pass`` while agreeing about everything else -- the kind of divergence
    the equivalence suite exists to prevent.
    """
    present = [f for f in _QC_FLAGS if f in group.attrs]
    if not present:
        return None
    ok = all(bool(group.attrs.get(f)) for f in present)
    if "n_imag_vib" in group.attrs:
        ok = ok and int(group.attrs.get("n_imag_vib", 0) or 0) == 0
    return ok


def parse_qc_token(token: str) -> RowFilter | None:
    """Parse ``qc:pass`` / ``qc:fail``, or return None when *token* is not one.

    Molecules whose verdict was never recorded match neither: "not known to
    have failed" is a weaker claim than "passed", and conflating them is how a
    QC filter ends up reassuring somebody about data it knows nothing about.
    """
    text = (token or "").strip().lower()
    if not text.startswith("qc:"):
        return None
    verdict = _QC_VALUES.get(text[3:].strip())
    if verdict is None:
        return None
    return RowFilter(qc_pass=verdict)


def split_size_tokens(tokens: list[str]) -> tuple[list[str], SizeFilter]:
    """Split *tokens* into search tokens and the merged AND'd constraint.

    Both the composition grammar (``C:5-9``, ``heavy<=12``) and the QC grammar
    (``qc:pass``) are restrictions rather than alternatives, so they are peeled
    out here and merged into one filter.
    """
    rest: list[str] = []
    size = SizeFilter()
    for token in tokens:
        parsed = parse_size_token(token)
        if parsed is None:
            parsed = parse_qc_token(token)
        if parsed is None:
            rest.append(token)
        else:
            size = size.merge(parsed)
    return rest, size


def split_search_query(query: str) -> list[str]:
    """Split a multi-token search query on commas, preserving SMARTS bodies.

    Comma-separated tokens are OR'd by :meth:`MoleculeIndex.search`. SMARTS
    patterns often contain commas (e.g. ``[NX3;H2,H1,H0]``), so a bare
    ``str.split(',')`` incorrectly fragments ``smarts:...`` queries into
    unrelated tokens (and can yield empty or wrong hits).

    If the query starts with ``smarts:``, the pattern is kept whole apart from
    any trailing size tokens. Otherwise commas separate OR alternatives (names,
    formulas, etc.).
    """
    text = (query or "").strip()
    if not text:
        return []
    if text.lower().startswith("smarts:"):
        return _split_smarts_query(text)
    return [part.strip() for part in text.split(",") if part.strip()]


def _split_smarts_query(text: str) -> list[str]:
    """A ``smarts:`` pattern plus any size tokens trailing it.

    SMARTS bodies legitimately contain commas — ``[NX3;H2,H1,H0]`` is one
    pattern, not three tokens — so the string cannot just be split on them.
    But ``smarts:[CX3](=O)[OX2H1], C:7`` still has to separate into a pattern
    and a filter. Fragments are therefore peeled off the right only for as long
    as they parse as size or QC tokens, which no SMARTS fragment does.
    """
    parts = [part.strip() for part in text.split(",")]
    trailing: list[str] = []
    while len(parts) > 1 and _is_filter_token(parts[-1]):
        trailing.append(parts.pop())
    return [",".join(parts), *reversed(trailing)]


def _is_filter_token(token: str) -> bool:
    """Whether *token* is a constraint rather than part of a pattern."""
    return parse_size_token(token) is not None or parse_qc_token(token) is not None


def looks_like_smiles(token: str) -> bool:
    """Heuristic: True when *token* is likely a SMILES string, not a common name."""
    text = token.strip()
    if not text or text.isdigit():
        return False
    if text.lower().startswith("smarts:"):
        return False
    if looks_like_formula(text):
        return False
    if any(ch in text for ch in "[]()=#@+/\\"):
        return True
    allowed = set("CNOPSBrlcnopsF0123456789[]()=#@+/-\\")
    organic = sum(1 for ch in text if ch in allowed)
    if organic / len(text) < 0.85:
        return False
    return any(ch.isdigit() for ch in text) or any(ch in "CNOPcnops" for ch in text)


def looks_like_formula(token: str) -> bool:
    """Return True for a molecular formula, in any element order.

    Hill order is what the index stores, but nobody is obliged to type it:
    ``C6H14O``, ``C6OH14`` and ``OH14C6`` are the same molecule and all three
    are accepted. :func:`normalize_formula` puts them back in Hill order for
    the lookup.
    """
    text = clean_formula_text(token)
    if not text or len(text) < 2:
        return False
    if not _FORMULA_RE.match(text):
        return False
    # Must contain at least one element with optional count; reject SMILES-like
    if any(ch in text for ch in "()=#@+/\\[]"):
        return False
    tokens = _ELEMENT_TOKEN.findall(text)
    if not tokens:
        return False
    # Reconstruct must match (reject garbage)
    rebuilt = "".join(el + (cnt if cnt else "") for el, cnt in tokens)
    if rebuilt != text or not all(el[0].isupper() for el, _ in tokens):
        return False
    # Ambiguous short tokens like CO/CN are usually SMILES, not formulas
    if not any(ch.isdigit() for ch in text) and "H" not in text:
        return False
    return True


def parse_element_counts(formula: str) -> dict[str, int]:
    """Parse a molecular formula into element → count, in any element order.

    Repeats are summed, so a condensed structural formula like ``CH3CH2OH``
    aggregates to the molecular one.
    """
    counts: dict[str, int] = {}
    for element, count_str in _ELEMENT_TOKEN.findall(clean_formula_text(formula)):
        counts[element] = counts.get(element, 0) + (int(count_str) if count_str else 1)
    return counts


def normalize_formula(formula: str) -> str:
    """Normalize a formula to Hill order (C, H, then alphabetical)."""
    counts = parse_element_counts(formula)
    if not counts:
        return ""
    parts: list[str] = []
    for element in _HILL_ORDER:
        if element in counts:
            n = counts.pop(element)
            parts.append(element if n == 1 else f"{element}{n}")
    for element in sorted(counts):
        n = counts[element]
        parts.append(element if n == 1 else f"{element}{n}")
    return "".join(parts)


def _rdkit_mol(smiles: str, *, quiet: bool = True):
    try:
        from rdkit import Chem, RDLogger
    except ImportError as exc:
        raise ImportError(
            "RDKit is required for SMILES search. Install with: uv sync --extra viz"
        ) from exc

    if quiet:
        RDLogger.DisableLog("rdApp.*")
    return Chem.MolFromSmiles(smiles)


def query_as_smarts(token: str, *, quiet: bool = True) -> str | None:
    """Return a SMARTS pattern when *token* is not valid SMILES.

    Canonical SMILES search is the primary path. This helper is the fallback
    for an unprefixed SMARTS string such as ``[CX3](=O)O``. Prefixed
    ``smarts:`` queries never come here.
    """
    if not token or not token.strip():
        return None
    try:
        from rdkit import Chem, RDLogger
    except ImportError as exc:
        raise ImportError(
            "RDKit is required for SMILES search. Install with: uv sync --extra viz"
        ) from exc

    if quiet:
        RDLogger.DisableLog("rdApp.*")

    mol = Chem.MolFromSmiles(token)
    if mol is not None:
        return Chem.MolToSmarts(mol)
    if Chem.MolFromSmarts(token) is not None:
        return token
    return None


def canonical_smiles(smiles: str, *, quiet: bool = True) -> str | None:
    """Return RDKit canonical SMILES, or None if parsing fails."""
    if not smiles or not smiles.strip():
        return None
    try:
        from rdkit import Chem
    except ImportError as exc:
        raise ImportError(
            "RDKit is required for SMILES search. Install with: uv sync --extra viz"
        ) from exc

    mol = _rdkit_mol(smiles, quiet=quiet)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, canonical=True)


def smiles_descriptors(smiles: str, *, quiet: bool = True) -> dict[str, Any]:
    """Derive canonical SMILES, formula, InChIKey, and element counts from SMILES."""
    empty: dict[str, Any] = {
        "canonical_smiles": "",
        "formula": "",
        "inchikey": "",
        "element_counts": {},
    }
    if not smiles or not smiles.strip():
        return empty
    try:
        from rdkit import Chem
        from rdkit.Chem import rdMolDescriptors
    except ImportError:
        return empty

    mol = _rdkit_mol(smiles, quiet=quiet)
    if mol is None:
        return empty

    canon = Chem.MolToSmiles(mol, canonical=True)
    formula_raw = rdMolDescriptors.CalcMolFormula(mol)
    formula = normalize_formula(formula_raw) or formula_raw
    inchikey = ""
    try:
        inchikey = Chem.MolToInchiKey(mol) or ""
    except Exception:  # noqa: BLE001 — InChI optional in some RDKit builds
        inchikey = ""
    return {
        "canonical_smiles": canon or "",
        "formula": formula,
        "inchikey": inchikey,
        "element_counts": parse_element_counts(formula),
    }


__all__ = [
    "canonical_smiles",
    "looks_like_formula",
    "looks_like_smiles",
    "normalize_formula",
    "parse_element_counts",
    "parse_qc_token",
    "parse_size_token",
    "query_as_smarts",
    "smiles_descriptors",
    "split_search_query",
    "split_size_tokens",
    "stored_qc_pass",
]

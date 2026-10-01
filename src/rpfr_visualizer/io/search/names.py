"""Chemical-name resolution for search.

Three tiers, cheapest first: the offline synonym table shipped with the package,
a persisted local cache, then PubChem. The network tier is opt-in — the
published notebook leaves it off so the panel never blocks on a request.

The synonym seed is baked into the index's ``names`` table at build time; this
module covers names the shipped index does not know.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.parse
import urllib.request
from importlib import resources
from pathlib import Path


def _load_synonym_seed() -> dict[str, str]:
    """Load offline name → SMILES synonyms shipped with the package."""
    try:
        root = resources.files("rpfr_visualizer.data")
        text = (root / "name_synonyms.json").read_text(encoding="utf-8")
        payload = json.loads(text)
    except (FileNotFoundError, OSError, json.JSONDecodeError, TypeError, AttributeError):
        # Fallback for editable installs without package data
        path = Path(__file__).resolve().parents[1] / "data" / "name_synonyms.json"
        if not path.exists():
            return {}
        payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        return {}
    return {str(k).strip().lower(): str(v).strip() for k, v in payload.items() if k and v}


_SYNONYM_SEED: dict[str, str] | None = None


def synonym_seed() -> dict[str, str]:
    """Return the offline chemical-name synonym map (cached)."""
    global _SYNONYM_SEED
    if _SYNONYM_SEED is None:
        _SYNONYM_SEED = _load_synonym_seed()
    return _SYNONYM_SEED


def name_resolve_cache_path(data_dir: Path | str | None = None) -> Path:
    """Path for persistent name → SMILES resolve cache."""
    if data_dir is None:
        data_dir = Path("data")
    data = Path(data_dir)
    if data.name == "raw":
        data = data.parent
    return data / "cache" / "name_resolve.json"


def load_name_resolve_cache(path: Path | str | None = None) -> dict[str, str]:
    """Load persisted name → canonical SMILES mappings."""
    cache_path = Path(path) if path else name_resolve_cache_path()
    if not cache_path.exists():
        return {}
    try:
        payload = json.loads(cache_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    if not isinstance(payload, dict):
        return {}
    return {str(k).strip().lower(): str(v).strip() for k, v in payload.items() if k and v}


def save_name_resolve_cache(mapping: dict[str, str], path: Path | str | None = None) -> None:
    """Persist name → SMILES mappings for offline reuse."""
    cache_path = Path(path) if path else name_resolve_cache_path()
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    existing = load_name_resolve_cache(cache_path)
    existing.update({k.strip().lower(): v for k, v in mapping.items() if k and v})
    cache_path.write_text(json.dumps(existing, indent=2, sort_keys=True), encoding="utf-8")


def resolve_common_name(name: str) -> str | None:
    """Resolve a common chemical name to canonical SMILES via PubChem."""
    token = name.strip()
    if not token:
        return None
    encoded = urllib.parse.quote(token)
    url = (
        "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/"
        f"{encoded}/property/CanonicalSMILES/JSON"
    )
    try:
        with urllib.request.urlopen(url, timeout=10) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, KeyError, OSError):
        return None
    try:
        props = payload["PropertyTable"]["Properties"][0]
        for key in ("CanonicalSMILES", "ConnectivitySMILES", "IsomericSMILES"):
            if key in props:
                return str(props[key])
    except (KeyError, IndexError, TypeError):
        return None
    return None


def resolve_name_hybrid(
    name: str,
    *,
    resolve_cache: dict[str, str] | None = None,
    resolve_cache_path: Path | str | None = None,
    allow_pubchem: bool = True,
) -> tuple[str | None, str]:
    """Resolve a chemical name via seed → cache → PubChem.

    Returns
    -------
    (smiles, source)
        source is ``local``, ``cache``, ``pubchem``, or ``miss``.
    """
    token = name.strip().lower()
    if not token:
        return None, "miss"

    seed = synonym_seed()
    if token in seed:
        return seed[token], "local"

    cache = (
        resolve_cache if resolve_cache is not None else load_name_resolve_cache(resolve_cache_path)
    )
    if token in cache:
        return cache[token], "cache"

    if not allow_pubchem:
        return None, "miss"

    resolved = resolve_common_name(name)
    if resolved:
        save_name_resolve_cache({token: resolved}, resolve_cache_path)
        return resolved, "pubchem"
    return None, "miss"


__all__ = [
    "load_name_resolve_cache",
    "name_resolve_cache_path",
    "resolve_common_name",
    "resolve_name_hybrid",
    "save_name_resolve_cache",
    "synonym_seed",
]

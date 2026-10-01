"""SQLite schema for the molecule search index.

The index is a single file built once from the HDF5 datasets and shipped
alongside the published data, so users never build it themselves. It is opened
read-only at query time.

Design notes that matter for portability:

* ``datasets.source_file`` stores a **basename**, never a resolved path — a
  published index has to stay valid wherever the user puts their HDF5 files.
* ``molecules.pattern_fp`` holds an RDKit pattern fingerprint as raw bytes.
  Substructure queries screen against these in one vectorised pass and only
  then verify survivors with RDKit (see :mod:`rpfr_visualizer.io.search.screening`).
* Canonical SMILES, InChIKeys and pattern fingerprints are all RDKit-version
  sensitive, so ``meta.rdkit_version`` is recorded and checked on open.
"""

from __future__ import annotations

# Bump when the schema or the meaning of any stored column changes.
# Mismatch at open time is a hard error with a rebuild instruction.
INDEX_VERSION = 4

# Bits in the stored pattern fingerprint.
FP_SIZE = 1024
FP_BYTES = FP_SIZE // 8
FP_KIND = "rdkit-pattern"

# Version of the topological key vocabulary. Bumping this invalidates
# key_bitmaps without touching anything else in the index.
KEY_KIND = "atom-bond-v2"

SCHEMA_SQL = f"""
PRAGMA user_version = {INDEX_VERSION};

CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS datasets (
    dataset        TEXT PRIMARY KEY,
    source_file    TEXT NOT NULL,
    n_molecules    INTEGER NOT NULL,
    temperatures   TEXT NOT NULL,
    schema_version TEXT
);

CREATE TABLE IF NOT EXISTS molecules (
    id               INTEGER PRIMARY KEY,
    dataset          TEXT NOT NULL REFERENCES datasets(dataset),
    key              TEXT NOT NULL,
    smiles           TEXT NOT NULL DEFAULT '',
    canonical_smiles TEXT NOT NULL DEFAULT '',
    formula          TEXT NOT NULL DEFAULT '',
    inchikey         TEXT NOT NULL DEFAULT '',
    inchikey14       TEXT NOT NULL DEFAULT '',
    n_atoms          INTEGER NOT NULL DEFAULT 0,
    n_heavy          INTEGER NOT NULL DEFAULT 0,
    element_counts   TEXT NOT NULL DEFAULT '{{}}',
    label            TEXT NOT NULL DEFAULT '',
    qc_pass          INTEGER,
    pattern_fp       BLOB,
    mol_bin          BLOB,
    UNIQUE(dataset, key)
);

CREATE INDEX IF NOT EXISTS idx_mol_canon   ON molecules(canonical_smiles);
CREATE INDEX IF NOT EXISTS idx_mol_formula ON molecules(formula);
CREATE INDEX IF NOT EXISTS idx_mol_inchi   ON molecules(inchikey);
CREATE INDEX IF NOT EXISTS idx_mol_inchi14 ON molecules(inchikey14);
CREATE INDEX IF NOT EXISTS idx_mol_key     ON molecules(key);
CREATE INDEX IF NOT EXISTS idx_mol_dataset ON molecules(dataset);
CREATE INDEX IF NOT EXISTS idx_mol_heavy   ON molecules(n_heavy);
CREATE INDEX IF NOT EXISTS idx_mol_natoms  ON molecules(n_atoms);

CREATE TABLE IF NOT EXISTS element_counts (
    mol     INTEGER NOT NULL REFERENCES molecules(id),
    element TEXT NOT NULL,
    count   INTEGER NOT NULL,
    PRIMARY KEY (mol, element)
) WITHOUT ROWID;

CREATE INDEX IF NOT EXISTS idx_elem ON element_counts(element, count);

CREATE TABLE IF NOT EXISTS names (
    id     INTEGER PRIMARY KEY,
    mol    INTEGER NOT NULL REFERENCES molecules(id),
    name   TEXT NOT NULL COLLATE NOCASE,
    source TEXT NOT NULL,
    UNIQUE(mol, name)
);

CREATE INDEX IF NOT EXISTS idx_names_name ON names(name COLLATE NOCASE);

CREATE TABLE IF NOT EXISTS fg_presets (
    id          INTEGER PRIMARY KEY,
    name        TEXT UNIQUE NOT NULL,
    description TEXT,
    smarts      TEXT NOT NULL,
    element     TEXT,
    is_primary  INTEGER NOT NULL DEFAULT 1
);

CREATE TABLE IF NOT EXISTS fg_membership (
    fg        INTEGER NOT NULL REFERENCES fg_presets(id),
    mol       INTEGER NOT NULL REFERENCES molecules(id),
    n_matches INTEGER NOT NULL DEFAULT 1,
    PRIMARY KEY (fg, mol)
) WITHOUT ROWID;

CREATE INDEX IF NOT EXISTS idx_fg_mol ON fg_membership(mol);

-- Topological screen (see rpfr_visualizer.io.search.keys). One row per distinct
-- atom or bond key, holding a bitmap over molecule ids: bit (id - 1) is set
-- when that molecule contains the key.
--
-- Bitmaps rather than a (key, mol) posting table: a query reads only the
-- handful of blobs its pattern compiles to, instead of scanning an index.
-- This also replaces the 113 MB pattern-fingerprint matrix, which had to be
-- loaded in full before the first substructure query could run.
--
-- The bitmaps are zlib-compressed because the vocabulary is very skewed —
-- across the published index the median key appears in 14 molecules and 74%
-- appear in fewer than 100, so most bitmaps are almost all zeros. Raw they
-- cost 238 MB; compressed, 3.4 MB. They are decompressed on first use and
-- cached for the process lifetime.
--
-- n_set is the popcount, stored so the query planner can intersect the most
-- selective feature first without decompressing anything.
CREATE TABLE IF NOT EXISTS key_bitmaps (
    token TEXT PRIMARY KEY,
    kind  TEXT NOT NULL,
    n_mol INTEGER NOT NULL,
    n_set INTEGER NOT NULL,
    bits  BLOB NOT NULL
) WITHOUT ROWID;
"""

# FTS5 is not guaranteed on every Python build, so it is created separately and
# its availability recorded in meta. Queries fall back to LIKE on the (small)
# names table when it is absent.
FTS_SQL = """
CREATE VIRTUAL TABLE IF NOT EXISTS names_fts USING fts5(
    name,
    content='names',
    content_rowid='id',
    tokenize="unicode61 remove_diacritics 2"
);
"""

FTS_REBUILD_SQL = "INSERT INTO names_fts(names_fts) VALUES('rebuild');"

# meta keys written by the builder and read back on open.
META_INDEX_VERSION = "index_version"
META_BUILT_AT = "built_at"
META_RDKIT_VERSION = "rdkit_version"
META_FP_KIND = "fp_kind"
META_FP_SIZE = "fp_size"
META_HAS_FTS = "has_fts"
META_FG_CATALOG_HASH = "fg_catalog_hash"
META_SOURCE_SIGNATURE = "source_signature"
META_KEY_KIND = "key_kind"
META_KEY_BITS = "key_bitmap_bits"
META_HAS_MOL_BIN = "has_mol_bin"

__all__ = [
    "FP_BYTES",
    "FP_KIND",
    "FP_SIZE",
    "FTS_REBUILD_SQL",
    "FTS_SQL",
    "INDEX_VERSION",
    "KEY_KIND",
    "META_BUILT_AT",
    "META_FG_CATALOG_HASH",
    "META_FP_KIND",
    "META_FP_SIZE",
    "META_HAS_FTS",
    "META_HAS_MOL_BIN",
    "META_INDEX_VERSION",
    "META_KEY_BITS",
    "META_KEY_KIND",
    "META_RDKIT_VERSION",
    "META_SOURCE_SIGNATURE",
    "SCHEMA_SQL",
]

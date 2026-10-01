# Molecule search

Searching the QM9s-RPFR datasets — by name, formula, SMILES, InChIKey, element
composition, functional group, or arbitrary SMARTS substructure.

## Quick start

The published data ships with a prebuilt index, so nothing needs building:

```python
from rpfr_visualizer.io.molecule_index import MoleculeIndex

index = MoleculeIndex.open("data/cache/qm9s_rpfr_index.sqlite")

index.search("C6H6")                        # formula
index.search("CCO")                         # SMILES (exact molecule, any writing)
index.search("ethanol")                     # name
index.search("smarts:[CX3](=O)[OX2H1]")     # substructure
index.search("fg:carboxylic_acid")          # curated functional group
index.search("fg:alcohol&amine")            # …carrying both groups
index.search("fg:alcohol, C:5-9")           # …with five to nine carbons
index.search("heavy<=12")                   # composition alone
index.filter_elements(must_have=["N"], must_not_have=["O"])
```

Every hit carries a `match_type` (`formula`, `smarts`, `name:local`, …) so a UI
can show *why* a molecule was returned.

## Why SQLite

The previous index held every molecule as a Python object in a JSON cache and
answered substructure queries by re-parsing every SMILES with RDKit. That is
fine for a 1% subsample and unusable at the full 881,757 molecules.

The index is one SQLite file, built once per data release and published
alongside the HDF5 data. It is opened read-only, so a query can never mutate it.

## Schema

| Table | Holds |
|---|---|
| `meta` | `index_version`, `rdkit_version`, fingerprint config, FTS availability, source signature |
| `datasets` | one row per HDF5 file: **basename**, molecule count, temperature grid |
| `molecules` | identifiers (key, SMILES, canonical SMILES, formula, InChIKey + skeleton), atom counts, stored QC verdict, pattern fingerprint, serialised RDKit mol |
| `element_counts` | `(molecule, element, count)` — powers "contains N", "3 ≤ C ≤ 6" |
| `key_bitmaps` | one compressed bitmap per topological key — the substructure screen |
| `names` + `names_fts` | shipped synonyms, with FTS5 prefix search |
| `fg_presets` / `fg_membership` | precomputed membership for the curated functional groups |

Two decisions worth knowing:

- **`datasets.source_file` is a basename, never a resolved path.** A published
  index has to keep working wherever the user puts their HDF5 files. A
  signature mismatch warns; it does not fail.
- **`index_version` mismatch is a hard error**, with the rebuild command in the
  message. Silently querying an index whose columns mean something else is
  worse than refusing.

## How substructure search works

A SMARTS query is answered by screening on **topological keys** and then
verifying every survivor with RDKit.

The screen used to be a 1024-bit RDKit *pattern fingerprint*. That turned out to
be information-limited rather than bit-limited: `Chem.PatternFingerprint` hashes
small subgraphs, so a query like `[CX3](=O)[OX2H1]` sets bits standing for
generic C–O and C=O paths that a third of the database genuinely contains.
34.6% of all molecules survived to find 0.6% true hits, and raising `fpSize`
from 1024 to 8192 moved that by 0.0 points.

What the fingerprint throws away is exactly what a functional-group query is
made of — how many neighbours an atom has, and how many hydrogens.
[keys.py](../src/rpfr_visualizer/io/search/keys.py) stores those instead. Each atom
contributes a key

```
(element, isotope, aromatic, total_degree, n_hydrogens)
```

and each bond contributes a key pairing its two atom keys with a bond order. A
SMARTS pattern is compiled to the keys any real match must contain, and a
molecule survives only if it holds a compatible key for each. On the same index
that takes the carboxylic-acid candidate set from 34.6% to **1.9%** of the
database.

Storage is one zlib-compressed bitmap per key in `key_bitmaps`. The vocabulary
is very skewed — the median key appears in 14 molecules — so raw bitmaps would
cost 238 MB where compressed ones cost **3.6 MB**. A query reads only the handful
of bitmaps its pattern compiles to, which also removes the ~1.9 s load of the
113 MB fingerprint matrix that every first substructure query used to pay.

The index also stores `Mol.ToBinary()` per molecule, because *rebuilding*
molecules rather than matching them was 80% of a query: `Chem.Mol(blob)` runs at
219k/s against 50k/s for `MolFromSmiles`, and `HasSubstructMatch` manages
630k/s once a molecule exists.

### Why this is correct, not just fast

**The screen is a filter, never an answer.** It returns a superset and every
survivor is verified, so results are identical to brute force. Three things
guarantee that:

- **Compatibility is decided by RDKit, not reimplemented.** Each stored key is
  rebuilt as a probe atom carrying those five properties, and the query's own
  `QueryAtom.Match` is asked whether it accepts it. The screen inherits SMARTS
  semantics rather than approximating them.
- **Primitives a lone probe cannot judge fall back.** Recursive environments
  (`[$(...)]`), ring membership and ring size all depend on the rest of the
  molecule. These are detected via `DescribeQuery` and the query reverts to the
  pattern fingerprint rather than silently dropping hits.
- **Atom keys are stored as well as bond keys**, so a molecule with no bonds at
  all still has keys and can still match a single-atom query. A bond-only index
  would have excluded methane.

Two test modules are the reason to trust this, and both compare against
`screen=False` on a real index rather than asserting on internals — every way
this screen has gone wrong has been silent, returning a plausible smaller answer
instead of raising. `tests/unit/io/test_keys.py` covers the compiler (27 cases);
`tests/unit/io/test_screen_equivalence.py` requires identical result *sets* for
24 patterns spanning multi-bond, single-atom, bondless, isotopic, aromatic,
negation, ring and recursive constructs.

Pass `screen=False` to any `by_smarts` call to bypass the screen when debugging.

### What actually costs time

Verification still dominates, but far less of it now happens. Prefer a curated
preset when one fits — `fg_membership` skips both stages — and pass `limit=` for
broad patterns.

Speedups on unlimited queries against the full index: 68× `[NX2]=[NX2]`, 66×
`[2H]`, 35× carboxylic acid, 32× benzoic acid, 15× `[S]`, 13× `[nH]`. Broad
queries gain 1–3×, and a pattern the screen cannot narrow at all (`[!C]`) is
marginally slower.

## Formulas

A formula is matched on composition, not on how it is written. The index
stores Hill order (carbon, hydrogen, then alphabetical), but nothing obliges
you to type it:

```python
index.search("C6H14O")       # Hill order
index.search("C6OH14")       # the same molecules
index.search("OH14C6")       # still the same
index.search("C6 H14 O")     # spaces are ignored
index.search("C₆H₁₄O")       # unicode subscripts, as pasted from a paper
index.search("CH3CH2OH")     # condensed structural; aggregates to C2H6O
```

Two deliberate exceptions:

- **Case is significant.** `CO` is carbon monoxide and `Co` is cobalt, and
  lowercasing would also collide with aromatic SMILES such as `c1ccccc1`. A
  lowercase `c6h14o` is not read as a formula.
- **A period is left alone**, because `.` separates components in SMILES.

Unicode subscripts were worse than unmatched before: the element pattern reads
ASCII digits only, so `C₆H₁₄O` parsed as `CHO` — every count silently dropped,
and a real formula that the index would look up and report as a match.

## Narrowing a query

Any query can be narrowed by composition or by quality. Unlike the
comma-separated tokens, which are OR'd together, these are **AND'd onto
everything else**, so `fg:alcohol, C:5-9` means alcohols that *also* have five
to nine carbons rather than the union of the two. A query of nothing but
constraints is a search in its own right.

### Size

| Form | Example | Means |
|---|---|---|
| `El:n` | `C:7`, `Br:1` | exactly n of that element |
| `El:a-b` | `C:5-9` | between a and b inclusive |
| `El` + comparison | `nC>3`, `C<=12`, `C=7` | `>`, `>=`, `<`, `<=`, `=` |
| `heavy` | `heavy<=12`, `heavy:5-9` | non-hydrogen atom count |
| `atoms` | `atoms=20` | total atom count |

The `n` prefix (`nC>3`) is optional and reads more naturally for comparisons.

### Functional groups

| Form | Means |
|---|---|
| `fg:alcohol` | carries that group |
| `fg:alcohol&amine` | carries **both** |
| `fg:alcohol, fg:amine` | carries **either** — commas are the usual OR |

`&` is the conjunction because a comma already means "either". It appears in
neither SMILES nor SMARTS, so it cannot be mistaken for part of a pattern.

Against the published QM9s tier the three forms agree exactly, which is the
check worth remembering: 376 alcohols + 374 amines − 85 that are both = 665
that are either.

### Quality

| Form | Means |
|---|---|
| `qc:pass` | the stored QC verdict was a pass (`qc:passed`, `qc:ok` also work) |
| `qc:fail` | the stored QC verdict was a failure (`qc:failed` also works) |

```python
index.search("fg:alcohol, qc:pass")     # alcohols that also passed QC
index.search("qc:fail")                 # everything that failed
```

**Molecules with no recorded verdict match neither.** A dataset can store no QC
flags at all, and "not known to have failed" is a weaker claim than "passed" —
folding the two together is how a quality filter ends up reassuring somebody
about data it knows nothing about. The Dataset page reports the three counts
separately for the same reason.

In the explorer this is the *Only molecules that passed QC* checkbox, which
appends `qc:pass` to whatever is in the search box.

The constraint is pushed into SQL rather than applied to the returned page, so a
limited query reports the first n molecules satisfying *both* halves instead of
filtering an already-truncated list. For SMARTS it is applied to the candidates
before RDKit verification, which is the expensive step.

Two details worth knowing. The grammar runs ahead of every other classifier, so
names, formulas, SMILES, SMARTS and InChIKeys must never parse as a size
constraint — the tests weight those negative cases heavily. And a `smarts:` body
legitimately contains commas (`[NX3;H2,H1,H0]`), so trailing fragments are
peeled off only while they parse as size tokens, which no SMARTS fragment does.

## Measured performance

Full index, 881,757 molecules, warm cache, Apple Silicon:

| Query | Time | Hits |
|---|---|---|
| `count()` | 11 ms | 881,757 |
| `by_key("1000")` | 3.0 ms | 2 |
| `by_name("ethanol")` | 3.2 ms | 4 |
| `by_elements(must_have=["N"], must_not_have=["O"], limit=500)` | 3.8 ms | 500 |
| `by_formula("C6H6")` | 21 ms | 278 |
| `by_functional_group("carboxylic_acid")` | 34 ms | 5,165 |
| `search("heavy<=12", limit=200)` | 55 ms | 200 |
| `functional_group_counts()` — runs on panel load | 86 ms | 18 groups |
| `by_functional_group("ether", limit=200)` | 102 ms | 200 |
| `by_smarts("[CX3](=O)[OX2H1]", limit=500)` | 131 ms | 500 |
| `by_smarts("[CX3](=O)[OX2H1]")` no limit | 219 ms | 5,165 |
| `by_smarts("c1ccccc1", limit=500)` | 451 ms | 500 |
| `by_smarts("[NX2]=C=O", limit=500)` | 518 ms | 500 |
| `by_functional_group("ether")` no limit | 1,986 ms | 257,186 |

Reading this table: identifier and composition lookups are effectively instant,
and every *bounded* query — which is all the UI ever issues — is comfortably
sub-second. The first query of a kind pays a one-off page-cache cost of roughly
3–6× these figures; there is no longer a multi-second matrix load.

The one slow row is an unbounded functional-group query, and the cost is
materialising 257,186 dataclasses rather than finding them. Pass `limit=`.

## Building an index

```bash
rpfr-visualizer index build --raw data/raw --out data/cache/qm9s_rpfr_index.sqlite
```

```bash
rpfr-visualizer index build --drive "$RPFR_VISUALIZER_DRIVE" --out qm9s_rpfr_index.sqlite
```

The full seven-dataset build (881,757 molecules) produces a 589 MB index. RDKit
descriptor derivation (~0.3 ms/molecule) dominates, plus one pass to enumerate
each HDF5 file's molecule groups. The index also carries the topological key
bitmaps and one serialised RDKit mol per molecule, which is most of the growth
over the 456 MB of the fingerprint-only layout.

Inspect or check an index:

```bash
rpfr-visualizer index info data/cache/qm9s_rpfr_index.sqlite
```

```bash
rpfr-visualizer index verify data/cache/qm9s_rpfr_index.sqlite --raw data/raw
```

`verify` runs an integrity check, confirms the version, warns when the running
RDKit differs from the one that built the index, and — given `--raw`/`--drive`
— cross-checks molecule counts per dataset.

## How the index is found

`open_index()` looks in order:

1. an explicit path argument,
2. `$RPFR_VISUALIZER_INDEX`,
3. `<data_dir>/cache/qm9s_rpfr_index.sqlite`.

If none exists it raises `MissingIndexError` with both the download hint and
the build command. During development, `open_or_build_index(catalog)` builds
one on the spot instead.

## Names and PubChem

The offline synonym table shipped with the package is baked into the index at
build time, so common-name lookups work with no network.

Name resolution falls back through: shipped synonyms → local resolve cache →
PubChem. **The PubChem tier is opt-in** (`allow_pubchem`), and
`MoleculeIndex.open()` defaults it to *off* so a panel never blocks on a network
request. It also never writes to the index — resolved names go to a separate
JSON cache.

## Backends

`MoleculeIndex` is a facade over two interchangeable backends:

- `SqliteSearchBackend` — the prebuilt index. Use this for the published data.
- `MemorySearchBackend` — scans a plain entry list. Used by tests and by
  `MoleculeIndex.build_from_catalog()` for small catalogs.

They implement the same interface and are held to identical results by
`tests/unit/io/test_search_equivalence.py`. Accessing `MoleculeIndex.entries`
materializes every record and warns above 50k — prefer `iter_entries()`,
`get()`, or `get_many()`.

## RDKit version sensitivity

Canonical SMILES, InChIKeys, topological keys, pattern fingerprints and the
serialised mols all depend on the RDKit version. The building version is
recorded in `meta.rdkit_version`, and `index verify` warns on a mismatch. If
exact-structure lookups start missing after an RDKit upgrade, rebuild the index.

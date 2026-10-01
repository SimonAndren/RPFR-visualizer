# The explorer

A local web application for looking at per-site isotope fingerprints. It runs on
your own machine; nothing is uploaded anywhere.

```bash
uv sync --extra gui
uv run rpfr-visualizer gui
```

That starts a small server and opens your browser. Press Ctrl-C to stop it.

## Trying it with no data

```bash
uv run rpfr-visualizer gui --demo
```

Generates a five-molecule dataset and a search index under
`~/.config/rpfr-visualizer/demo`, then opens the app on it. Nothing is downloaded and
the build takes about a second; a second run reuses what is there.

**The demo's numbers are invented.** It is a valid schema 2.0 file on the real
temperature grid, and its five molecules are picked to cover what the interface
has to render — a clean molecule, a QC failure, a translation/rotation warning,
a site-label mismatch — but no β in it was computed from anything. It registers
under the dataset name `demo` so the picker says so while you are looking at it.
Use it to check that the software runs and to learn the controls; never to read
a number off. `--demo` refuses to combine with `--data` or `--drive`.

## What you are looking at

For every atom in a molecule the database stores **β**, the reduced isotopic
partition function ratio, across a grid of temperatures. A site with a large β
concentrates the heavy isotope. The explorer draws those values onto the
molecule: **colour is the element** (canonical CPK — carbon black, nitrogen
blue, oxygen red) and **opacity is the magnitude of the RPFR for that
element**, scaled to the dataset's 5th–95th percentile for that element at that
temperature. An opaque oxygen and an opaque carbon are each strong for their
element, not equal. The caption under the figures gives each element's range.

That describes the **3D view**, which also labels each focus atom with its
value as well as its name: opacity ranks sites against one another but never
states one, and in 3D there is room to answer both questions at once.

The **2D structure carries no values at all** — it is a structural diagram.
CoordGen lays the molecule out, carbons stay bare vertices, heteroatoms keep
their symbols, stereocentres come out as wedges, and each atom's index is drawn
*beside* it as an annotation. There is no shading behind the atoms: a tinted
disc at every vertex is not something a chemist draws, and on a one-carbon
molecule — where RDKit writes `CH4` rather than a skeleton — it was a grey
circle around the letters. A lone heavy atom gets no number either, since there
is nothing to tell apart. The only disc drawn here marks an atom you picked.

The numbers are database atom indices, matching the site table, not IUPAC
locants.

Three quantities are offered:

| Shown as | Meaning |
|---|---|
| `β — partition function ratio` | read from the database; needs no reference |
| `α — fractionation factor` | β ÷ a reference β; 1 means no fractionation |
| `1000 ln α (‰)` | α in per mil, the isotope-geochemistry convention |

ε = 1000 (α − 1) is not offered. For effects of the size seen here it is
within a fraction of a per mil of 1000 ln α, and two near-identical per-mil
numbers in one dropdown only invited the question of which to use. It remains
available from Python as `AnalysisKind.EPSILON`.

`clumping` and `mass_law` are not in that list, because the per-atom colour
normalises against cached dataset percentiles and those exist for β and α
only. They have their own place — the **Derived** panel — where an absolute
scale is fine:

| Shown as | What it is |
|---|---|
| β against temperature | the thirteen stored values as markers, plus the fit of ln β against 1/T that off-grid temperatures are read from. Log scale: β spans many orders of magnitude between 20 K and 420 K. The caption names which fit drew the line. |
| mass-law exponent λ | the triple-oxygen exponent. Stored for **oxygen only** — every other atom is "not applicable" (NaN in schema 2.2, 0 before it) rather than "λ = 0" — and only at the stored temperatures. Drawn as markers with no line, because λ is not fitted between them. |
| clumping Δ | Δᵢⱼ = 1000 ln(βᵢⱼ / βᵢβⱼ) in per mil for every pair of atoms: how far the doubly substituted isotopologue's β exceeds the product of the two singly substituted ones. Shown as a matrix on a diverging scale centred on zero, beside a table of the strongest pairs. The diagonal is not a pair and is left off the scale. Click a cell or a row to highlight that pair. |

**Clumping and λ exist only at the stored temperatures.** Unlike β they are
not fitted between them, so a temperature that is not stored makes them
unavailable rather than approximate — the panel says so and offers the nearest
stored temperatures as buttons.

### When the structure does not match the geometry

About 0.3% of schema 2.2 molecules have a stored SMILES that no longer
describes the geometry their values were computed on. They are flagged
(`geometry_match = False`), never dropped, and they keep the SMILES — so a
skeletal structure can still be drawn from it, and something will.

It is drawn, with a warning on the pane. The picture comes from the SMILES and
the numbers come from the coordinates, and for these molecules those are two
different arrangements of the same atoms. The **3D view is built from the
coordinates themselves**, so it is unaffected and carries the values as usual.

Schema 2.0 and 2.1 never computed the flag; absent reads as `True` there
rather than flagging every older molecule.

## Temperature is never rounded silently

A stored temperature is read directly. Anything else is **fitted** in ln β
against 1/T, by default with a local quadratic through the four nearest stored
points. The line under the controls says which case you are in, and names the
fit. Typing `298` is a different temperature from `298.15 K`.

### Choosing the fit

An off-grid temperature is fitted in ln β against 1/T. The picker sits next to
the temperature and appears only when it has something to do. Measured on
schema 2.2 sites by hiding each stored point and predicting it, in per mil of
1000 ln β:

| | typical error | worst 5% | below 60 K |
|---|---|---|---|
| `local2`, local quadratic (default) | 0.3 ‰ | 9 ‰ | physical |
| `pchip`, monotone spline | 0.1 ‰ | 1.7 ‰ | physical |
| `poly5` | 1.0 ‰ | 171 ‰ | overshoots — β rises with temperature |
| `poly4` | 3.6 ‰ | 75 ‰ | physical |
| `poly3` | 5.7 ‰ | 49 ‰ | physical |
| `poly2` | 7.7 ‰ | 52 ‰ | physical |

The default is a second-degree polynomial in 1/T fitted to the four stored
points around the temperature — two below, two above, shifted inward at the
ends of the grid. A quadratic in 1/T carries the 1/T² term that dominates ln β
as temperature rises, and a local window follows curvature one quadratic
across the whole grid cannot: `poly2` is about thirty times worse.

Two below and two above, rather than the four nearest by distance: nearest
changes the window partway through four of the intervals (206.6, 239.1, 326.6
and 359.1 K) and the curve takes a visible step there. Bracketing changes the
window only at stored temperatures, which are read rather than fitted, so the
curve never steps — at no measurable cost in accuracy.

Degree 5 was the default until 0.5.0. It overshoots below 60 K in every site
measured. None of the methods is *accurate* between 20 K and 60 K, where
nothing is stored; below 60 K, prefer the stored points.

The setting applies to everything that reports β at one temperature — the
curve, the atom colours, and the CSV, which records it in a `beta_method`
column. From Python it is the `method=` argument on `interpolate_beta`,
`beta_curve`, `beta_at_temperature` and `SiteRecord.beta_at`, and
`AnalysisRequest.beta_method` for a whole analysis.

### Downloading several temperatures at once

The Download page takes a comma-separated list — `250, 273.15, 298.15`. Leave
it blank and you get the one temperature on screen, which is what the export
always did.

One row per site **per temperature**, with a site's temperatures contiguous, so
a calibration fit reads straight off the file. Each row carries its own
`temperature_K`, `on_grid` and `beta_method`, so a file mixing stored and
fitted temperatures says which each row was. The count above the button
already accounts for the multiplication, and a multi-temperature file is named
`..._3T.csv` rather than for any one of them.

Capped at 50 temperatures: the cost is one analysis per molecule per
temperature, so a thousand-molecule search across fifty of them is fifty
thousand analyses.

## Finding molecules

One field takes all of it: a name, a formula, a SMILES, a molecule ID, or an
explicit `fg:alcohol` / `smarts:[CX3](=O)[OX2H1]`. Each result says *why* it
matched.

Any of those can be narrowed by size — `C:5-9`, `nC>3`, `heavy<=12`, `atoms=20`
— and a size token is **AND'd** onto the rest of the query rather than OR'd with
it. So `fg:alcohol, C:5-9` reads as "alcohols with five to nine carbons", and a
query of nothing but size tokens is a composition search on its own. The full
grammar is in [docs/search.md](search.md#size-constraints).

`Browse by group` is a checklist, not a dropdown: tick as many as you like. The
**any / all** toggle beside it decides what more than one selection means —
*any* finds molecules carrying at least one of them, *all* finds only those
carrying every one. Either way the picker writes the equivalent query into the
search box and runs it, so the syntax is visible rather than hidden, and one
grammar decides what a search means.

Groups with no molecules in the current dataset stay in the list, greyed —
absence is information.

**The results list is the ranking pool.** Pick a group, switch to *Ranked sites*,
and you are ranking sites across every molecule the search returned. The rule
worth remembering:

> The group chooses the **molecules**. Element chooses the **sites**.

So "carbon sites in alcohol-containing molecules" is two clicks.

### Two honest limits

**Results are capped and unordered.** The index has no cursor and no relevance
ranking, so results come back in the order they sit in the file. The app shows
the first 200 and says so; it does not offer a "more" button that would page
through an arbitrary order. Narrow the search instead.

**A molecule without a stored SMILES can only be found by ID.** Every molecule
in the published QM9s tier has one; the *Dataset* page shows the coverage, so a
zero-hit structural search on other data reads as "unanswerable here" rather
than "none here".

## The Download page

Takes the molecule on screen, or the current search results, and writes a CSV
at the temperature and quantity already chosen on Explore. One row is one atom
site: dataset, molecule key, SMILES, formula, atom index, element, β, the
displayed quantity, and the reference β when α or 1000 ln α is selected.

Explore lists the first 200 matches; a results download includes every match,
and the page gives the true count before you click. Several temperatures can be
exported at once (`250, 273.15, 298.15`); a `beta_method` column says which
rows were fitted.

CSV is the only format. Excel needs another dependency and dies at 1,048,576
rows; a full site table at one temperature is millions of rows.

## The Dataset page

Molecule counts, structure coverage, the QC split, the temperature grid, the
per-element percentile ranges the colour scale uses, and the functional-group
census. It also reports which RDKit version built the index versus which one you
are running: canonical SMILES, InChIKeys and the substructure screen all depend
on it, so a mismatch can make structure lookups miss molecules that are there.

## Getting data

```bash
uv run rpfr-visualizer data fetch --tier qm9s-1pct --dest data
```

~92 MB: QM9s at 1% (1,298 molecules, schema 2.2) and the index built over exactly that
file. Checksummed on arrival; a file that does not match is deleted rather than
left somewhere it might be trusted.

**This release publishes QM9s only.** The explorer opens any QM9s-RPFR dataset
you point it at — `--data` takes a file, a directory or a drive root, and
`rpfr-visualizer index build` will index whatever it finds — but `data fetch` offers
QM9s.

**Tiers are fetched whole.** The index always comes with the data it describes,
because a SQLite index is self-contained: paired with the wrong files it keeps
answering searches happily while every hit fails to open. `data fetch` will not
create that state, and `factory.index_data_mismatch()` detects it if some other
route does.

`rpfr-visualizer data list` shows the tiers this version knows about, and
`data fetch --verify-only` re-checks what is already on disk without
downloading anything.

## Pointing it at data

`--data` (and the setup screen's path box) takes **any** of these, because
someone with data on disk should not have to know which loader wants which
layout:

| You have | Give it |
|---|---|
| one HDF5 file | `--data path/to/qm9s_tr_project.h5` |
| the repo layout | `--data data` (finds `data/raw/*.h5`) |
| the external drive | `--data "$RPFR_VISUALIZER_DRIVE"` |
| a folder of `.h5` files | `--data /some/folder` |

The startup banner says what it opened and which index it matched, so a wrong
guess is visible immediately rather than three clicks later.

**The index is chosen by coverage, not by filename.** If several `.sqlite` files
are present, each is opened and the one covering the most molecules of your
dataset — while still describing the same molecules — wins. That matters: a 1%
subsample index opened against the full data would search 1% of it and look
perfectly healthy, so an index covering less than half the file is rejected with
an explanation rather than used.

## Where it looks

In order: command-line flags, `$RPFR_VISUALIZER_INDEX`, your saved settings, then
auto-discovery — which walks up from the working directory looking for a
`data/raw` or `data/cache`, checks `$RPFR_VISUALIZER_DATA`, and offers the external
drive only if it is already mounted.

If two index files are present (say a small subsample index and the full one), the
app **lists both with their sizes and molecule counts and asks**, rather than
guessing. Your answer is saved to `~/.config/rpfr-visualizer/gui.json`; delete that file
or pass `--reset-config` to start over.

An index is never built implicitly — over the full datasets that takes tens of
minutes. If none is found the app says so and gives you the command.

### Options

```
rpfr-visualizer gui [--data DIR] [--index PATH] [--drive ROOT] [--dataset NAME]
              [--port 8765] [--no-browser] [--reset-config] [-v]
```

`--host` only accepts loopback. The explorer has no authentication and reads
local files, so it is not something to expose to a network.

## Without the external drive

The full-dataset HDF5 files live on external storage, but the search index is a
self-contained SQLite file. That has a consequence worth knowing: with the drive
unplugged, a full index **keeps answering searches** across all 881,757 molecules
while none of the results can actually be opened. The app detects that mismatch on
startup and tells you which index belongs to which data, rather than letting you
click a result and get an error.

The 1% subsamples in `data/raw/` need no drive at all, and that is what the app
opens by default.

## Using it from Python instead

Everything the GUI does is available without it — the browser layer holds no
logic of its own:

```python
from rpfr_visualizer.viz.explorer import controller_for_data_dir

explorer = controller_for_data_dir("data")
explorer.search_functional_group(preset_name="alcohol", limit=200)
explorer.load_hits_into_pool()
explorer.state.element = "C"
for site in explorer.rank():
    print(site.key, site.element, site.value, site.local_smiles[0])
```

The notebook that ships with the package is
[explorer_tutorial.ipynb](explorer_tutorial.ipynb): read it, then run the start
cell to open this explorer in Chrome.

## Working on it

`scripts/dev_gui.py` starts the app the way you want it while you are changing
it: no browser, every request logged, and the dev-only client-error endpoint on.

```bash
uv run python scripts/dev_gui.py --fixture --port 8765 \
    --log /tmp/gui.log --pid-file /tmp/gui.pid
until curl -sf http://127.0.0.1:8765/healthz >/dev/null; do sleep 0.2; done
...
kill "$(cat /tmp/gui.pid)"
```

`--fixture` builds a five-molecule dataset and its index under
`data/cache/dev-gui` once, so it starts in about a second and needs neither
`data/raw` nor the drive. It also redirects `RPFR_VISUALIZER_GUI_CONFIG` to a scratch
file, so clicking "Use this" during development cannot rewrite your real
remembered settings.

**A front-end edit needs no restart.** The three static assets are re-read per
request and served `no-store`, so reloading the browser is enough. Only a Python
change needs the server restarted.

### When something breaks in the browser

Started with `--dev`, an uncaught error in `app.js` is posted to the server,
printed with its file and line, and kept readable over HTTP:

```bash
curl -s localhost:8765/api/_client_error | python -m json.tool
```

Both `/api/_client_error` routes answer 404 without `--dev`. They store text
supplied by the browser, so they exist only when you asked for them.

### Tests

Cheapest first — most cycles never start a browser:

```bash
uv run pytest tests/unit/gui --no-cov -q               # ~1 s
uv run pytest tests/e2e --no-cov -q -x --lf            # ~4 s, real browser
uv run pytest tests/e2e -k gallery --gallery --no-cov  # screenshots
```

`tests/unit/gui/test_static_contract.py` checks what is statically knowable in
about ten milliseconds: every id `app.js` reaches for exists in `index.html`,
every endpoint it calls is a registered route, and `innerHTML` is still used
exactly once. Those catch a renamed id or route without a browser at all.

`tests/e2e/` drives the real UI. It needs the `e2e` extra
(`uv sync --extra e2e`), uses the Chrome you already have rather than
downloading one, and removes itself from collection when the extra is absent.
Add `--headed --slowmo 400` to watch it work.

`--gallery` writes screenshots of eight states to `tests/e2e/_screens/`. There
are deliberately no pixel-diff baselines: the 2D pane is RDKit's SVG and the 3D
pane is WebGL, so both move with the RDKit release, the platform fonts and the
GPU. Layout is asserted geometrically instead, where the oracle is exact.

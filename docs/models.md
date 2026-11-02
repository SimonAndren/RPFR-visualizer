# Empirical models

The explorer can show β predicted by a model trained on the dataset, in place of
the DFT values stored in the file. Pick the source from **β from** in the
header. That control appears once at least one model has been trained.

Every model predicts `1000 ln β` (in ‰) for each site at each of the 15 stored
temperatures. The models are separate for each element (H, C, N, O).

## The protocol

All models are trained against **one shared split**, made by molecule and not
by site, so a molecule never contributes sites to both training and test:

| Split | Molecules | Used for |
|-------|-----------|----------|
| train | 100,000 | fitting |
| validation | the rest (about 4,000 of QM9s) | choosing hyperparameters |
| test | 24,000 | the reported score, used once |

The test set is cut first. Changing `--n-train` therefore never moves a molecule
into or out of the test set. Only molecules that pass the pipeline QC and have
`bond_order` and `local_smiles` are split. The others are listed as excluded,
with the reason for each.

After tuning, the model is **not** refitted on train + validation. The saved
model is exactly the one the test score describes.

The explorer shows whether the open molecule is in the **training set**, the
**validation set** or the **test set**. A prediction for a training molecule
says little about how well the model generalises. A prediction for a test
molecule does.

## Training

The split needs the full file, `qm9s_rpfr_v2.2.h5`, which has enough molecules
for 100k + 24k:

```bash
rpfr-visualizer models split --data /path/to/qm9s_rpfr_v2.2.h5 --out data/models/split.json
rpfr-visualizer models train bond_additivity cluster_mean \
    --data /path/to/qm9s_rpfr_v2.2.h5 --split data/models/split.json
rpfr-visualizer models list --dir data/models
```

`train` reads the file once and trains every kind you name. It then prints each
model's test scores and writes `data/models/<kind>.npz` next to `split.json`.
The explorer looks for models in `<data dir>/models`.

To try the workflow on the 1% subsample, scale the split down:

```bash
rpfr-visualizer models split --data data/raw/qm9s_v2.2_1pt.h5 --n-train 1000 --n-test 240
```

## Models

### Bond additivity (`bond_additivity`)

A site's `1000 ln β` is an element baseline plus one increment for each bond
path leaving the site, fitted by ridge regression:

    1000 ln β(site, T) = b_e(T) + Σ_p n_p · w_p(T)

The paths come from `bond_order` (`C-H`, `C-C=O`, `C-N-C-H`, ...). Depth 1
covers only the bonds the atom itself makes, which is classic bond additivity.
Depths 2 and 3 add next-nearest corrections. Validation picks the depth (1–3)
and the ridge strength for each element.

### Cluster mean (`cluster_mean`)

A site gets the mean value of the training sites with the same local
environment (`local_smiles`, radii R1–R5). It uses the widest radius whose
cluster has at least `min_count` training sites. If no radius qualifies, it
falls back to narrower radii and finally to the element mean. Validation picks
`R_max` and `min_count` for each element. This is the nearest-neighbour
baseline, against which any learned model should be judged.

### Planned

MPNN and OrbNet-Equi models will use the same split and file format, behind the
same switch. They are not implemented yet.

## Scores

Scores are reported for each element, because the elements live on different
scales: an H-site error of 20 ‰ is ordinary, and a C-site error of 20 ‰ is not.
For each element there is a mean absolute error and a root-mean-square error at
every temperature, and MAE, RMSE and R² at 298.15 K. These are stored in the
model file and shown by `models list`.

For reference, the 1% subsample split (1,000 / 54 / 240 molecules) gives these
test MAEs at 298 K, in ‰:

| | C | H | N | O |
|---|---|---|---|---|
| bond additivity | 2.56 | 27.0 | 2.46 | 2.69 |
| cluster mean | 2.12 | 21.6 | 2.58 | 1.96 |

These numbers are from a small training set. Train on the full file before
quoting any of them.

## What a model does not provide

- **Clumping and mass-law.** Models predict β only. Under a model, the Derived
  panel shows the β curve and says the other two are not predicted. It does not
  show the dataset's values next to predicted β.
- **Molecules without descriptors.** A molecule that lacks `bond_order` or
  `local_smiles` (a handful, where RDKit failed) gets no prediction. Its sites
  show as missing values rather than as a guess.
- **Another temperature grid.** A model trained on the 15-point grid refuses a
  13-point schema 2.0 file rather than attaching its columns to the wrong
  temperatures.

The CSV download has a `beta_source` column (`dataset` or
`model:<name>`), so a model export can always be told apart from a dataset
export.

## File format

A model is a single `.npz` file: numeric arrays plus a JSON header (`__meta__`).
The header records the kind, the temperature grid, the split fingerprint, the
chosen hyperparameters, and the validation and test scores. Strings are packed
into byte arrays, so a model file loads with `allow_pickle=False` and cannot run
code.

To add a model kind, subclass `EmpiricalModel` (or `SiteTableModel` if it only
needs the per-site descriptors) and decorate it with `@register`. See
`rpfr_visualizer/empirical/`.

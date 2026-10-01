# Data files

The HDF5 files and the search index are not in this repository.

Fetch the published 1% tier:

    rpfr-visualizer data fetch --tier qm9s-1pct --dest data

That gives:

    data/raw/qm9s_v2.2_1pt.h5            1% of QM9s v2.2, 1,298 molecules
    data/cache/qm9s_rpfr_index.sqlite    the search index over that file

For the full dataset, point $RPFR_VISUALIZER_DRIVE (or --drive) at the folder
holding qm9s/qm9s_rpfr_v2.2.h5. To cut your own subsample from it:

    uv run python scripts/make_subsample.py

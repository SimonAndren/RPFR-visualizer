"""QM9s-RPFR: programmatic access to the QM9s-RPFR isotope effects database."""

# Read from the installed distribution rather than written here twice. The
# hardcoded string had drifted a release behind pyproject.toml, so an installed
# 0.4.0 introduced itself as 0.3.0 -- and a version a user reports in a bug is
# only useful if it is the real one.
try:  # pragma: no cover - trivial, and the fallback needs an uninstalled tree
    from importlib.metadata import PackageNotFoundError
    from importlib.metadata import version as _version

    __version__ = _version("rpfr-visualizer")
except PackageNotFoundError:  # running from a source tree with no install
    __version__ = "0.0.0+unknown"

from rpfr_visualizer.io.loader import Molecule, QM9sDataset, Site
from rpfr_visualizer.qc.models import DatasetQC, MoleculeQC

__all__ = [
    "DatasetQC",
    "Molecule",
    "MoleculeQC",
    "QM9sDataset",
    "Site",
    "__version__",
]

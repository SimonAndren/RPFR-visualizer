"""Frequency-related QC checks."""

from __future__ import annotations

from typing import Any

import h5py

from rpfr_visualizer.qc.fallback import (
    count_soft_modes,
    source_regression_metrics,
    worst_imag_vib_cm1,
)
from rpfr_visualizer.qc.models import CheckResult
from rpfr_visualizer.qc.thresholds import QcThresholds


def _decode_scalar(value: Any) -> Any:
    if isinstance(value, (bytes, bytearray)):
        return value.decode("utf-8")
    return value


def _read_attr(group: h5py.Group, name: str, default: Any = None) -> Any:
    if name not in group.attrs:
        return default
    return _decode_scalar(group.attrs[name])


def _tr_residual_status(value: float | None, thresholds: QcThresholds) -> tuple[str, str]:
    if value is None:
        return "info", "TR residual not available"
    if value < thresholds.tr_residual_pass:
        return "pass", f"TR residual {value:.2f} cm-1 within pass band"
    if value <= thresholds.tr_residual_fail:
        return "warn", f"TR residual {value:.2f} cm-1 in warn band"
    return "fail", f"TR residual {value:.2f} cm-1 exceeds fail threshold"


def _imag_worst_status(value: float | None, thresholds: QcThresholds) -> tuple[str, str]:
    if value is None:
        return "pass", "No imaginary modes detected"
    magnitude = abs(value)
    if magnitude < thresholds.imaginary_vib_pass:
        return "pass", f"Worst imaginary mode {value:.2f} cm-1 (numerical noise)"
    if magnitude < thresholds.imaginary_vib_info:
        return "info", f"Worst imaginary mode {value:.2f} cm-1 (small)"
    if magnitude < thresholds.imaginary_vib_warn:
        return "warn", f"Worst imaginary mode {value:.2f} cm-1"
    return "fail", f"Worst imaginary mode {value:.2f} cm-1 exceeds warn threshold"


def _soft_modes_status(
    count: int | None, group: h5py.Group, thresholds: QcThresholds
) -> tuple[str, str]:
    if count is None:
        return "info", "Soft mode count unavailable (need Eckart harmonic_frequencies)"
    if "harmonic_frequencies" in group:
        freqs = group["harmonic_frequencies"][()]
        min_soft = min(
            (float(v) for v in freqs if 0 < float(v) <= thresholds.soft_modes_upper), default=None
        )
        if min_soft is not None and min_soft <= thresholds.soft_modes_fail_single:
            return (
                "fail",
                f"{count} soft modes after TR projection; minimum {min_soft:.1f} cm-1",
            )
    if count >= thresholds.soft_modes_fail_count:
        return (
            "fail",
            f"{count} soft modes after TR projection (>= {thresholds.soft_modes_fail_count})",
        )
    if count >= thresholds.soft_modes_warn_count:
        return "warn", f"{count} soft modes after Eckart TR projection"
    return "pass", f"{count} soft modes after Eckart TR projection"


def _metric_status(
    value: float | None,
    *,
    pass_limit: float,
    warn_limit: float,
    label: str,
) -> tuple[str, str]:
    if value is None:
        return "info", f"{label} N/A (QM9s)"
    if value < pass_limit:
        return "pass", f"{label} {value:.3f} cm-1"
    if value <= warn_limit:
        return "warn", f"{label} {value:.3f} cm-1 in warn band"
    return "fail", f"{label} {value:.3f} cm-1 exceeds fail threshold"


def run_frequency_checks(
    group: h5py.Group,
    *,
    thresholds: QcThresholds,
) -> list[CheckResult]:
    """Run all frequencies.* checks."""
    checks: list[CheckResult] = []

    tr_converged = bool(_read_attr(group, "tr_converged", True))
    checks.append(
        CheckResult(
            check_id="frequencies.tr_converged",
            category="frequencies",
            status="pass" if tr_converged else "fail",
            value=tr_converged,
            threshold="True",
            message="TR removal converged" if tr_converged else "TR removal did not converge",
            source="hdf5_attr",
        )
    )

    max_tr = _read_attr(group, "max_tr_rot_abs")
    max_tr_f = float(max_tr) if max_tr is not None else None
    tr_status, tr_message = _tr_residual_status(max_tr_f, thresholds)
    checks.append(
        CheckResult(
            check_id="frequencies.tr_residual",
            category="frequencies",
            status=tr_status,
            value=max_tr_f,
            threshold=f"< {thresholds.tr_residual_pass} pass; > {thresholds.tr_residual_fail} fail",
            message=tr_message,
            source="hdf5_attr" if max_tr is not None else "computed",
        )
    )

    n_imag = int(_read_attr(group, "n_imag_vib", 0))
    checks.append(
        CheckResult(
            check_id="frequencies.imaginary_vib_count",
            category="frequencies",
            status="pass" if n_imag == 0 else "warn",
            value=n_imag,
            threshold="0 for non-TS",
            message="No imaginary modes" if n_imag == 0 else f"{n_imag} imaginary modes",
            source="hdf5_attr",
        )
    )

    worst = worst_imag_vib_cm1(group)
    imag_status, imag_message = _imag_worst_status(worst, thresholds)
    checks.append(
        CheckResult(
            check_id="frequencies.imaginary_vib_worst",
            category="frequencies",
            status=imag_status,
            value=worst,
            threshold=f"magnitude < {thresholds.imaginary_vib_warn} warn",
            message=imag_message,
            source="hdf5_attr"
            if _read_attr(group, "worst_imag_vib_cm1") is not None
            else "computed",
        )
    )

    soft_count = count_soft_modes(group, thresholds=thresholds)
    soft_status, soft_message = _soft_modes_status(soft_count, group, thresholds)
    checks.append(
        CheckResult(
            check_id="frequencies.soft_modes",
            category="frequencies",
            status=soft_status,
            value=soft_count,
            threshold=(
                f"0 < nu <= {thresholds.soft_modes_upper} cm-1 in Eckart harmonic_frequencies"
            ),
            message=soft_message,
            source="hdf5_attr" if _read_attr(group, "n_soft_modes") is not None else "computed",
        )
    )

    mae, max_err, lengths_match = source_regression_metrics(group)
    checks.append(
        CheckResult(
            check_id="frequencies.source_regression_lengths",
            category="frequencies",
            status="pass" if lengths_match else "fail",
            value=lengths_match,
            threshold="matching mode counts",
            message="Source and harmonic mode counts match"
            if lengths_match
            else "Source and harmonic mode counts differ",
            source="computed",
        )
    )

    mae_status, mae_message = _metric_status(
        mae,
        pass_limit=thresholds.source_mae_pass,
        warn_limit=thresholds.source_mae_warn,
        label="Source MAE",
    )
    checks.append(
        CheckResult(
            check_id="frequencies.source_regression_mae",
            category="frequencies",
            status=mae_status,
            value=mae,
            threshold=f"< {thresholds.source_mae_pass} pass",
            message=mae_message,
            source="hdf5_attr" if _read_attr(group, "source_mae_cm1") is not None else "computed",
        )
    )

    max_status, max_message = _metric_status(
        max_err,
        pass_limit=thresholds.source_max_pass,
        warn_limit=thresholds.source_max_warn,
        label="Source max error",
    )
    checks.append(
        CheckResult(
            check_id="frequencies.source_regression_max",
            category="frequencies",
            status=max_status,
            value=max_err,
            threshold=f"< {thresholds.source_max_pass} pass",
            message=max_message,
            source="hdf5_attr" if _read_attr(group, "source_max_cm1") is not None else "computed",
        )
    )

    return checks

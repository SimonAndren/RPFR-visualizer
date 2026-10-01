"""The command line: argument parsing, dispatch, and the --demo guard.

cli.py is the package's only entry point and had no test at all, so a typo in a
flag name or a broken dispatch would have reached a user before it reached CI.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from rpfr_visualizer.cli import _cmd_gui, build_parser, demo_data_dir


def parse(argv: list[str]):
    return build_parser().parse_args(argv)


# --- dispatch ---------------------------------------------------------------


def test_every_subcommand_binds_a_handler() -> None:
    for argv in (
        ["gui"],
        ["index", "build"],
        ["index", "info", "x.sqlite"],
        ["index", "verify", "x.sqlite"],
    ):
        assert callable(parse(argv).func), argv


def test_a_bare_invocation_is_an_error() -> None:
    """A subcommand is required, so `rpfr-visualizer` alone must not silently do
    nothing."""
    with pytest.raises(SystemExit):
        parse([])


def test_an_unknown_subcommand_is_an_error() -> None:
    with pytest.raises(SystemExit):
        parse(["explore"])


# --- gui --------------------------------------------------------------------


def test_gui_defaults_discover_rather_than_guess() -> None:
    """--data defaults to None, never to a relative path: a hardcoded relative
    default resolves against the process working directory."""
    args = parse(["gui"])
    assert args.data is None
    assert args.index is None
    assert args.drive is None
    assert args.dataset is None
    assert args.demo is False
    assert args.host == "127.0.0.1"
    assert args.port == 8765
    assert args.no_browser is False
    assert args.dev is False


def test_gui_binds_loopback_by_default() -> None:
    """The app has no authentication, so the default bind must not be routable."""
    assert parse(["gui"]).host == "127.0.0.1"


def test_gui_flags_round_trip() -> None:
    args = parse(
        ["gui", "--data", "d", "--index", "i", "--dataset", "qm9s", "--port", "0", "--no-browser"]
    )
    assert (args.data, args.index, args.dataset, args.port) == ("d", "i", "qm9s", 0)
    assert args.no_browser is True


def test_demo_is_off_unless_asked() -> None:
    assert parse(["gui"]).demo is False
    assert parse(["gui", "--demo"]).demo is True


@pytest.mark.parametrize("conflicting", [["--data", "somewhere"], ["--drive", "somewhere"]])
def test_demo_refuses_to_combine_with_a_data_source(conflicting: list[str]) -> None:
    """--demo generates its own dataset. Silently ignoring --data would open
    something other than what the user named."""
    args = parse(["gui", "--demo", *conflicting])
    with pytest.raises(SystemExit) as excinfo:
        _cmd_gui(args)
    assert "--demo" in str(excinfo.value)


def test_demo_data_dir_sits_beside_the_remembered_settings(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("RPFR_VISUALIZER_GUI_CONFIG", str(tmp_path / "gui.json"))
    assert demo_data_dir() == tmp_path / "demo"


# --- index ------------------------------------------------------------------


def test_index_build_defaults_to_the_repo_layout() -> None:
    args = parse(["index", "build"])
    assert args.raw == "data/raw"
    assert args.drive is None
    assert args.out is None
    assert args.no_functional_groups is False


def test_index_build_accepts_a_dataset_subset() -> None:
    args = parse(["index", "build", "--datasets", "qm9s_v2.2", "qm9s_skip"])
    assert args.datasets == ["qm9s_v2.2", "qm9s_skip"]


def test_index_info_and_verify_take_a_path() -> None:
    assert parse(["index", "info", "a.sqlite"]).path == "a.sqlite"
    verify = parse(["index", "verify", "b.sqlite", "--raw", "data/raw"])
    assert (verify.path, verify.raw) == ("b.sqlite", "data/raw")


def test_index_requires_a_subcommand() -> None:
    with pytest.raises(SystemExit):
        parse(["index"])


# --- data -------------------------------------------------------------------


def test_data_requires_a_subcommand() -> None:
    with pytest.raises(SystemExit):
        parse(["data"])


def test_data_list_binds_a_handler() -> None:
    assert callable(parse(["data", "list"]).func)


def test_data_fetch_defaults_to_the_small_tier_and_the_repo_layout() -> None:
    args = parse(["data", "fetch"])
    assert args.tier == "qm9s-1pct"
    assert args.dest == "data"
    assert args.force is False
    assert args.verify_only is False


def test_data_fetch_flags_round_trip() -> None:
    args = parse(["data", "fetch", "--tier", "full", "--dest", "/tmp/x", "--force"])
    assert (args.tier, args.dest, args.force) == ("full", "/tmp/x", True)


def test_data_fetch_rejects_an_unknown_tier() -> None:
    """Naming a tier that does not exist must say which do, not 404 mid-download."""
    from rpfr_visualizer.cli import _cmd_data_fetch

    with pytest.raises(SystemExit) as excinfo:
        _cmd_data_fetch(parse(["data", "fetch", "--tier", "nonesuch"]))
    assert "nonesuch" in str(excinfo.value)
    assert "qm9s-1pct" in str(excinfo.value)


def test_verify_only_reports_missing_files_without_downloading(tmp_path: Path) -> None:
    from rpfr_visualizer.cli import _cmd_data_fetch

    code = _cmd_data_fetch(parse(["data", "fetch", "--verify-only", "--dest", str(tmp_path)]))

    assert code == 1

"""Unit tests for bundled py3Dmol loader."""

from __future__ import annotations

import pytest

pytest.importorskip("py3Dmol")

from rpfr_visualizer.viz.py3dmol_local import (
    bundled_3dmol_js,
    create_view,
    view_to_data_url_iframe,
    view_to_document,
    view_to_iframe_html,
)


def test_bundled_3dmol_js_forces_window_export() -> None:
    source = bundled_3dmol_js()
    assert "3Dmol" in source
    assert len(source) > 100_000
    assert 'window["3Dmol"]=window.$3Dmol=factory();' in source
    assert "module.exports=factory()" not in source


def test_create_view_uses_inline_loader() -> None:
    view = create_view(width=400, height=300)
    assert "var module, exports, define;" in view.startjs
    assert ").call(window);" in view.startjs
    assert "Promise.resolve()" in view.startjs
    assert "force bundled loader" in view.startjs
    assert "width: 400px; height: 300px" in view.startjs
    assert "3Dmol.js failed to load for some reason" not in view.startjs
    assert "failed to initialize" in view.endjs
    assert view.endjs.index("failed to initialize") < view.endjs.index("</script>")


def test_create_view_does_not_fetch_local_url() -> None:
    view = create_view(width=200, height=200)
    assert "loadScriptAsync('__LOCAL__')" not in view.startjs
    assert "typeof $3Dmolpromise === 'undefined'" not in view.startjs


def test_make_html_keeps_guard_inside_script() -> None:
    view = create_view(width=120, height=120)
    html = view._make_html()
    assert "failed to initialize" in html
    after = html.split("</script>", 1)[-1].strip()
    assert "failed to initialize" not in after
    assert "$3Dmolpromise.catch" not in after


def test_iframe_wrappers_embed_viewer() -> None:
    view = create_view(width=180, height=160)
    srcdoc = view_to_iframe_html(view)
    assert srcdoc.startswith("<iframe")
    assert "srcdoc=" in srcdoc
    assert "sandbox=" in srcdoc
    data = view_to_data_url_iframe(view)
    assert 'src="data:text/html;base64,' in data
    assert "180" in data and "160" in data


def test_bundle_src_links_instead_of_inlining() -> None:
    """A host that can serve static files should not pay 537 KB per render.

    Inlining re-parses the whole library in every fresh iframe; a URL is fetched
    and parsed once and then cached.
    """
    linked = view_to_document(create_view(width=400, height=300, bundle_src="/static/3Dmol.js"))
    inlined = view_to_document(create_view(width=400, height=300))

    assert linked.count('<script src="/static/3Dmol.js"></script>') == 1
    assert "var module, exports, define;" not in linked  # the inline IIFE is gone
    assert len(linked) < 20_000
    assert len(inlined) > 500_000

    # The promise patch is what makes either form work, so it must survive.
    assert "force bundled loader" in linked


def test_linked_bundle_loads_before_it_is_used() -> None:
    """Script order is the whole correctness argument.

    A classic ``<script src>`` executes before the following inline script, so
    ``window.$3Dmol`` exists by the time the promise patch tests for it.
    """
    document = view_to_document(create_view(width=400, height=300, bundle_src="/static/3Dmol.js"))

    src_at = document.index('<script src="/static/3Dmol.js">')
    inline_at = document.index("<script>")
    used_at = document.index("$3Dmolpromise")
    assert src_at < inline_at < used_at


def test_view_to_document_returns_a_whole_document() -> None:
    view = create_view(width=400, height=300, bundle_src="/static/3Dmol.js")
    document = view_to_document(view)
    assert document.startswith("<div id=")
    assert document.rstrip().endswith("</script>")

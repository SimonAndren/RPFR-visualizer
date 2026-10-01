"""Bundled 3Dmol.js loader for notebook environments that block CDNs.

The vendored webpack build exports ``window["3Dmol"]`` (no dollar). py3Dmol
expects the global ``$3Dmol``. Jupyter / VS Code / Cursor notebooks often
define CommonJS ``module``/``exports`` or AMD ``define``, which steers the
UMD footer away from ``window``.

Cursor/VS Code notebook webviews also frequently **strip ``<script>`` tags**
from ``IPython.display.HTML`` output. That leaves raw JS source visible (the
``' + '3Dmol.js failed…`` fragments) and prevents the viewer from running.
This loader therefore:

    1. Rewrites the UMD footer to always assign ``window.$3Dmol``
    2. Forces a fresh resolved ``$3Dmolpromise`` (ignores stale notebook globals)
    3. Displays the viewer inside a sandboxed ``iframe`` (``srcdoc`` / data URL)
       so scripts execute even when the parent output sanitizes HTML
"""

from __future__ import annotations

import base64
import html as html_module
import re
from functools import lru_cache
from typing import Any

_UMD_FOOTER = (
    '"object"==typeof exports&&"object"==typeof module?module.exports=factory():'
    '"function"==typeof define&&define.amd?define([],factory):'
    '"object"==typeof exports?exports["3Dmol"]=factory():'
    'root["3Dmol"]=factory();'
)

_WINDOW_FOOTER = 'window["3Dmol"]=window.$3Dmol=factory();'


@lru_cache(maxsize=1)
def bundled_3dmol_js() -> str:
    """Return the vendored 3Dmol-min.js source with a window-forcing UMD footer."""
    from importlib import resources

    path = resources.files("rpfr_visualizer.viz") / "static" / "3Dmol-min.js"
    source = path.read_text(encoding="utf-8")
    if _UMD_FOOTER not in source:
        raise RuntimeError(
            "3Dmol-min.js UMD footer changed; update rpfr_visualizer.viz.py3dmol_local"
        )
    return source.replace(_UMD_FOOTER, _WINDOW_FOOTER, 1)


def _bundle_block() -> str:
    """IIFE that installs ``$3Dmol`` on ``window`` even under notebook UMD traps."""
    return (
        "if (typeof window !== 'undefined' && "
        "!(window.$3Dmol && window.$3Dmol.createViewer)) {\n"
        "  (function () {\n"
        "    var module, exports, define;\n"
        f"    {bundled_3dmol_js()}\n"
        "    if (!(window.$3Dmol && window.$3Dmol.createViewer) && window['3Dmol']) {\n"
        "      window.$3Dmol = window['3Dmol'];\n"
        "    }\n"
        "  }).call(window);\n"
        "}\n"
    )


def _fix_viewer_dimensions(startjs: str, width: int, height: int) -> str:
    """py3Dmol sometimes swaps width/height in the container style."""
    return re.sub(
        r"style=\"position: relative; width: \d+px; height: \d+px;\"",
        f'style="position: relative; width: {width}px; height: {height}px;"',
        startjs,
        count=1,
    )


def _hide_warning_element(startjs: str) -> str:
    """Remove the red failure banner from the initial HTML (we manage loading)."""
    return re.sub(
        r"<p id=\"3dmolwarning_UNIQUEID\"[^>]*>.*?</p>\s*",
        "",
        startjs,
        count=1,
        flags=re.DOTALL,
    )


def _inject_bundle(startjs: str) -> str:
    marker = "var loadScriptAsync = function(uri)"
    bundle = _bundle_block()
    if marker not in startjs:
        inline = f"<script>\n{bundle}{marker}"
        return startjs.replace(f"<script>\n{marker}", inline, 1)
    return startjs.replace(marker, f"{bundle}{marker}", 1)


def _patch_promise(startjs: str) -> str:
    """Never fetch ``__LOCAL__``; always refresh the notebook-global promise."""
    pattern = re.compile(
        r"if\s*\(\s*typeof\s+\$3Dmolpromise\s*===\s*'undefined'\s*\)\s*\{"
        r".*?\$3Dmolpromise\s*=\s*loadScriptAsync\([^)]+\);\s*"
        r"\}",
        flags=re.DOTALL,
    )
    replacement = (
        "// rpfr_visualizer: force bundled loader (ignore stale $3Dmolpromise)\n"
        "$3Dmolpromise = (typeof window !== 'undefined' && window.$3Dmol "
        "&& window.$3Dmol.createViewer)\n"
        "  ? Promise.resolve()\n"
        "  : Promise.reject(new Error('bundled $3Dmol missing'));"
    )
    patched, n = pattern.subn(replacement, startjs, count=1)
    if n:
        return patched
    return re.sub(
        r"\$3Dmolpromise\s*=\s*loadScriptAsync\([^)]+\);",
        (
            "$3Dmolpromise = (typeof window !== 'undefined' && window.$3Dmol "
            "&& window.$3Dmol.createViewer)"
            " ? Promise.resolve()"
            " : Promise.reject(new Error('bundled $3Dmol missing'));"
        ),
        startjs,
        count=1,
    )


def _install_endjs_guard(endjs: str) -> str:
    """Surface init failures inside the script tag (endjs contains ``</script>``)."""
    # Single string literal — avoids ugly fragments if a host ever strips <script>.
    guard = (
        "if (typeof $3Dmolpromise !== 'undefined' && $3Dmolpromise && $3Dmolpromise.catch) {\n"
        "  $3Dmolpromise.catch(function (err) {\n"
        "    var el = document.getElementById('3dmolviewer_UNIQUEID');\n"
        "    if (!el) return;\n"
        "    var msg = (err && err.message) ? err.message : String(err);\n"
        "    el.textContent = '3Dmol.js failed to initialize (' + msg + '). '\n"
        "      + 'Use the matplotlib static view.';\n"
        "    el.style.background = '#ffcccc';\n"
        "    el.style.color = 'black';\n"
        "    el.style.padding = '8px';\n"
        "  });\n"
        "}\n"
    )
    if "</script>" in endjs:
        return endjs.replace("</script>", guard + "</script>", 1)
    return endjs + "\n" + guard


def _viewer_size_px(view: Any, default_w: int = 640, default_h: int = 480) -> tuple[int, int]:
    """Best-effort width/height from the viewer HTML container style."""
    html = getattr(view, "startjs", "") or ""
    match = re.search(r"width:\s*(\d+)px;\s*height:\s*(\d+)px", html)
    if match:
        return int(match.group(1)), int(match.group(2))
    return default_w, default_h


def view_to_iframe_html(view: Any, *, width: int | None = None, height: int | None = None) -> str:
    """Wrap a py3Dmol view in an iframe so scripts survive notebook sanitization."""
    inner = view._make_html()
    w, h = _viewer_size_px(view)
    if width is not None:
        w = width
    if height is not None:
        h = height
    # Prefer srcdoc (no size limits of some data-URL hosts). Escape for an HTML attribute.
    escaped = html_module.escape(inner, quote=True)
    return (
        f'<iframe srcdoc="{escaped}" width="{w}" height="{h}" '
        f'style="border:0;max-width:100%;" '
        f'sandbox="allow-scripts allow-same-origin" '
        f'title="3Dmol viewer"></iframe>'
    )


def view_to_data_url_iframe(
    view: Any, *, width: int | None = None, height: int | None = None
) -> str:
    """Fallback iframe using a base64 data URL (some hosts strip srcdoc)."""
    inner = view._make_html()
    w, h = _viewer_size_px(view)
    if width is not None:
        w = width
    if height is not None:
        h = height
    payload = base64.b64encode(inner.encode("utf-8")).decode("ascii")
    return (
        f'<iframe src="data:text/html;base64,{payload}" width="{w}" height="{h}" '
        f'style="border:0;max-width:100%;" '
        f'sandbox="allow-scripts allow-same-origin" '
        f'title="3Dmol viewer"></iframe>'
    )


def _link_bundle(startjs: str, src: str) -> str:
    """Reference 3Dmol.js by URL instead of inlining 537 KB of it.

    A classic ``<script src>`` earlier in document order is fetched and executed
    before the following inline script, so ``window.$3Dmol`` exists by the time
    :func:`_patch_promise`'s check runs — the same condition the inlined bundle
    satisfies, without the payload.
    """
    tag = f'<script src="{src}"></script>\n'
    marker = "<script>"
    if marker in startjs:
        return startjs.replace(marker, tag + marker, 1)
    return tag + startjs


def create_view(
    width: int = 640,
    height: int = 480,
    *,
    bundle_src: str | None = None,
    **kwargs: Any,
):
    """Create a py3Dmol view using bundled 3Dmol.js.

    By default the bundle is inlined, which is what makes the viewer work in
    Cursor, VS Code and Jupyter without a network fetch.

    *bundle_src* instead references the bundle at that URL. A host that can
    serve static files — a local web server, say — should use it: inlining costs
    537 KB per render and re-parses the whole library in every fresh iframe,
    whereas a URL is fetched and parsed once and then cached. The URL must serve
    the *patched* source from :func:`bundled_3dmol_js`, not the raw file.
    """
    import py3Dmol

    view = py3Dmol.view(width=width, height=height, js="__LOCAL__", **kwargs)
    if bundle_src is None:
        startjs = _inject_bundle(view.startjs)
    else:
        startjs = _link_bundle(view.startjs, bundle_src)
    startjs = _patch_promise(startjs)
    startjs = _fix_viewer_dimensions(startjs, width, height)
    startjs = _hide_warning_element(startjs)
    view.startjs = startjs
    view.endjs = _install_endjs_guard(view.endjs)
    return view


def view_to_document(view: Any) -> str:
    """The full HTML document for a viewer, for use as an ``<iframe src>``.

    Keeps py3Dmol's private ``_make_html()`` inside the module that already owns
    its quirks, rather than having other packages reach for it.
    """
    return str(view._make_html())


def display_view(view) -> None:
    """Display a py3Dmol view via iframe (script-safe in Cursor/VS Code notebooks)."""
    from IPython.display import HTML, display

    # data: URL avoids giant srcdoc attributes (escaped quotes balloon the payload)
    # and survives parent-output sanitizers that strip bare <script> tags.
    display(HTML(view_to_data_url_iframe(view)))


__all__ = [
    "bundled_3dmol_js",
    "create_view",
    "display_view",
    "view_to_data_url_iframe",
    "view_to_document",
    "view_to_iframe_html",
]

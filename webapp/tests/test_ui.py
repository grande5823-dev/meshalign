"""Playwright smoke test for the meshalign console UI.

Boots the real app against a temp runs dir, loads the page in headless
Chromium, selects the demo Connector pair, runs a real alignment, and asserts
the data-testid contract: status pill transitions, metric cards, 3D canvas
rendering, histogram, table, log console, run history.

Run from webapp/:
    pytest tests/test_ui.py
Requires: playwright + chromium (playwright install chromium).
"""
from __future__ import annotations

import os
import sys
import threading
import time

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DEMO_REF = "mech_parts/Connector.stl"
DEMO_TEST = "mech_parts/Connector_test.stl"


@pytest.fixture(scope="module")
def server(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("ui")
    os.environ["MESHALIGN_RUNS_DIR"] = str(tmp / "runs")
    os.environ["MESHALIGN_UPLOADS_DIR"] = str(tmp / "uploads")
    os.environ["MESHALIGN_BENCH_DB"] = str(tmp / "no.db")  # unused here
    import app as appmod
    import importlib
    importlib.reload(appmod)
    appmod.manager = appmod.runner_mod.RunManager(str(tmp / "runs"))

    import uvicorn
    config = uvicorn.Config(appmod.app, host="127.0.0.1", port=8897,
                            log_level="error")
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    for _ in range(50):
        try:
            import socket
            with socket.create_connection(("127.0.0.1", 8897), timeout=0.2):
                break
        except OSError:
            time.sleep(0.1)
    yield "http://127.0.0.1:8897"
    server.should_exit = True
    thread.join(timeout=10)


@pytest.fixture()
def page(server):
    pw = pytest.importorskip("playwright.sync_api")
    from playwright.sync_api import sync_playwright
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": 1440, "height": 900})
        page._console_errors = []
        page.on("pageerror", lambda e: page._console_errors.append(str(e)))
        page.on("console", lambda m: page._console_errors.append(m.text)
                if m.type == "error" else None)
        yield server, page
        browser.close()


def test_console_run_lifecycle(page):
    server, page = page
    page.goto(server, wait_until="networkidle")
    page.wait_for_timeout(1000)

    # idle state: params editable, Stop disabled (§10). Run is disabled
    # until ref+test are selected (§9: submit disabled while invalid)
    assert page.locator("[data-testid=run-status]").get_attribute("data-state") == "idle"
    assert not page.locator("[data-testid=run-button]").is_enabled()
    assert not page.locator("[data-testid=stop-button]").is_enabled()
    assert page.locator("[data-testid=empty-3d]").is_visible()

    # select the demo pair and run
    page.select_option("[data-testid=param-ref]", DEMO_REF)
    page.select_option("[data-testid=param-test]", DEMO_TEST)
    assert page.locator("[data-testid=run-button]").is_enabled()
    page.click("[data-testid=run-button]")

    # running: controls locked, Stop enabled
    page.wait_for_selector('[data-testid=run-status][data-state=running]',
                           timeout=15000)
    assert not page.locator("[data-testid=run-button]").is_enabled()
    assert page.locator("[data-testid=stop-button]").is_enabled()
    assert page.locator("[data-testid=param-mode]").is_disabled()

    # log console streams while running
    page.wait_for_function(
        "() => document.querySelector('[data-testid=log-console]').textContent.length > 0",
        timeout=20000)

    # succeeded
    page.wait_for_selector('[data-testid=run-status][data-state=succeeded]',
                           timeout=120000)
    page.wait_for_timeout(4000)  # let the viewer payload load

    # metric cards carry real numbers (§11: canvas numbers also in DOM)
    mae = float(page.locator("[data-testid=metric-mae_um]").inner_text())
    assert 0 < mae < 1000
    assert 0 < float(page.locator("[data-testid=metric-p95_um]").inner_text()) < 2000
    assert float(page.locator("[data-testid=metric-iou]").inner_text()) > 0.9

    # 3D viewer rendered geometry (readback over the live canvas)
    drew = page.evaluate("""() => new Promise(resolve => {
      const c = document.getElementById('three-canvas');
      const gl = c.getContext('webgl2');
      if (!gl) { resolve(-1); return; }
      requestAnimationFrame(() => requestAnimationFrame(() => {
        const px = new Uint8Array(4); let n = 0;
        for (let x = 100; x < 1300; x += 100) for (let y = 120; y < 800; y += 80) {
          gl.readPixels(x, y, 1, 1, gl.RGBA, gl.UNSIGNED_BYTE, px);
          if (px[0] || px[1] || px[2]) n++;
        }
        resolve(n);
      }));
    })""")
    assert drew > 10
    assert "verts" in page.locator("#mesh-stats").inner_text()

    # histogram tab renders the Plotly bar chart
    page.click("[data-testid=tab-histogram]")
    page.wait_for_timeout(2000)
    assert page.evaluate(
        "() => document.getElementById('hist-plot').classList.contains('js-plotly-plot')")
    bins = page.evaluate(
        "() => { const el = document.getElementById('hist-plot');"
        " return el.data ? el.data[0].x.length : 0; }")
    assert bins == 80

    # table tab: provenance + headline + transform
    page.click("[data-testid=tab-table]")
    page.wait_for_timeout(1500)
    tbl = page.locator("#table-content").inner_text()
    assert "Run ID" in tbl and "Surface DFT MAE" in tbl
    assert "Registration transform" in tbl
    assert "Copy params as JSON" in tbl

    # history row exists and clicking it re-loads the run
    assert page.locator("[data-testid=run-row]").count() >= 1
    page.locator("[data-testid=run-row]").first.click()
    page.wait_for_timeout(2500)
    assert float(page.locator("[data-testid=metric-mae_um]").inner_text()) == mae

    # no JS errors surfaced during the whole flow
    assert page._console_errors == [], page._console_errors


def test_view_toggle_buttons(page):
    server, page = page
    page.goto(server, wait_until="networkidle")
    page.wait_for_selector("[data-testid=run-row]", timeout=15000)
    page.locator("[data-testid=run-row]").first.click()
    page.wait_for_timeout(4000)
    # 3D hud buttons operate without errors
    page.click("[data-testid=view-reset]")
    page.click("[data-testid=view-fit]")
    page.click("[data-testid=view-wireframe]")
    page.click("[data-testid=view-wireframe]")
    page.click("[data-testid=view-points]")
    page.click("[data-testid=view-points]")
    page.click("[data-testid=view-reference]")  # solid
    page.click("[data-testid=view-reference]")  # hidden
    page.click("[data-testid=view-reference]")  # back to wireframe
    assert page._console_errors == [], page._console_errors


def test_validation_blocks_run(page):
    server, page = page
    page.goto(server, wait_until="networkidle")
    page.wait_for_timeout(800)
    # with no selection, Run is disabled
    assert not page.locator("[data-testid=run-button]").is_enabled()
    # select only the reference
    page.select_option("[data-testid=param-ref]", DEMO_REF)
    assert not page.locator("[data-testid=run-button]").is_enabled()


def _open_all_advanced(page):
    page.evaluate("() => { const d = document.getElementById('adv'); if (d) d.open = true; }")


def test_every_command_has_a_form_control_per_cli_option(page):
    """CLI/web parity: each command's registry params all render as controls."""
    server, page = page
    page.goto(server, wait_until="networkidle")
    page.wait_for_selector("[data-testid=cmd-run]")
    from meshalign.commands import web_schema
    for cmd, spec in web_schema().items():
        page.click(f"[data-testid=cmd-{cmd}]")
        _open_all_advanced(page)
        for p in spec["params"]:
            assert page.locator(f"[data-testid=param-{p['name']}]").count() == 1, (cmd, p["name"])
        assert page.locator("[data-testid=cli-line]").inner_text().startswith(f"meshalign {cmd}")


def test_cli_line_tracks_form(page):
    server, page = page
    page.goto(server, wait_until="networkidle")
    page.wait_for_selector("[data-testid=cmd-register]")
    page.click("[data-testid=cmd-register]")
    page.select_option("[data-testid=param-ref]", DEMO_REF)
    page.select_option("[data-testid=param-test]", DEMO_TEST)
    page.select_option("[data-testid=param-path]", "global")
    page.uncheck("[data-testid=param-write]")
    line = page.locator("[data-testid=cli-line]").inner_text()
    assert line == ("meshalign register data/mech_parts/Connector.stl "
                    "data/mech_parts/Connector_test.stl --global --no-write"), line


def test_register_and_chamfer_from_the_ui(page):
    server, page = page
    page.goto(server, wait_until="networkidle")
    page.wait_for_selector("[data-testid=cmd-register]")
    # register: viewer shows the aligned pair, rotation/translation cards fill in
    page.click("[data-testid=cmd-register]")
    page.select_option("[data-testid=param-ref]", DEMO_REF)
    page.select_option("[data-testid=param-test]", DEMO_TEST)
    page.select_option("[data-testid=param-mode]", "fast")
    assert page.locator("[data-testid=run-button]").inner_text() == "Register"
    page.click("[data-testid=run-button]")
    page.wait_for_selector('[data-testid=run-status][data-state=succeeded]', timeout=120000)
    page.wait_for_function(
        "() => parseFloat(document.querySelector('[data-testid=metric-rotation_deg]').textContent) >= 0",
        timeout=15000)
    page.wait_for_timeout(2000)
    assert "verts" in page.locator("#mesh-stats").inner_text()
    page.click("[data-testid=tab-table]")
    page.wait_for_timeout(1500)
    tbl = page.locator("#table-content").inner_text()
    assert "Registration transform" in tbl and "aligned.stl" in tbl
    # chamfer: no viewer, result lands on the Table tab
    page.click("[data-testid=cmd-chamfer]")
    page.select_option("[data-testid=param-a]", DEMO_REF)
    page.select_option("[data-testid=param-b]", DEMO_TEST)
    page.click("[data-testid=run-button]")
    page.wait_for_selector('[data-testid=run-status][data-state=succeeded]', timeout=120000)
    page.wait_for_function(
        "() => parseFloat(document.querySelector('[data-testid=metric-chamfer_mm]').textContent) > 0",
        timeout=15000)
    assert "Chamfer distance" in page.locator("#table-content").inner_text()
    # history shows both commands, with labels
    rows = page.locator("[data-testid=run-row]").all_inner_texts()
    assert any("chamfer" in r.lower() for r in rows) and any("register" in r.lower() for r in rows)
    assert page._console_errors == [], page._console_errors


def test_ensemble_requires_two_meshes(page):
    server, page = page
    page.goto(server, wait_until="networkidle")
    page.wait_for_selector("[data-testid=cmd-ensemble]")
    page.click("[data-testid=cmd-ensemble]")
    assert not page.locator("[data-testid=run-button]").is_enabled()
    boxes = page.locator("[data-testid=param-meshes] input[type=checkbox]")
    boxes.nth(0).check()
    assert not page.locator("[data-testid=run-button]").is_enabled()
    boxes.nth(1).check()
    assert page.locator("[data-testid=run-button]").is_enabled()

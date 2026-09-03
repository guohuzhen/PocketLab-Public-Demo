from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen
from uuid import uuid4

from PIL import Image
from playwright.sync_api import Browser, Error, Page, sync_playwright

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_VIEWPORT = {"width": 1440, "height": 900}


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        return int(listener.getsockname()[1])


def _wait_for_health(base_url: str, process: subprocess.Popen[str]) -> None:
    deadline = time.monotonic() + 30
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"PocketLab server exited before readiness (code {process.returncode}).")
        try:
            with urlopen(f"{base_url}/health", timeout=1) as response:
                if response.status == 200:
                    return
        except (OSError, URLError) as exc:
            last_error = exc
        time.sleep(0.15)
    raise RuntimeError("PocketLab server did not become healthy within 30 seconds.") from last_error


def _capture(page: Page, target: Path, frames: list[Path]) -> None:
    page.screenshot(path=target, full_page=False)
    frames.append(target)


def _write_demo_gif(frames: list[Path], target: Path) -> None:
    if not frames:
        raise ValueError("at least one screenshot is required to build the demo GIF")
    rendered: list[Image.Image] = []
    for source in frames:
        with Image.open(source) as original:
            frame = original.convert("RGB")
            width = 960
            height = round(frame.height * width / frame.width)
            rendered.append(frame.resize((width, height), Image.Resampling.LANCZOS))
    durations = [2200] * len(rendered)
    durations[0] = 3000
    durations[-1] = 4200
    rendered[0].save(
        target,
        save_all=True,
        append_images=rendered[1:],
        duration=durations,
        loop=0,
        optimize=True,
        disposal=2,
    )
    for image in rendered:
        image.close()


def _launch_browser(playwright, channel: str | None, headed: bool) -> Browser:
    kwargs: dict[str, object] = {"headless": not headed}
    if channel:
        kwargs["channel"] = channel
    try:
        return playwright.chromium.launch(**kwargs)
    except Error as exc:
        raise RuntimeError(
            "Chromium is unavailable. Run `uv run playwright install chromium`, "
            "or pass `--browser-channel msedge`/`chrome` for an installed browser."
        ) from exc


def _run_journey(
    browser: Browser,
    base_url: str,
    *,
    artifacts_dir: Path | None,
) -> dict[str, object]:
    context = browser.new_context(viewport=DEFAULT_VIEWPORT)
    page = context.new_page()
    console_errors: list[str] = []
    page.on(
        "console",
        lambda message: console_errors.append(message.text) if message.type == "error" else None,
    )
    username = "pocketlab_demo"
    password = f"PocketLab-e2e-{uuid4().hex}"
    frame_dir_context = tempfile.TemporaryDirectory(prefix="pocketlab-e2e-frames-")
    frame_dir = Path(frame_dir_context.name)
    frames: list[Path] = []

    try:
        page.goto(f"{base_url}/", wait_until="domcontentloaded")
        page.locator("#registerTab").click()
        page.locator("#registerUsername").fill(username)
        page.locator("#registerDisplayName").fill("PocketLab Demo")
        page.locator("#registerPassword").fill(password)
        page.locator("#registerSubmit").click()
        page.wait_for_url(f"{base_url}/app", timeout=15_000)
        page.get_by_role("heading", name="工作台", exact=True).wait_for()
        _capture(page, frame_dir / "01-workspace.png", frames)
        if artifacts_dir is not None:
            page.screenshot(path=artifacts_dir / "workspace.png")

        page.goto(f"{base_url}/app/cases/new", wait_until="domcontentloaded")
        page.locator("#showcaseDiagnosticStartButton").click()
        page.wait_for_url(f"{base_url}/app/cases/*", timeout=15_000)
        page.get_by_text("回放原始偏载工况基线", exact=True).wait_for()
        _capture(page, frame_dir / "02-diagnostic-ready.png", frames)

        page.locator("#publicDiagnosticRunButton").click()
        page.get_by_text("只重新均匀分布同一批衣物", exact=True).wait_for()
        _capture(page, frame_dir / "03-diagnostic-evidence.png", frames)

        page.locator("#publicDiagnosticRunButton").click()
        page.locator("#finalReportBlock").wait_for(state="visible")
        final_report = page.locator("#finalReportBlock").inner_text()
        assert "重新均匀分布衣物后，振动强度显著下降" in final_report
        assert "服务器冻结回放 · 0 次模型请求" in final_report
        agent_message = page.locator("#diagnosticAgentMessage").inner_text()
        assert "终止向量已满足" in agent_message
        assert "还需要" not in agent_message
        page.locator("#finalReportBlock").evaluate(
            "element => window.scrollTo(0, element.offsetTop - 90)"
        )
        _capture(page, frame_dir / "04-diagnostic-report.png", frames)
        if artifacts_dir is not None:
            page.screenshot(path=artifacts_dir / "diagnostic-report.png")

        page.goto(f"{base_url}/app/explore", wait_until="domcontentloaded")
        page.locator("#showcaseExplorationStartButton").click()
        page.wait_for_url(f"{base_url}/app/explore/general/runs/*", timeout=15_000)
        page.get_by_text("近距离参考位置 · 光线 · 第 1 轮回放", exact=True).wait_for()
        _capture(page, frame_dir / "05-exploration-ready.png", frames)

        expected_tasks = (
            "距离加倍位置 · 光线 · 第 1 轮回放",
            "近距离参考位置 · 光线 · 第 2 轮回放",
            "距离加倍位置 · 光线 · 第 2 轮回放",
        )
        for index, expected_task in enumerate(expected_tasks, start=1):
            page.locator("#generalSimulateMeasurement").click()
            page.get_by_text(expected_task, exact=True).wait_for()
            _capture(page, frame_dir / f"{index + 5:02d}-exploration-step.png", frames)

        page.locator("#generalSimulateMeasurement").click()
        page.locator("#generalFinalReport").wait_for(state="visible")
        report_text = page.locator("#generalFinalReport").inner_text()
        assert "灯距增大后，照度稳定降到明显更低的平台" in report_text
        assert "近距离参考位置" in report_text
        assert "距离加倍位置" in report_text
        assert "physical false" in page.locator("body").inner_text()
        page.get_by_text("条件对比可视化", exact=True).scroll_into_view_if_needed()
        _capture(page, frame_dir / "09-optical-report.png", frames)
        if artifacts_dir is not None:
            page.screenshot(path=artifacts_dir / "optical-exploration-report.png")

        exploration_url = page.url
        page.reload(wait_until="domcontentloaded")
        page.locator("#generalFinalReport").wait_for(state="visible")
        assert "灯距增大后" in page.locator("#generalReportAnswer").inner_text()

        page.goto(f"{base_url}/app/history", wait_until="domcontentloaded")
        page.wait_for_function(
            "document.querySelector('#caseHistoryCount')?.textContent === '1' "
            "&& document.querySelector('#explorationHistoryCount')?.textContent === '1' "
            "&& document.querySelector('#sessionHistoryCount')?.textContent === '2'",
            timeout=10_000,
        )
        page.locator("#diagnosticHistoryModule summary").click()
        page.locator("#explorationHistoryModule summary").click()
        assert "洗衣机脱水振动：偏载还是地面放大？" in page.locator(
            "#diagnosticHistoryModule"
        ).inner_text()
        assert "灯离远一倍，照度会怎样变化？" in page.locator(
            "#explorationHistoryModule"
        ).inner_text()
        assert page.locator("#caseHistoryCount").inner_text() == "1"
        assert page.locator("#explorationHistoryCount").inner_text() == "1"
        assert page.locator("#sessionHistoryCount").inner_text() == "2"
        _capture(page, frame_dir / "10-history.png", frames)
        if artifacts_dir is not None:
            page.screenshot(path=artifacts_dir / "history.png")
            _write_demo_gif(frames, artifacts_dir / "pocketlab-demo.gif")

        assert not console_errors, f"browser console errors: {console_errors}"
        return {
            "passed": True,
            "registration": "passed",
            "diagnostic_clicks": 2,
            "exploration_clicks": 4,
            "refresh_restore": "passed",
            "history_restore": "passed",
            "duplicate_writes": 0,
            "console_errors": 0,
            "exploration_url_shape": exploration_url.replace(base_url, "<base_url>"),
            "artifacts": (
                sorted(path.name for path in artifacts_dir.iterdir())
                if artifacts_dir is not None
                else []
            ),
        }
    finally:
        context.close()
        frame_dir_context.cleanup()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run PocketLab's zero-key browser release-candidate journey."
    )
    parser.add_argument(
        "--browser-channel",
        default=os.getenv("POCKETLAB_E2E_BROWSER_CHANNEL", "").strip() or None,
        help="Optional installed Chromium channel, for example msedge or chrome.",
    )
    parser.add_argument("--headed", action="store_true")
    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        default=None,
        help="Optional directory for reviewed screenshots and a 20–40 second GIF.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    artifacts_dir = args.artifacts_dir.resolve() if args.artifacts_dir else None
    if artifacts_dir is not None:
        artifacts_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="pocketlab-browser-e2e-") as temp_dir:
        port = _free_port()
        base_url = f"http://127.0.0.1:{port}"
        environment = os.environ.copy()
        environment.update(
            {
                "POCKETLAB_DB_PATH": str(Path(temp_dir) / "candidate.sqlite3"),
                "LLM_API_KEY": "",
                "LLM_BASE_URL": "",
                "LLM_MODEL": "",
                "LLM_REASONING_STRATEGY": "high",
                "PYTHONUNBUFFERED": "1",
            }
        )
        process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "uvicorn",
                "pocketlab.main:app",
                "--host",
                "127.0.0.1",
                "--port",
                str(port),
                "--log-level",
                "warning",
            ],
            cwd=ROOT,
            env=environment,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        try:
            _wait_for_health(base_url, process)
            with sync_playwright() as playwright:
                browser = _launch_browser(playwright, args.browser_channel, args.headed)
                try:
                    report = _run_journey(browser, base_url, artifacts_dir=artifacts_dir)
                finally:
                    browser.close()
        finally:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
            if os.name == "nt":
                time.sleep(0.5)

    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

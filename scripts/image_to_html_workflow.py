"""Iterative workflow to rebuild a webpage from a screenshot using mini-swe-agent.

This script demonstrates how to:
- Capture a screenshot of a target URL with Playwright.
- Ask mini-swe-agent (Python bindings) to write HTML/CSS that matches the screenshot.
- Render the generated HTML, compare it to the reference screenshot, and iterate.

The agent is guided with compact image metadata (base64 previews and diff stats)
to refine the HTML over multiple iterations without manually pasting full code.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
import argparse
from pathlib import Path
from typing import Iterable

from PIL import Image, ImageChops, ImageStat
from minisweagent.agents.default import DefaultAgent
from minisweagent.environments.local import LocalEnvironment
from minisweagent.models import get_model
from playwright.async_api import async_playwright


@dataclass
class IterationResult:
    iteration: int
    agent_status: str
    agent_message: str
    html_path: Path
    render_path: Path
    diff_stats: dict[str, float | int | str]


class ImageToHtmlRunner:
    """Run iterative HTML generation from a reference screenshot."""

    def __init__(
        self,
        model_name: str = "gpt-4o-mini",
        viewport: tuple[int, int] = (1280, 720),
        workspace: Path | str | None = None,
        model_config: dict | None = None,
        navigation_timeout_ms: int = 120_000,
    ):
        self.model_name = model_name
        self.viewport = viewport
        self.workspace = Path(workspace) if workspace else Path.cwd() / "image_to_html_runs"
        self.workspace.mkdir(parents=True, exist_ok=True)
        self.navigation_timeout_ms = navigation_timeout_ms
        self.agent = DefaultAgent(
            get_model(input_model_name=model_name, config=model_config), LocalEnvironment(cwd=str(self.workspace))
        )

    async def _capture_page(self, url: str, screenshot_path: Path) -> Path:
        async with async_playwright() as p:
            browser = await p.chromium.launch()
            context = await browser.new_context(
                viewport={"width": self.viewport[0], "height": self.viewport[1]}, ignore_https_errors=True
            )
            page = await context.new_page()
            page.set_default_timeout(self.navigation_timeout_ms)
            await page.goto(url, wait_until="networkidle")
            await page.wait_for_timeout(2_000)
            await page.screenshot(path=str(screenshot_path), full_page=True)
            await browser.close()
        return screenshot_path

    async def _render_html(self, html_path: Path, screenshot_path: Path) -> Path:
        async with async_playwright() as p:
            browser = await p.chromium.launch()
            context = await browser.new_context(
                viewport={"width": self.viewport[0], "height": self.viewport[1]}, ignore_https_errors=True
            )
            page = await context.new_page()
            page.set_default_timeout(self.navigation_timeout_ms)
            await page.goto(html_path.resolve().as_uri())
            await page.wait_for_timeout(2_000)
            await page.screenshot(path=str(screenshot_path), full_page=True)
            await browser.close()
        return screenshot_path

    def _diff_images(self, reference: Path, candidate: Path) -> dict[str, float | int | str]:
        ref_image = Image.open(reference).convert("RGB")
        cand_image = Image.open(candidate).convert("RGB")
        ref_resized = ref_image.resize(self.viewport)
        cand_resized = cand_image.resize(self.viewport)
        diff = ImageChops.difference(ref_resized, cand_resized)
        stat = ImageStat.Stat(diff)
        mean_diff = sum(stat.mean) / len(stat.mean)
        bbox = diff.getbbox()
        diff_area = 0
        if bbox:
            dx = max(bbox[2] - bbox[0], 0)
            dy = max(bbox[3] - bbox[1], 0)
            diff_area = dx * dy
        return {
            "mean_diff": round(mean_diff, 3),
            "diff_area_px": diff_area,
            "bbox": str(bbox) if bbox else "None",
        }

    def _agent_prompt(
        self,
        *,
        reference_image: Path,
        iteration: int,
        prior_html: Path | None,
        diff_stats: dict[str, float | int | str] | None,
        prior_render: Path | None,
    ) -> tuple[str, list[Path]]:
        images = [reference_image]
        base_prompt = [
            "You are recreating a webpage as standalone HTML/CSS.",
            "Constraints:",
            "- Do not fetch external resources; embed assets with data URIs or basic shapes.",
            "- Save your HTML to 'generated.html' in the working directory.",
            "The first attached image is the reference screenshot you must match.",
        ]
        if prior_html:
            base_prompt.append("\nCurrent HTML (path): " + str(prior_html))
        if diff_stats and prior_render:
            base_prompt.append(f"\nSecond attached image: previous render (iteration {iteration - 1}).")
            base_prompt.append(f"Previous render difference stats: {diff_stats}")
            base_prompt.append("Focus on reducing the mean difference and matching layout/colors.")
            images.append(prior_render)
        base_prompt.append(
            "Return a single bash command that writes the improved HTML to generated.html (e.g., using cat <<'EOF')."
        )
        return "\n".join(base_prompt), images

    def _run_agent(self, prompt: str, images: list[Path]) -> tuple[str, str]:
        return self.agent.run(prompt, images=images)

    def _write_iteration_html(self, iteration_dir: Path, html_file: Path) -> Path:
        iteration_html = iteration_dir / "generated.html"
        if html_file.exists():
            iteration_html.write_text(html_file.read_text(), encoding="utf-8")
        return iteration_html

    def run(self, url: str, iterations: int = 2) -> list[IterationResult]:
        results: list[IterationResult] = []
        reference_path = self.workspace / "reference.png"
        html_path = self.workspace / "generated.html"

        asyncio.run(self._capture_page(url, reference_path))

        for i in range(iterations):
            iteration_dir = self.workspace / f"iter_{i+1}"
            iteration_dir.mkdir(exist_ok=True)

            diff_stats: dict[str, float | int | str] | None = None
            prior_render: Path | None = None
            if i > 0:
                prior_render = results[-1].render_path
                diff_stats = self._diff_images(reference_path, prior_render)

            prompt, prompt_images = self._agent_prompt(
                reference_image=reference_path,
                iteration=i + 1,
                prior_html=html_path if html_path.exists() else None,
                diff_stats=diff_stats,
                prior_render=prior_render,
            )
            status, message = self._run_agent(prompt, prompt_images)

            if html_path.exists():
                iteration_html_path = self._write_iteration_html(iteration_dir, html_path)
            else:
                iteration_html_path = iteration_dir / "generated.html"

            render_output = iteration_dir / "render.png"
            asyncio.run(self._render_html(iteration_html_path, render_output))

            iteration_result = IterationResult(
                iteration=i + 1,
                agent_status=status,
                agent_message=message,
                html_path=iteration_html_path,
                render_path=render_output,
                diff_stats=diff_stats or {},
            )
            results.append(iteration_result)

        return results


def visualize_results(results: Iterable[IterationResult]):
    for result in results:
        print(f"Iteration {result.iteration}")
        print(f"  Agent status: {result.agent_status}")
        print(f"  Agent message: {result.agent_message[:200]}...")
        print(f"  HTML: {result.html_path}")
        print(f"  Render: {result.render_path}")
        print(f"  Diff stats: {result.diff_stats}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Iteratively recreate a webpage screenshot as HTML.")
    parser.add_argument("--url", default="https://example.com", help="Target URL to capture.")
    parser.add_argument("--iterations", type=int, default=2, help="Number of refinement iterations to run.")
    parser.add_argument("--model", default="gpt-4o-mini", help="Model name for the agent.")
    parser.add_argument("--workspace", default=None, help="Workspace directory for artifacts.")
    parser.add_argument("--api-key", dest="api_key", default=None, help="Override model API key.")
    parser.add_argument("--base-url", dest="base_url", default=None, help="Override model base URL (if supported).")
    parser.add_argument(
        "--model-class",
        dest="model_class",
        default=None,
        help="Optional model class override (e.g., 'litellm', 'openrouter').",
    )
    parser.add_argument(
        "--timeout",
        dest="navigation_timeout",
        type=int,
        default=120,
        help="Navigation timeout in seconds for page loads and renders.",
    )

    args = parser.parse_args()

    model_config: dict | None = None
    model_kwargs: dict[str, str] = {}
    if args.api_key:
        model_kwargs["api_key"] = args.api_key
    if args.base_url:
        # Provide both base_url and api_base to maximize compatibility with different providers.
        model_kwargs["base_url"] = args.base_url
        model_kwargs["api_base"] = args.base_url
        model_kwargs["openai_api_base"] = args.base_url
        model_kwargs.setdefault("api_type", "openai")
        model_kwargs.setdefault("custom_llm_provider", "openai")
    if model_kwargs:
        model_config = {"model_kwargs": model_kwargs}
    if args.model_class:
        model_config = (model_config or {}) | {"model_class": args.model_class}

    runner = ImageToHtmlRunner(
        model_name=args.model,
        workspace=Path(args.workspace) if args.workspace else None,
        model_config=model_config,
        navigation_timeout_ms=args.navigation_timeout * 1000,
    )
    output = runner.run(args.url, iterations=args.iterations)
    visualize_results(output)

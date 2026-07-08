from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from track2_captioner.harness import DEFAULT_STYLES, run_harness


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Track 2 harness and validate output shape.")
    parser.add_argument("--input", type=Path, default=Path("sample_input/tasks.json"), help="Tasks JSON path.")
    parser.add_argument("--output", type=Path, default=Path("sample_output/eval_results.json"), help="Results JSON path.")
    parser.add_argument("--real", action="store_true", help="Call the configured model backend instead of dry-run mode.")
    parser.add_argument("--max-frames", type=int, default=None, help="Override TRACK2_MAX_FRAMES.")
    parser.add_argument("--runtime-budget", type=int, default=None, help="Override TRACK2_RUNTIME_BUDGET_SECONDS.")
    return parser.parse_args()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def validate_results(tasks: list[dict[str, Any]], results: list[dict[str, Any]]) -> list[str]:
    errors: list[str] = []
    if len(tasks) != len(results):
        errors.append(f"Expected {len(tasks)} results, got {len(results)}.")

    result_by_id = {str(item.get("task_id", "")): item for item in results}
    for index, task in enumerate(tasks, start=1):
        task_id = str(task.get("task_id", "") or f"task_{index}")
        result = result_by_id.get(task_id)
        if result is None:
            errors.append(f"Missing result for task_id={task_id}.")
            continue

        captions = result.get("captions")
        if not isinstance(captions, dict):
            errors.append(f"Result {task_id} has no captions object.")
            continue

        styles = [str(style) for style in (task.get("styles") or DEFAULT_STYLES)]
        for style in styles:
            caption = captions.get(style)
            if not isinstance(caption, str) or not caption.strip():
                errors.append(f"Result {task_id} has empty caption for style={style}.")

    return errors


def main() -> int:
    args = parse_args()
    tasks = read_json(args.input)
    if not isinstance(tasks, list):
        raise ValueError("Input tasks file must contain a JSON array.")

    os.environ["TRACK2_INPUT"] = str(args.input)
    os.environ["TRACK2_OUTPUT"] = str(args.output)
    if not args.real:
        os.environ["TRACK2_DRY_RUN"] = "true"
    if args.max_frames is not None:
        os.environ["TRACK2_MAX_FRAMES"] = str(args.max_frames)
    if args.runtime_budget is not None:
        os.environ["TRACK2_RUNTIME_BUDGET_SECONDS"] = str(args.runtime_budget)

    started = time.monotonic()
    exit_code = run_harness(args.input, args.output)
    elapsed = time.monotonic() - started

    results = read_json(args.output)
    if not isinstance(results, list):
        raise ValueError("Output results file must contain a JSON array.")

    errors = validate_results(tasks, results)
    mode = "real" if args.real else "dry-run"
    print(f"Track 2 {mode} eval: {len(results)} result(s), {elapsed:.1f}s elapsed.")
    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        return 1
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())

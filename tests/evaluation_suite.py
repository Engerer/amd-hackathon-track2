"""Standalone evaluation suite for the Track 2 multimodal captioning pipeline.

Run with:
    python -m tests.evaluation_suite --dry-run

This script loads the held-out eval dataset and calibration subset, runs a
multi-model judge panel, calculates key metrics, computes calibration
correlation against human ratings, and prints a formatted report.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import statistics
import sys
import time
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_TESTS_DIR = Path(__file__).resolve().parent
_EVAL_DATASET_PATH = _TESTS_DIR / "eval_dataset.json"
_CALIBRATION_SUBSET_PATH = _TESTS_DIR / "calibration_subset.json"

# All four target styles
ALL_STYLES = ["formal", "sarcastic", "humorous_tech", "humorous_non_tech"]


# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------

def load_eval_dataset(path: Path | None = None) -> list[dict[str, Any]]:
    """Load the held-out evaluation dataset."""
    path = path or _EVAL_DATASET_PATH
    with path.open(encoding="utf-8") as fh:
        data = json.load(fh)
    assert isinstance(data, list), "eval_dataset.json must be a JSON array"
    return data


def load_calibration_subset(path: Path | None = None) -> list[dict[str, Any]]:
    """Load the human-reviewed calibration subset."""
    path = path or _CALIBRATION_SUBSET_PATH
    with path.open(encoding="utf-8") as fh:
        data = json.load(fh)
    assert isinstance(data, list), "calibration_subset.json must be a JSON array"
    return data


# ---------------------------------------------------------------------------
# Judge panel
# ---------------------------------------------------------------------------

def _parse_judge_response(raw: str) -> dict[str, Any]:
    """Extract the JSON object from a judge model response."""
    # Try to find JSON in the response
    match = re.search(r"\{[^{}]*\}", raw, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    # Fallback: try the whole string
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {}


def _default_judge_scores() -> dict[str, Any]:
    """Return a zeroed-out judge score dict for fallback."""
    return {
        "factual_accuracy": 0.0,
        "subject_action_coverage": 0.0,
        "unsupported_claims": [],
        "omissions": [],
        "style_strength": 0.0,
        "naturalness": 0.0,
        "concision": 0.0,
        "overall_score": 0.0,
        "repair_instructions": "",
    }


def run_judge_panel(
    evidence: dict[str, Any],
    caption: str,
    style: str,
    *,
    judge_fn: Any | None = None,
    num_judges: int = 3,
    allow_stub: bool = False,
) -> dict[str, Any]:
    """Run a multi-model judge panel and average numeric scores.

    Parameters
    ----------
    evidence : dict
        The evidence ledger or ground-truth observations.
    caption : str
        The generated caption to evaluate.
    style : str
        The target style (e.g. "formal").
    judge_fn : callable, optional
        A callable ``(evidence, caption, style) -> str`` that returns raw JSON
        from the judge model. If *None*, a stub is available only when
        ``allow_stub=True`` for an explicit dry run.
    num_judges : int
        How many independent judge calls to make and average.

    Returns
    -------
    dict
        Averaged judge scores in the schema defined by ``judge.txt``.
    """
    if judge_fn is None:
        if not allow_stub:
            raise ValueError("A real judge_fn is required outside explicit dry-run mode.")
        # Explicit dry-run stub: deterministic values for exercising aggregation.
        return {
            "factual_accuracy": 0.95,
            "subject_action_coverage": 0.90,
            "unsupported_claims": [],
            "omissions": [],
            "style_strength": 0.85,
            "naturalness": 0.88,
            "concision": 0.90,
            "overall_score": 0.90,
            "repair_instructions": "",
        }

    numeric_keys = [
        "factual_accuracy",
        "subject_action_coverage",
        "style_strength",
        "naturalness",
        "concision",
        "overall_score",
    ]
    collected: dict[str, list[float]] = {k: [] for k in numeric_keys}
    all_unsupported: list[str] = []
    all_omissions: list[str] = []
    all_repairs: list[str] = []

    for _ in range(num_judges):
        try:
            raw = judge_fn(evidence, caption, style)
            scores = _parse_judge_response(raw)
            if not scores:
                scores = _default_judge_scores()
        except Exception:
            scores = _default_judge_scores()

        for key in numeric_keys:
            val = scores.get(key, 0.0)
            collected[key].append(float(val) if isinstance(val, (int, float)) else 0.0)

        all_unsupported.extend(scores.get("unsupported_claims", []))
        all_omissions.extend(scores.get("omissions", []))
        repair = scores.get("repair_instructions", "")
        if repair:
            all_repairs.append(repair)

    averaged = {k: statistics.mean(v) for k, v in collected.items()}
    averaged["unsupported_claims"] = sorted(set(all_unsupported))
    averaged["omissions"] = sorted(set(all_omissions))
    averaged["repair_instructions"] = " | ".join(all_repairs) if all_repairs else ""
    return averaged


# ---------------------------------------------------------------------------
# Metric calculators
# ---------------------------------------------------------------------------

def _extract_nouns_verbs(caption: str) -> list[str]:
    """Rough extraction of content words (nouns/verbs) from a caption.

    This is a simplified heuristic — splits on whitespace, strips
    punctuation, and filters out common stop words.
    """
    stop_words = {
        "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
        "have", "has", "had", "do", "does", "did", "will", "would", "shall",
        "should", "may", "might", "must", "can", "could", "to", "of", "in",
        "for", "on", "with", "at", "by", "from", "as", "into", "through",
        "during", "before", "after", "above", "below", "between", "out",
        "off", "over", "under", "again", "further", "then", "once", "here",
        "there", "when", "where", "why", "how", "all", "both", "each",
        "few", "more", "most", "other", "some", "such", "no", "nor", "not",
        "only", "own", "same", "so", "than", "too", "very", "and", "but",
        "or", "if", "while", "because", "until", "that", "which", "who",
        "whom", "this", "these", "those", "it", "its", "like", "just",
        "about", "up", "what", "apparently", "clearly", "naturally",
    }
    words = re.findall(r"[a-zA-Z]+", caption.lower())
    return [w for w in words if w not in stop_words and len(w) > 2]


def unsupported_noun_verb_rate(
    caption: str,
    ground_truth: dict[str, Any],
) -> float:
    """Fraction of content words in the caption not found in ground truth text.

    Lower is better. 0.0 means every content word is grounded.
    """
    gt_text = " ".join(
        str(v) if isinstance(v, str) else " ".join(str(x) for x in v)
        for v in ground_truth.values()
        if isinstance(v, (str, list))
    ).lower()
    content_words = _extract_nouns_verbs(caption)
    if not content_words:
        return 0.0
    unsupported = sum(1 for w in content_words if w not in gt_text)
    return unsupported / len(content_words)


def caption_accuracy(judge_scores: dict[str, Any]) -> float:
    """Return the factual_accuracy score from judge output."""
    return float(judge_scores.get("factual_accuracy", 0.0))


def style_match(judge_scores: dict[str, Any]) -> float:
    """Return the style_strength score from judge output."""
    return float(judge_scores.get("style_strength", 0.0))


def fallback_error_rate(results: list[dict[str, Any]]) -> float:
    """Fraction of results that used fallback captions."""
    if not results:
        return 0.0
    fallback_keywords = [
        "visible subjects and activity",
        "activity running like a process",
        "standing still simply was not",
        "scene insists on staying busy",
    ]
    fallback_count = 0
    total = 0
    for result in results:
        captions = result.get("captions", {})
        for caption in captions.values():
            total += 1
            if any(kw in caption for kw in fallback_keywords):
                fallback_count += 1
    return fallback_count / max(1, total)


def p95_latency(latencies: list[float]) -> float:
    """Compute the 95th-percentile latency from a list of durations (seconds)."""
    if not latencies:
        return 0.0
    sorted_lat = sorted(latencies)
    idx = int(math.ceil(0.95 * len(sorted_lat))) - 1
    return sorted_lat[max(0, idx)]


# ---------------------------------------------------------------------------
# Calibration correlation
# ---------------------------------------------------------------------------

def _pearson_r(xs: list[float], ys: list[float]) -> float:
    """Compute Pearson correlation coefficient between two equal-length lists."""
    n = len(xs)
    if n < 2:
        return 0.0
    mean_x = statistics.mean(xs)
    mean_y = statistics.mean(ys)
    num = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    denom_x = math.sqrt(sum((x - mean_x) ** 2 for x in xs))
    denom_y = math.sqrt(sum((y - mean_y) ** 2 for y in ys))
    if denom_x == 0 or denom_y == 0:
        return 0.0
    return num / (denom_x * denom_y)


def calibration_correlation(
    calibration_data: list[dict[str, Any]],
    judge_fn: Any | None = None,
    allow_stub: bool = False,
) -> dict[str, float]:
    """Compute Pearson correlation between human ratings and judge scores.

    Returns a dict mapping metric name -> correlation coefficient.
    """
    human_accuracy: list[float] = []
    judge_accuracy: list[float] = []
    human_style: list[float] = []
    judge_style: list[float] = []
    human_naturalness: list[float] = []
    judge_naturalness: list[float] = []

    for entry in calibration_data:
        caption = entry.get("caption", "")
        style = entry.get("style", "formal")
        human = entry.get("human_ratings", {})

        # Use a minimal evidence stub for calibration
        evidence = {"subjects": [], "actions": [], "setting": ""}
        scores = run_judge_panel(
            evidence,
            caption,
            style,
            judge_fn=judge_fn,
            num_judges=1,
            allow_stub=allow_stub,
        )

        if "factual_accuracy" in human:
            human_accuracy.append(human["factual_accuracy"])
            judge_accuracy.append(scores.get("factual_accuracy", 0.0))

        if "style_match" in human:
            human_style.append(human["style_match"])
            judge_style.append(scores.get("style_strength", 0.0))

        if "naturalness" in human:
            human_naturalness.append(human["naturalness"])
            judge_naturalness.append(scores.get("naturalness", 0.0))

    return {
        "factual_accuracy_r": _pearson_r(human_accuracy, judge_accuracy),
        "style_match_r": _pearson_r(human_style, judge_style),
        "naturalness_r": _pearson_r(human_naturalness, judge_naturalness),
    }


# ---------------------------------------------------------------------------
# Full evaluation run
# ---------------------------------------------------------------------------

def run_evaluation(
    *,
    dry_run: bool = True,
    judge_fn: Any | None = None,
    caption_fn: Any | None = None,
) -> dict[str, Any]:
    """Execute the full evaluation suite and return aggregated metrics.

    Parameters
    ----------
    dry_run : bool
        If True, uses stub functions instead of real model calls.
    judge_fn : callable, optional
        ``(evidence, caption, style) -> str`` for judging.
    caption_fn : callable, optional
        ``(evidence, style) -> str`` for caption generation.
        If None in dry-run mode, uses placeholder captions.
    """
    eval_data = load_eval_dataset()
    calibration_data = load_calibration_subset()

    if not dry_run and caption_fn is None:
        raise ValueError("Live evaluation requires a caption_fn wired to the real pipeline.")
    if not dry_run and judge_fn is None:
        raise ValueError("Live evaluation requires a real judge_fn.")

    all_accuracy: list[float] = []
    all_style: list[float] = []
    all_unsupported_rate: list[float] = []
    all_latencies: list[float] = []
    results_for_fallback: list[dict[str, Any]] = []

    print(f"\n{'='*60}")
    print(f"  EVALUATION SUITE — {'DRY RUN' if dry_run else 'LIVE'}")
    print(f"{'='*60}\n")
    print(f"  Eval entries:       {len(eval_data)}")
    print(f"  Calibration entries: {len(calibration_data)}")
    print(f"  Styles:             {', '.join(ALL_STYLES)}")
    print()

    for entry in eval_data:
        video_id = entry["video_id"]
        ground_truth = entry["ground_truth"]
        styles = entry.get("expected_styles", ALL_STYLES)

        captions: dict[str, str] = {}
        for target_style in styles:
            start = time.monotonic()

            if caption_fn is not None:
                caption = caption_fn(ground_truth, target_style)
            elif dry_run:
                # Placeholder caption for dry run
                subjects = ", ".join(ground_truth.get("subjects", ["scene"]))
                setting = ground_truth.get("setting", "a location")
                caption = f"A view of {subjects} in {setting}."
            else:
                caption = f"[no caption_fn provided for {video_id}/{target_style}]"

            elapsed = time.monotonic() - start
            all_latencies.append(elapsed)
            captions[target_style] = caption

            # Judge the caption
            scores = run_judge_panel(
                ground_truth, caption, target_style,
                judge_fn=judge_fn, num_judges=3 if not dry_run else 1,
                allow_stub=dry_run,
            )

            all_accuracy.append(caption_accuracy(scores))
            all_style.append(style_match(scores))
            all_unsupported_rate.append(
                unsupported_noun_verb_rate(caption, ground_truth)
            )

            print(
                f"  [{video_id}] {target_style:20s}  "
                f"acc={scores.get('factual_accuracy', 0):.2f}  "
                f"style={scores.get('style_strength', 0):.2f}  "
                f"overall={scores.get('overall_score', 0):.2f}  "
                f"latency={elapsed:.3f}s"
            )

        results_for_fallback.append({"task_id": video_id, "captions": captions})

    # Aggregate metrics
    metrics = {
        "unsupported_noun_verb_rate": (
            statistics.mean(all_unsupported_rate) if all_unsupported_rate else 0.0
        ),
        "caption_accuracy": (
            statistics.mean(all_accuracy) if all_accuracy else 0.0
        ),
        "style_match": (
            statistics.mean(all_style) if all_style else 0.0
        ),
        "fallback_error_rate": fallback_error_rate(results_for_fallback),
        "p95_latency": p95_latency(all_latencies),
    }

    # Calibration correlation
    cal_corr = calibration_correlation(
        calibration_data,
        judge_fn=judge_fn,
        allow_stub=dry_run,
    )

    # Print report
    print(f"\n{'-'*60}")
    print("  AGGREGATE METRICS")
    print(f"{'-'*60}")
    print(f"  Unsupported noun/verb rate:  {metrics['unsupported_noun_verb_rate']:.4f}")
    print(f"  Caption accuracy (mean):     {metrics['caption_accuracy']:.4f}")
    print(f"  Style match (mean):          {metrics['style_match']:.4f}")
    print(f"  Fallback error rate:         {metrics['fallback_error_rate']:.4f}")
    print(f"  P95 latency:                 {metrics['p95_latency']:.4f}s")

    print(f"\n{'-'*60}")
    print("  CALIBRATION CORRELATION (Pearson r vs. human ratings)")
    print(f"{'-'*60}")
    for key, val in cal_corr.items():
        print(f"  {key:30s}  {val:+.4f}")

    print(f"\n{'='*60}")
    print("  EVALUATION COMPLETE")
    print(f"{'='*60}\n")

    return {
        "metrics": metrics,
        "calibration_correlation": cal_corr,
    }


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the Track 2 captioning evaluation suite.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Use stub functions instead of real model calls.",
    )
    parser.add_argument(
        "--eval-dataset",
        type=str,
        default=None,
        help="Path to eval_dataset.json (default: tests/eval_dataset.json).",
    )
    parser.add_argument(
        "--calibration-subset",
        type=str,
        default=None,
        help="Path to calibration_subset.json (default: tests/calibration_subset.json).",
    )
    args = parser.parse_args()

    # Override paths if provided
    if args.eval_dataset:
        global _EVAL_DATASET_PATH
        _EVAL_DATASET_PATH = Path(args.eval_dataset)
    if args.calibration_subset:
        global _CALIBRATION_SUBSET_PATH
        _CALIBRATION_SUBSET_PATH = Path(args.calibration_subset)

    results = run_evaluation(dry_run=args.dry_run)

    # Exit with non-zero if accuracy is below threshold
    accuracy = results["metrics"]["caption_accuracy"]
    if accuracy < 0.5:
        print(f"WARNING: Caption accuracy {accuracy:.4f} is below 0.5 threshold.", file=sys.stderr)
        raise SystemExit(1)

    raise SystemExit(0)


if __name__ == "__main__":
    main()

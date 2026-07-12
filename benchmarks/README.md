# Public Track 2 validation set

`public_validation_tasks.json` contains the eight retired judging examples published in the AMD Hackathon Judging FAQ and Self-Check Guide. The harness ignores `expected_visual_details`, so the file can be used directly as `TRACK2_INPUT`.

Score all 32 generated captions on five binary checks:

1. The main subject or scene is correct.
2. At least one additional concrete visual detail is correct.
3. No major literal detail is hallucinated.
4. The requested style is unmistakable.
5. The caption is concise and readable.

Compare prompt or sampling revisions against the same set. Treat factual correctness as the first tie-breaker, followed by style recognition and specificity.

Example local run:

```powershell
$env:TRACK2_INPUT = "benchmarks/public_validation_tasks.json"
$env:TRACK2_OUTPUT = "outputs/public_validation_results.json"
python -m track2_captioner.harness
```

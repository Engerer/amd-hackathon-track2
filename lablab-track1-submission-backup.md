# LabLab Track 1 Submission Backup

Saved from LabLab submission page on 2026-07-09.

## Basic Information

Title:

```text
Brocacho General-Purpose AI Agent-
```

Short description:

```text
A Dockerized Track 1 AI agent that answers hidden natural-language tasks across eight categories using Fireworks models through the judging proxy.
```

Long description:

```text
Brocacho General-Purpose AI Agent is a Dockerized Track 1 submission built for the AMD Developer Hackathon automated judging pipeline. The container reads tasks from /input/tasks.json, processes each prompt, and writes valid answers to /output/results.json before exiting. It supports the required capability areas: factual knowledge, mathematical reasoning, sentiment classification, summarization, named entity recognition, code debugging, logical reasoning, and code generation.

The agent reads FIREWORKS_API_KEY, FIREWORKS_BASE_URL, and ALLOWED_MODELS from the runtime environment, then routes all Fireworks API calls through the provided base URL. It avoids hardcoded model IDs, avoids cached answers, and only selects models from the allowed runtime list. The published image is public, linux/amd64 compatible, small, and tested with live Fireworks calls.
```

Categories:

```text
Developer Tools
Productivity
```

Event track:

```text
Hybrid Token-Efficient Routing Agent
```

Technologies used:

```text
AI/ML API
```

## Media

Video presentation:

```text
https://storage.googleapis.com/lablab-video-submissions/pn62mtt8xm33vu67q8z0673s/raw/submission-video-x-pn62mtt8xm33vu67q8z0673s-b5yuead808dzbpiuo35rbhmc_vv77nv8lfwucir2qfcc7yqt9.mp4
```

Slide presentation:

```text
https://storage.googleapis.com/lablab-static-eu/presentations/submissions/md88dj5r8aclo6inw76hvehy/md88dj5r8aclo6inw76hvehy-1783509187630_lhs9mtnur07gvq6k7vt1k79n.pdf
```

## Application

GitHub repository:

```text
https://github.com/AdamHarrisNyirop
```

Demo application URL:

```text
https://github.com/YongXianShen/Track-1-testing/pkgs/container/track-1-testing
```

Docker image:

```text
ghcr.io/yongxianshen/track-1-testing:latest
```

Additional information:

```text
Track 1 General-Purpose AI Agent submission. Public linux/amd64 Docker image that reads /input/tasks.json and writes /output/results.json. The agent reads FIREWORKS_API_KEY, FIREWORKS_BASE_URL, and ALLOWED_MODELS from the runtime environment, routes Fireworks calls through the judging proxy, and only uses allowed model IDs. Tested with Docker, live Fireworks calls, and the official practice-task format.
```

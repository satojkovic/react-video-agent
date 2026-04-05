# react-video-agent

[日本語版 README はこちら](README_ja.md)

A ReAct (Reasoning + Acting) agent that answers natural language questions about video files.

## Overview

The agent receives a video file and a question, then iteratively calls tools in a THINK → ACT → OBSERVE loop to derive an answer.

```
Question
  ↓
THINK → ACT → OBSERVE  (up to MAX_ITERATIONS times)
  ↓
Answer
```

**Available tools:**

| Tool | Role |
|------|------|
| `global_browse_tool` | Semantic search over the entire video for events and subjects |
| `clip_search_tool` | Semantic search for clips matching a text description |
| `frame_inspect_tool` | Visual analysis of frames in a given time range via VLM (enabled when `LITE_MODE=False`) |

## Background

This project is a restructured implementation based on [DeepVideoDiscovery](https://github.com/microsoft/DeepVideoDiscovery.git) by Microsoft.

Key changes from the original:
- Abstracted LLM backend to allow easy model swapping
- Separated video processing, tools, and agent core into distinct layers
- Reorganized module structure inspired by [react-from-scratch](https://github.com/satojkovic/react-from-scratch)

## Setup

**Requirements:** Python 3.12 or higher

```bash
pip install -e .
```

**Environment variables:** Create a `.env` file in the project root.

```env
# For OpenAI API
OPENAI_API_KEY=sk-...

# For Azure OpenAI, edit endpoint settings directly in src/config/settings.py
```

## Usage

```bash
python agent_run.py <path/to/video> "<question>"
```

**Example:**

```bash
python agent_run.py ./video.mp4 "Where was the main character at the end?"
```

On the first run, the following steps are performed automatically:

1. Decode video into frames → `video_database/<video_id>/frames/`
2. Generate captions → `video_database/<video_id>/captions/captions.json`
3. Build vector DB → `video_database/<video_id>/database.json`

Subsequent runs reuse the cache, so different questions on the same video respond much faster.

## Configuration

Key settings in `src/config/settings.py`:

| Setting | Default | Description |
|---------|---------|-------------|
| `LITE_MODE` | `True` | When `True`, uses subtitle text only (skips frame-level VLM analysis) |
| `MAX_ITERATIONS` | `3` | Maximum number of agent loop iterations |
| `AOAI_ORCHESTRATOR_LLM_MODEL_NAME` | `o3` | Model used for orchestration |
| `AOAI_TOOL_VLM_MODEL_NAME` | `gpt-4.1-mini` | Model used for tool-level VLM calls |
| `GLOBAL_BROWSE_TOPK` | `300` | Maximum number of clips returned by `global_browse_tool` |

## Code Structure

```
src/
├── config/
│   ├── logging.py      # Logging setup
│   └── settings.py     # Configuration values (model names, endpoints, etc.)
├── llm/
│   ├── base.py         # Abstract LLM interface (BaseLLM)
│   └── openai.py       # OpenAI / Azure OpenAI implementation (OpenAILLM)
├── react/
│   └── agent.py        # ReAct agent core (DVDCoreAgent)
├── tools/
│   ├── clip_search.py  # clip_search_tool
│   ├── frame_inspect.py # frame_inspect_tool
│   └── global_browse.py # global_browse_tool
├── utils/
│   ├── retry.py        # Exponential backoff retry decorator
│   └── schema.py       # JSON schema auto-generation for OpenAI Function Calling
└── video/
    ├── caption.py      # Frame captioning pipeline
    ├── database.py     # Vector DB construction and management
    └── utils.py        # Video download and frame extraction
```

## Extending the Agent

### Adding a new LLM backend

Subclass `BaseLLM` from `src/llm/base.py` and pass the instance to `DVDCoreAgent`.

```python
# src/llm/qwen.py (example)
from src.llm.base import BaseLLM

class QwenLLM(BaseLLM):
    def __init__(self, base_url: str, model_name: str):
        self.base_url = base_url
        self.model_name = model_name

    def call_with_tools(self, messages, tools=None, **kwargs) -> dict:
        # Call an OpenAI-compatible endpoint (e.g. vLLM / Ollama)
        ...

    def get_embeddings(self, text) -> list:
        ...
```

```python
# Usage in agent_run.py
from src.llm.qwen import QwenLLM

agent = DVDCoreAgent(
    video_db_path=video_db_path,
    video_caption_path=caption_file,
    max_iterations=15,
    llm=QwenLLM(base_url="http://localhost:8000", model_name="qwen2.5"),
)
```

### Adding a new tool

1. Create a new file in `src/tools/`

```python
# src/tools/my_tool.py (example)
from typing import Annotated as A
from src.utils.schema import doc as D

def my_tool(
    query: A[str, D("Search query")],
) -> str:
    """Tool description (used as a prompt sent to OpenAI)."""
    ...
    return result
```

2. Add the function to `self.tools` in `src/react/agent.py`

```python
from src.tools.my_tool import my_tool

self.tools = [frame_inspect_tool, clip_search_tool, global_browse_tool, my_tool, finish]
```

The agent will automatically recognize and call the new tool.

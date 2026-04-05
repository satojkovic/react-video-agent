# react-video-agent

動画に対して自然言語で質問すると、ReAct（Reasoning + Acting）ループを通じて回答を返すエージェント。

## 概要

動画ファイルを入力として受け取り、ユーザーの質問に対してエージェントが以下のツールを繰り返し呼び出しながら回答を導き出す。

```
質問
  ↓
THINK → ACT → OBSERVE（最大 MAX_ITERATIONS 回）
  ↓
回答
```

**利用可能なツール：**

| ツール | 役割 |
|--------|------|
| `global_browse_tool` | 動画全体のイベント・登場人物をセマンティック検索 |
| `clip_search_tool` | テキスト説明でクリップをセマンティック検索 |
| `frame_inspect_tool` | 指定した時間範囲のフレームを VLM で視覚的に解析（`LITE_MODE=False` のとき有効） |

## 背景

[DeepVideoDiscovery](https://github.com/microsoft/DeepVideoDiscovery.git)（Microsoft）のコードをベースに、モジュール構造を再設計したプロジェクト。

主な変更点：
- LLM バックエンドを抽象化し、モデルの差し替えを容易化
- 動画処理・ツール・エージェントコアをレイヤーごとに分離
- ReAct チュートリアル（[react-from-scratch](https://github.com/satojkovic/react-from-scratch)）の構造を参考に整理

## セットアップ

**必要な環境：** Python 3.12 以上

```bash
# 依存関係のインストール
pip install -e .
```

**環境変数：** プロジェクトルートに `.env` ファイルを作成する。

```env
# OpenAI API を使う場合
OPENAI_API_KEY=sk-...

# Azure OpenAI を使う場合は src/config/settings.py のエンドポイント設定を直接編集
```

## 使い方

```bash
python agent_run.py <動画ファイルのパス> "<質問>"
```

**例：**

```bash
python agent_run.py ./video.mp4 "主人公は最後どこにいましたか？"
```

初回実行時は自動的に以下を処理する。

1. 動画をフレームに分解 → `video_database/<動画ID>/frames/`
2. キャプション生成 → `video_database/<動画ID>/captions/captions.json`
3. ベクトルDB構築 → `video_database/<動画ID>/database.json`

2回目以降はキャッシュが使われるため、同じ動画への別の質問は高速に応答する。

## 主な設定項目

`src/config/settings.py` で動作を調整できる。

| 設定 | デフォルト | 説明 |
|------|-----------|------|
| `LITE_MODE` | `True` | `True` のとき字幕テキストのみ使用（フレーム画像の VLM 解析をスキップ） |
| `MAX_ITERATIONS` | `3` | エージェントのループ最大回数 |
| `AOAI_ORCHESTRATOR_LLM_MODEL_NAME` | `o3` | オーケストレーター LLM のモデル名 |
| `AOAI_TOOL_VLM_MODEL_NAME` | `gpt-4.1-mini` | ツール VLM のモデル名 |
| `GLOBAL_BROWSE_TOPK` | `300` | `global_browse_tool` が返す最大クリップ数 |

## コード構成

```
src/
├── config/
│   ├── logging.py      # ロギング設定
│   └── settings.py     # 設定値（モデル名・エンドポイント等）
├── llm/
│   ├── base.py         # LLM 抽象インターフェース（BaseLLM）
│   └── openai.py       # OpenAI / Azure OpenAI 実装（OpenAILLM）
├── react/
│   └── agent.py        # ReAct エージェントコア（DVDCoreAgent）
├── tools/
│   ├── clip_search.py  # clip_search_tool
│   ├── frame_inspect.py # frame_inspect_tool
│   └── global_browse.py # global_browse_tool
├── utils/
│   ├── retry.py        # 指数バックオフリトライデコレータ
│   └── schema.py       # OpenAI Function Calling 用 JSON スキーマ自動生成
└── video/
    ├── caption.py      # フレームキャプション生成パイプライン
    ├── database.py     # ベクトル DB の構築・管理
    └── utils.py        # 動画のダウンロード・フレーム抽出
```

## 拡張方法

### 新しい LLM バックエンドの追加

`src/llm/base.py` の `BaseLLM` を継承して実装し、`DVDCoreAgent` に渡すだけでよい。

```python
# src/llm/qwen.py（例）
from src.llm.base import BaseLLM

class QwenLLM(BaseLLM):
    def __init__(self, base_url: str, model_name: str):
        self.base_url = base_url
        self.model_name = model_name

    def call_with_tools(self, messages, tools=None, **kwargs) -> dict:
        # vLLM / Ollama の OpenAI 互換エンドポイントを呼ぶ
        ...

    def get_embeddings(self, text) -> list:
        ...
```

```python
# agent_run.py での使い方
from src.llm.qwen import QwenLLM

agent = DVDCoreAgent(
    video_db_path=video_db_path,
    video_caption_path=caption_file,
    max_iterations=15,
    llm=QwenLLM(base_url="http://localhost:8000", model_name="qwen2.5"),
)
```

### 新しいツールの追加

1. `src/tools/` に新しいファイルを作成する

```python
# src/tools/my_tool.py（例）
from typing import Annotated as A
from src.utils.schema import doc as D

def my_tool(
    query: A[str, D("検索クエリ")],
) -> str:
    """ツールの説明（OpenAI に渡されるプロンプトになる）"""
    ...
    return result
```

2. `src/react/agent.py` の `self.tools` リストに追加する

```python
from src.tools.my_tool import my_tool

self.tools = [frame_inspect_tool, clip_search_tool, global_browse_tool, my_tool, finish]
```

以上で、エージェントが自動的に新しいツールを認識して呼び出せるようになる。

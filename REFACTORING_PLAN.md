# リファクタリング計画

## 背景

`react-video-agent` は `DeepVideoDiscovery` のコードを構造化して配置したプロジェクト。
`react-from-scratch`（ReActエージェントのチュートリアル）の整理された構造を参考に、コードを整理する。

---

## 現状のフォルダ構成

```
src/
├── config/
│   └── __init__.py         # ほぼ空
├── react/
│   ├── __init__.py
│   ├── agent.py            # エージェント本体 + LLM呼び出し
│   ├── config.py           # 設定（react/配下に混在）
│   ├── frame_caption.py    # 動画キャプション処理（react/配下に混在）
│   ├── func_call_shema.py  # スキーマ生成ユーティリティ
│   ├── utils.py            # LLM呼び出し・リトライ・埋め込みが混在
│   └── video_utils.py      # 動画処理ユーティリティ
└── tools/
    ├── build_database.py   # 複数ツール + DB管理が1ファイルに混在
    └── finish.py
```

---

## 参考: react-from-scratch のフォルダ構成

```
src/
├── config/
│   ├── logging.py          # ロギング設定（独立）
│   └── setup.py            # Config singleton（独立）
├── llm/
│   └── gemini.py           # LLMラッパーが独立したモジュール
├── react/
│   └── agent.py            # エージェントコアのみ
├── tools/
│   ├── serp.py             # ツールが個別ファイルに分離
│   └── wiki.py
└── utils/
    └── io.py               # I/Oユーティリティが独立
```

---

## 現状の問題点

| 問題 | 現状 | scratchの対応 |
|------|------|--------------|
| LLM呼び出しが分離されていない | `react/utils.py` にOpenAI呼び出し・リトライ・埋め込みが混在 | `src/llm/` として独立モジュール |
| 設定が `react/` 配下に置かれている | `react/config.py` | `src/config/settings.py` として独立 |
| ロギングが未整備 | 専用モジュールなし | `src/config/logging.py` として独立 |
| ツールが1ファイルに集中 | `build_database.py` に3ツール + DB管理が混在 | ツールごとに個別ファイル |
| 動画処理が `react/` 配下に混在 | `frame_caption.py`, `video_utils.py` が `react/` 配下 | 独立した層に分離 |
| I/Oユーティリティがない | ファイル読み書きが各所に散在 | `src/utils/io.py` として独立（今回は対象外） |

---

## 目標のフォルダ構成

```
src/
├── config/
│   ├── logging.py          # 新規追加
│   └── settings.py         # react/config.py を移動・リネーム
├── llm/
│   ├── base.py             # 抽象基底クラス（共通インターフェース定義）★重要
│   └── openai.py           # OpenAI/Azure実装
├── react/
│   └── agent.py            # エージェントコアのみ（BaseLLMに依存）
├── tools/
│   ├── frame_inspect.py    # build_database.py から分離
│   ├── clip_search.py      # build_database.py から分離
│   └── global_browse.py    # build_database.py から分離
├── video/
│   ├── caption.py          # react/frame_caption.py を移動
│   ├── database.py         # build_database.py のDB管理部分を移動
│   └── utils.py            # react/video_utils.py を移動
└── utils/
    ├── retry.py            # react/utils.py のリトライ部分を分離
    └── schema.py           # react/func_call_shema.py を移動
```

> **注:** `tools/finish.py` は空ファイルのため削除。`utils/io.py` は今回のスコープ外。

### llm/base.py のイメージ

```python
from abc import ABC, abstractmethod

class BaseLLM(ABC):
    @abstractmethod
    def call_with_tools(self, messages, tools, **kwargs) -> dict:
        ...

    @abstractmethod
    def get_embeddings(self, text) -> list:
        ...
```

`agent.py` は `BaseLLM` だけに依存し、どの実装クラスを使うかは
`settings.py` や起動時の引数で切り替える設計にする。
これにより、Qwen や vLLM 等のローカルLLMへの切り替えが容易になる。

---

## 依存関係の全体像（現状）

```
agent_run.py
  └─ react/config.py          ← 全モジュールが参照
  └─ react/agent.py
       └─ react/utils.py      ← LLM呼び出しの核心
       └─ react/func_call_shema.py
       └─ tools/build_database.py
            └─ react/utils.py
            └─ react/func_call_shema.py
  └─ react/frame_caption.py
       └─ react/utils.py
  └─ react/video_utils.py
```

---

## 作業単位と推奨順序

### Unit 1 — config分離
**`react/config.py` → `config/settings.py`**

- 他モジュールへの依存がゼロなので最も安全
- import文を一括更新（`src.react.config` → `src.config.settings`）
- **完了後に `agent_run.py` が動くことを確認**

### Unit 2 — ユーティリティ分解
**`react/utils.py` と `react/func_call_shema.py` を分解**

```
react/utils.py       → utils/retry.py    (retry_with_exponential_backoff)
                     → utils/schema.py   (func_call_shema.py と統合)
                     → llm/base.py       (抽象インターフェース 新規作成)
                     → llm/openai.py     (call_openai_model_with_tools
                                          AzureOpenAIEmbeddingService
                                          extract_answer)
```

Unit 1完了後に着手。**この作業が最も影響範囲が広く、後続すべての単位の基盤になる。**

### Unit 3 — video層の分離
**`react/frame_caption.py`, `react/video_utils.py` → `video/`**

```
react/video_utils.py   → video/utils.py
react/frame_caption.py → video/caption.py
```

Unit 2完了後に着手。`agent.py` への変更なし。

### Unit 4 — tools分割
**`tools/build_database.py` を個別ファイルに分割**

```
tools/build_database.py → tools/frame_inspect.py
                        → tools/clip_search.py
                        → tools/global_browse.py
                        → video/database.py     (init_single_video_db等)
```

Unit 3と並行して進められる。

### Unit 5 — agent.pyのLLM抽象化
**`agent.py` が `BaseLLM` インターフェースを使うように変更**

Unit 2〜4がすべて完了してから着手。変更範囲は `agent.py` 内のみ。

### Unit 6 — logging整備（任意）
**`config/logging.py` の新規追加**

どの単位とも独立しているので最後に追加。

---

## 作業単位サマリー

| # | 作業 | 前提 | リスク | 確認方法 |
|---|------|------|--------|----------|
| 1 | config分離 | なし | 低 | agent_run.py 実行 |
| 2 | utils/llm分解 | Unit 1 | 中〜高 | agent_run.py 実行 |
| 3 | video層分離 | Unit 2 | 低 | agent_run.py 実行 |
| 4 | tools分割 | Unit 2 | 中 | agent_run.py 実行 |
| 5 | agent LLM抽象化 | Unit 2-4 | 中 | agent_run.py 実行 |
| 6 | logging整備 | なし | 低 | ログ出力確認 |

各単位完了後に `agent_run.py` を実際に動かして動作確認しながら進める。

---

## 完成後のフォルダ構成（実績）

```
src/
├── config/
│   ├── __init__.py
│   ├── logging.py          ✅ Unit 6
│   └── settings.py         ✅ Unit 1
├── llm/
│   ├── __init__.py
│   ├── base.py             ✅ Unit 2
│   └── openai.py           ✅ Unit 2, 5
├── react/
│   ├── __init__.py
│   └── agent.py            ✅ Unit 5
├── tools/
│   ├── clip_search.py      ✅ Unit 4
│   ├── frame_inspect.py    ✅ Unit 4
│   └── global_browse.py    ✅ Unit 4
├── utils/
│   ├── __init__.py
│   ├── retry.py            ✅ Unit 2
│   └── schema.py           ✅ Unit 2
└── video/
    ├── __init__.py
    ├── caption.py          ✅ Unit 3
    ├── database.py         ✅ Unit 4
    └── utils.py            ✅ Unit 3
```

# Python SDK Reference

The canonical reference for the `privyscope` core SDK. The core is
language-neutral — install at least one language pack (`privyscope-ko`,
`privyscope-en`, `privyscope-zh-hans`, …) so engines have data + weights to load.

```python
from privyscope import Privyscope
```

## Public surface

Everything importable from the top-level package:

| Symbol | Kind | Purpose |
|---|---|---|
| `Privyscope` | class | Single-language engine + constructors |
| `RedactionResult` | dataclass | Result of one `redact` call |
| `DetectedSpan` | dataclass | One detected PII span |
| `LanguagePlugin` | dataclass | Language-pack registration record |
| `installed_languages()` | func | `dict[code -> LanguagePlugin]` of discovered packs |
| `SCHEMA_VERSION` | int | Output-schema version (bump = breaking change) |
| `__version__` | str | Core package version |

Anything under a `_`-prefixed module (`privyscope._core`, `privyscope._api`, …) is
private and may change without notice — depend only on the surface above.

## Loading an engine

### `Privyscope.from_pretrained(operating_point="balanced", *, lang=None, repo_id=None, revision=None, cache_dir=None, regex_rules=None, providers=None, regex_only=False) -> Privyscope`

Loads a **single-language** engine, downloading ONNX weights from Hugging Face
Hub on first use (cached at `~/.cache/privyscope/`).

| Parameter | Default | Purpose |
|---|---|---|
| `operating_point` | `"balanced"` | `high_recall` / `balanced` / `high_precision`, or 6 raw biases |
| `lang` | `None` | language code; omit to use the sole installed pack (raises if several) |
| `repo_id` | plugin default | HF Hub weights repo (overrides the plugin's `default_repo`) |
| `revision` | `None` | pin a specific weight version (FR-2.8) |
| `cache_dir` | `None` | local bundle dir → **fully offline**, no network at inference |
| `regex_rules` | packaged | override the regex ruleset path |
| `providers` | `["CPUExecutionProvider"]` | ONNX execution providers |
| `regex_only` | `False` | skip the NER stage (no weights required) |

```python
engine = Privyscope.from_pretrained(lang="ko")
engine = Privyscope.from_pretrained(lang="en", operating_point="high_recall")
engine = Privyscope.from_pretrained(lang="ko", cache_dir="/path/to/bundle")  # offline
```

When exactly one language pack is installed, `lang` may be omitted:

```python
engine = Privyscope.from_pretrained()    # the sole installed language
```

### `Privyscope.regex_only(*, lang=None, regex_rules=None) -> Privyscope`

Stage-1-only engine — detects structurally obvious PII (PHONE, EMAIL, ID_NUM,
BANK, SECRET) with **no model weights**. Handy offline, in CI, and for fast
structural-only passes.

```python
engine = Privyscope.regex_only(lang="en")
```

### `Privyscope.auto(operating_point="balanced", *, cache_dir=None, providers=None, regex_only=False) -> AutoPrivyscope`

Returns a **multi-language dispatcher**. Each text is routed to its language by a
Unicode-script heuristic (restricted to installed packs); per-language engines
are built lazily and cached. Use this when several packs are installed and you
don't want to pick a language per call.

```python
auto = Privyscope.auto()
auto.redact("John Smith 555-123-4567")    # → English engine
auto.redact("홍길동 010-1234-5678")        # → Korean engine
auto.redact("张伟 13812345678")            # → Simplified-Chinese engine
```

`AutoPrivyscope` mirrors the `Privyscope` inference surface (`redact`,
`batch_redact`, `has_ner`), plus:

| Member | Purpose |
|---|---|
| `available_languages` | installed codes this dispatcher can route to |
| `route(text) -> str` | the language code chosen for `text` (no redaction) |
| `preload(langs=None) -> self` | eagerly build/cache engines (see below) |

## Redacting

### `engine.redact(text, entity_types=None, output_mode="typed", operating_point=None) -> RedactionResult`

```python
r = engine.redact("홍길동의 전화번호는 010-1234-5678")
r.redacted_text      # "<PER>의 전화번호는 <PHONE>"
r.masked_text        # alias of redacted_text
r.detected_spans     # [DetectedSpan(label="PER", start=0, end=3, ...), ...]
r.summary            # {"span_count": 2, "by_label": {...}, ...}
r.to_dict()          # full JSON structure (see OUTPUT_SCHEMAS.md)

engine.redact("...", entity_types=["PER", "PHONE"])   # filter labels
engine.redact("...", output_mode="redacted")          # generic <REDACTED>
engine.redact("...", operating_point="high_recall")   # per-call override
```

### `engine.batch_redact(texts, ...) -> list[RedactionResult]`

```python
results = engine.batch_redact(["sentence 1", "sentence 2", "sentence 3"])
```

### `engine.has_ner -> bool`

`True` when the NER stage is loaded; `False` in regex-only mode.

## Result objects

`RedactionResult` fields: `text`, `detected_spans`, `redacted_text`,
`output_mode`, `decoded_mismatch`, `schema_version`, plus the `masked_text` and
`summary` convenience properties. `to_dict()` produces JSON-ready output keyed by
`schema_version`, `summary`, `text`, `detected_spans`, `redacted_text`.

Each `DetectedSpan` has `label`, `start` (inclusive), `end` (exclusive), `text`,
`placeholder`. Full schema: [OUTPUT_SCHEMAS](OUTPUT_SCHEMAS.md).

## Introspection

```python
from privyscope import installed_languages, __version__, SCHEMA_VERSION

installed_languages()      # {"ko": LanguagePlugin(...), "en": ...}
__version__                # "0.1.0"
SCHEMA_VERSION             # 1  — output-schema version
```

Each `LanguagePlugin` exposes `code`, `display_name`, `default_repo`, `package`,
`scripts`, `default_base_model`, and `regex_rules_path()` / `entity_config_path()`.

## Concurrency & long-running services

The SDK is designed to sit inside a long-running process (a worker pool, a web
server, the future REST API):

- **Build once, reuse.** Constructing an engine downloads weights and creates an
  ONNX Runtime session — do it at startup, not per request. `AutoPrivyscope`
  already caches one engine per language.
- **Thread-safe engine cache.** `AutoPrivyscope` guards per-language construction
  with a lock (double-checked), so concurrent first-hits for the same language
  share a single engine and a single weights download. Once built, ONNX Runtime
  sessions are safe for concurrent inference.
- **Warm up with `preload()`.** Build engines eagerly at startup to fail fast on
  missing weights and avoid a cold-start latency spike on the first request:

  ```python
  ENGINE = Privyscope.auto().preload()          # all installed languages
  ENGINE = Privyscope.auto().preload(["en"])    # just English
  ```

- **JSON-ready output.** `RedactionResult.to_dict()` is the natural HTTP response
  body; `SCHEMA_VERSION` gives you API versioning for free.

### Sketch: wrapping the SDK in a REST endpoint

```python
# Illustrative (FastAPI). The engine is CPU-bound, so the async route hands work
# to a thread; the thread-safe cache + preload above make this safe.
from fastapi import FastAPI
from pydantic import BaseModel
from starlette.concurrency import run_in_threadpool
from privyscope import Privyscope, __version__, installed_languages

app = FastAPI()
ENGINE = Privyscope.auto().preload()           # warm at startup

class RedactIn(BaseModel):
    text: str
    output_mode: str = "typed"

@app.get("/health")
def health():
    return {"version": __version__, "languages": sorted(installed_languages())}

@app.post("/v1/redact")
async def redact(body: RedactIn):
    result = await run_in_threadpool(ENGINE.redact, body.text, None, body.output_mode)
    return result.to_dict()
```

## Proxies & offline

- `HTTPS_PROXY` is honoured during weight download (FR-2.8).
- Pass `cache_dir` pointing at a pre-downloaded bundle for **zero network calls**
  at inference time (SRS §8.3).
- Full walkthrough (download, air-gapped transfer, `HF_HUB_OFFLINE`, integrity
  check): see [Offline Usage](OFFLINE.md) where shipped with a language pack.

# privyscope

**Multilingual PII detection & masking engine** — the language-neutral core of the
[privyscope](https://github.com/zafrem) project.

This package contains the engine (a two-stage hybrid pipeline: a regex filter plus
an optional ONNX BIOES/Viterbi NER stage), the `privyscope` console command, and the
plugin machinery that discovers installed **language packs**. It ships **no language
data** on its own — install at least one language pack to do real work:

```bash
pip install privyscope-ko      # Korean   (pulls in this core automatically)
pip install privyscope-en      # English
```

Installing a language pack pulls in this core as a dependency, so you normally never
`pip install privyscope` directly.

## Why privyscope?

Compared to traditional NER systems from 3-4 years ago, `privyscope` delivers significantly higher precision and robustness by employing several modern architectural advancements:

1. **BIOES Tagging:** Unlike the traditional BIO scheme, we use the BIOES (Begin, Inside, Outside, End, Single) scheme. This allows the model to explicitly learn span boundaries and single-token entities, which is critical for CJK languages where particles often follow PII entities without spaces.
2. **Constrained Viterbi Decoding:** Instead of making independent per-token decisions, we use a Viterbi decoder with learned transition biases. This ensures that the output sequence is always grammatically valid (e.g., an "Inside" tag can never follow an "Outside" tag), preventing fragmented or "broken" masking results.
3. **Domain-Adaptive MLM:** Before fine-tuning on PII extraction, we perform additional Masked Language Modeling (MLM) on a large-scale synthetic PII corpus. This "domain adaptation" phase allows the model to internalize the structural characteristics of PII (like API keys, ID numbers, and complex addresses) that are rarely seen in general-purpose datasets like Wikipedia.
4. **LLM-Augmented Synthesis:** We leverage LLMs and Faker to generate hundreds of thousands of high-fidelity synthetic PII sentences across various registers (formal, conversational, etc.). This massive increase in training variety ensures the model remains robust against novel contexts and diverse writing styles.

## One command, any combination of languages

The `privyscope` command lives **only** here in the core, so co-installed language
packs never collide over it (each pack only adds its own `privyscope_<lang>` data
package).

```bash
# one language installed → it's used automatically
privyscope redact "홍길동 010-1234-5678"

# several installed → auto-detected per text, or forced with --lang
privyscope redact "John Smith 555-123-4567"          # → English
privyscope redact --lang ko "Call 010-1234-5678"     # force Korean
privyscope --version                                  # lists installed languages
```

## Python API

```python
from privyscope import Privyscope

# one language (explicit, or the sole one installed)
engine = Privyscope.from_pretrained(lang="ko")
engine.redact("홍길동의 전화번호는 010-1234-5678").masked_text   # "<PER>의 전화번호는 <PHONE>"

# multiple languages → route each text automatically
auto = Privyscope.auto()
auto.redact("John Smith 555-123-4567")    # English engine
auto.redact("홍길동 010-1234-5678")        # Korean engine
```

See [`docs/`](../docs) (in the repo) for the CLI reference, offline usage, the
output schema, fine-tuning, and the **multilingual use-case guide**.

## Writing a language pack

A language pack is tiny: a package that ships two YAML files
(`regex_rules.yaml`, `entity_config.yaml`) and registers a `LanguagePlugin` under
the `privyscope.languages` entry-point group.

```python
# privyscope_xx/__init__.py
from privyscope import Privyscope, LanguagePlugin, __version__

LANG = LanguagePlugin(
    code="xx", display_name="Example", default_repo="org/privyscope-xx",
    package="privyscope_xx", scripts=("Latin",), default_base_model="roberta-base",
)
```

```toml
# pyproject.toml
[project]
dependencies = ["privyscope>=0.1.0"]

[project.entry-points."privyscope.languages"]
xx = "privyscope_xx:LANG"
```

## License

Apache-2.0. See [LICENSE](LICENSE).

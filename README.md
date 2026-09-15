# Paper Summarizer

Section-wise summarizer for research papers, run one stage per notebook cell. Point it at
an arXiv id and it returns a one-page HTML digest: the reading plan, the gap the paper
targets, the method as a diagram, the paper's own key figures, and the results as charts
that show whether the proposed method actually won.

```bash
papersum 1706.03762 --model qwen3:8b
# -> papers/reports/1706.03762v7_report.md
# -> papers/reports/1706.03762v7_report.html   <- open this one
```

## Input sources

| You have | Pass | What happens |
|---|---|---|
| an arXiv id | `papersum 2405.04517` | metadata from the arXiv API, text from the HTML rendering |
| an arXiv URL | `papersum https://arxiv.org/abs/2405.04517` | same |
| only the title | `papersum --arxiv "attention is all you need"` | title search, then the best match |
| a local file | `papersum papers/input/x.pdf` | PDF / Markdown / EPUB loader |

**Prefer arXiv when the paper is there.** arXiv publishes an experimental LaTeXML
rendering at `/html/<id>`, and it carries everything the PDF loader otherwise has to infer
from font sizes: real `<h2>`/`<h3>` headings in document order, real table markup, and
figures with captions and image URLs. Papers without an HTML rendering fall back to
downloading the PDF automatically, so either way the command is the same.

Real figures are what make the report visual. The `key_figures` stage shows the model
every caption and asks which few a reader actually needs, plus one line on why each earns
its place; the report then reproduces them inline. xLSTM ships 26 figures — the digest
keeps four.

## The stages

```
1 load LLM → 2 load paper + classify outline → 3 retrieve repo/web + reading plan
→ 4 background & gap → 5 motivation & method → 6 workflow diagram + key figures
→ 7 results (tiles + charts + tables) → 8 limitations & next gap → 9 conclusion
→ export report.md + report.html
```

Each stage is one `Paper` method and one notebook cell, so any of them can be re-run after
changing the model or fixing a section. `paper.run_all()` does the lot.

## File tree

```
paper-summarizer/
├── src/papersum/            the library (installed with `pip install -e .`)
│   ├── arxiv.py             arXiv API: resolve by id/URL/title, fetch the HTML rendering
│   ├── loaders.py           arXiv HTML / PDF / Markdown / EPUB → text + tables + figures
│   ├── sections.py          heading → canonical section (regex + LLM outline pass), links
│   ├── llm.py               ollama / anthropic / openai-compatible client, .json() helper
│   ├── retrieve.py          GitHub tree + model/config files, HF config.json, web search
│   ├── viz.py               Markdown / Mermaid / tables / bar charts inline (notebook)
│   ├── report_html.py       the one-page HTML report
│   ├── pipeline.py          Paper class — one method per stage
│   └── cli.py               `papersum <id|file>` entry point
├── notebooks/summarize.ipynb   the 9-step notebook (VS Code / JupyterLab)
├── scripts/run.py           CLI without installing the package
├── papers/
│   ├── input/               drop PDFs / .md / .epub here (or use the upload widget)
│   └── reports/             generated <name>_report.md and <name>_report.html
├── docker/{Dockerfile,docker-compose.yml}
├── tests/test_pipeline.py   offline tests (fake PDF + stub LLM, no network)
├── pyproject.toml           package + deps
├── requirements.txt         same deps, for plain pip
├── Makefile                 setup / lab / docker / test shortcuts
├── .env.example             copy to .env: OLLAMA_HOST, API keys, GITHUB_TOKEN
└── .gitignore
```

## Run locally

```bash
python -m venv .venv && .venv/Scripts/pip install -e ".[notebook]"   # Windows
# make setup                                                          # macOS / Linux
ollama pull qwen3:8b
```

Then either open `notebooks/summarize.ipynb` and pick the `.venv` kernel, or use the CLI:

```bash
papersum 1706.03762 --backend ollama --model qwen3:8b
```

| flag | effect |
|---|---|
| `--arxiv` | treat the argument as an arXiv query (id, URL, or title) |
| `--backend` | `ollama` (default), `anthropic`, `openai` |
| `--model` | model name for the backend |
| `--offline` | skip GitHub / HuggingFace / web retrieval |
| `--no-html` | write only the Markdown report |
| `--out` | report path (default `papers/reports/<name>_report.md`) |

Without installing the package: `python scripts/run.py 1706.03762`.

## Run with Docker

```bash
cp .env.example .env      # fill in keys if using anthropic/openai
make docker               # = docker compose -f docker/docker-compose.yml up --build
```

Open http://localhost:8888/lab/tree/notebooks/summarize.ipynb. Ollama stays on the host
(`ollama serve`); the container reaches it via `host.docker.internal`. `papers/` is mounted.

## LLM backends

| backend | model examples | needs |
|---|---|---|
| `ollama` | `qwen3:8b`, `llama3.1:8b`, `qwen2.5:14b` | Ollama running locally |
| `anthropic` | `claude-sonnet-4-6` | `ANTHROPIC_API_KEY` |
| `openai` | `gpt-4o-mini`, any OpenAI-compatible | `OPENAI_API_KEY`, opt. `OPENAI_BASE_URL` |

8B handles the prose stages; 14B+ is noticeably better at the JSON stages (results
extraction, layer spec, outline classification). For reference, `qwen3:8b` on an 8GB
RTX 4060 runs a full paper in roughly three minutes.

`GITHUB_TOKEN` raises the GitHub API limit (60/h → 5000/h) for repo retrieval.

## Notes

- **Reasoning models are asked not to think.** `qwen3` and `deepseek-r1` otherwise spend
  the whole `num_predict` budget on a `<think>` trace, get truncated before the answer, and
  every stage returns an empty string. Pass `LLM(..., think=True)` to re-enable.
- **Section classification.** From a PDF, headings are found by font size and judged one at
  a time, so subsections like `Optimizer` or `Positional Encoding` fall into `other` and
  never reach the method/experiments stages. `paper.classify_outline()` sends the whole
  outline to the model in one call and fixes most of it; it runs first in `run_all()`. You
  can still override by hand: `paper.sections[i].kind = "method"`. arXiv HTML avoids the
  problem at the source.
- **Charts use the emphasis form** — the paper's own method in the accent hue, baselines in
  gray, so the comparison reads before any number does. This relies on the model tagging
  its rows with `proposed` during results extraction; when it doesn't, the bars fall back
  to a single neutral hue.
- **Figures and Mermaid need the network.** Figures are referenced from arxiv.org rather
  than inlined, and Mermaid renders from a CDN, so offline the diagram shows as source text
  and figures won't load. Everything else in the report is self-contained.
- **Both formats are written by default.** The Markdown is for diffing and grepping; the
  HTML is the one to read.

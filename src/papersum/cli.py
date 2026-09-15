"""CLI:  papersum papers/input/x.pdf --backend ollama --model llama3.1:8b [--offline]"""
import argparse
import sys
from pathlib import Path

from . import LLM, Paper

# Status lines and report text contain non-ASCII (checkmarks, em dashes, math symbols);
# on consoles with a legacy codepage (cp932/cp1252) printing them raises UnicodeEncodeError.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="papersum")
    ap.add_argument("file", help="local .pdf/.md/.epub, or an arXiv id or arxiv.org URL")
    ap.add_argument("--arxiv", action="store_true",
                    help="treat the argument as an arXiv query (id, URL, or paper title)")
    ap.add_argument("--backend", default="ollama", choices=["ollama", "anthropic", "openai"])
    ap.add_argument("--model", default="llama3.1:8b")
    ap.add_argument("--offline", action="store_true", help="skip repo/web retrieval")
    ap.add_argument("--out", default=None, help="report path (default: papers/reports/<name>_report.md)")
    ap.add_argument("--no-html", action="store_true", help="skip the single-page HTML report")
    a = ap.parse_args(argv)

    llm = LLM(a.backend, a.model)
    print(llm.check())
    paper = Paper(llm)
    if a.arxiv:
        from .loaders import load_arxiv
        paper.load_doc(load_arxiv(a.file))
    else:
        paper.load(a.file)
    stem = paper.doc.meta.get("id", "").replace("/", "_") or Path(a.file).stem
    paper.run_all(use_web=not a.offline)
    out = Path(a.out or Path("papers/reports") / f"{stem}_report.md")
    out.parent.mkdir(parents=True, exist_ok=True)
    paper.export(out)
    print(f"markdown: {out}")
    if not a.no_html:
        print(f"html:     {paper.export_html(out.with_suffix('.html'))}")


if __name__ == "__main__":
    main()

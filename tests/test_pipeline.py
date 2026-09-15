"""Offline smoke test: fake PDF + stub LLM through every stage.  `pytest -q tests`"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import pymupdf  # noqa: E402

from papersum import LLM, Paper, load_document, split_sections  # noqa: E402

SECTIONS = [
    ("Abstract", "We propose FooNet for Japanese vowel devoicing detection in L2 speech."),
    ("1 Introduction", "Prior work uses HMMs. No method handles L2 speech; that is the gap."),
    ("3 Method", "Mel spectrograms feed a 6-layer Transformer (d_model=512) with a CTC head."),
    ("5 Results", "FooNet reaches 12.3% WER vs 15.1% for CTC-base and 20.4% for HMM."),
    ("6 Limitations", "Read speech only; spontaneous speech is future work."),
    ("7 Conclusion", "FooNet improves devoicing detection for L2 learners."),
]


class StubLLM(LLM):
    def __init__(self):
        super().__init__("ollama", "stub")

    def __call__(self, prompt, system=""):
        self.calls += 1
        if "Mermaid" in prompt:
            return "flowchart LR\n A[Mel] --> B[Transformer] --> C[CTC]"
        if "metrics" in prompt:
            return json.dumps({"summary": "FooNet best.", "metrics": [
                {"group": "WER", "label": "FooNet", "value": 12.3}, {"group": "WER", "label": "HMM", "value": 20.4}]})
        return "stub"


def make_pdf(path):
    doc = pymupdf.open()
    page = doc.new_page()
    y = 40
    page.insert_text((50, y), "FooNet: Devoicing Detection for L2 Japanese", fontsize=16); y += 30
    for h, b in SECTIONS:
        page.insert_text((50, y), h, fontsize=13, fontname="hebo"); y += 18
        page.insert_text((50, y), b, fontsize=10); y += 22
    doc.save(path)


def test_sections_and_stages(tmp_path):
    pdf = tmp_path / "fake.pdf"
    make_pdf(pdf)
    doc = load_document(pdf)
    kinds = {s.kind for s in split_sections(doc.text)}
    assert {"abstract", "introduction", "method", "results", "limitations", "conclusion"} <= kinds

    p = Paper(StubLLM()).load(pdf).run_all(use_web=False)
    out = p.export(tmp_path / "r.md")
    assert "Conclusion" in out.read_text(encoding="utf-8") and p.llm.calls >= 7


def test_url_wrapped_across_lines_is_rejoined():
    """A URL broken after '/' by PDF line wrapping must survive text normalisation."""
    from papersum.loaders import _dehyphenate
    from papersum.sections import find_links

    text = _dehyphenate("code is available at https://github.com/\ntensorflow/tensor2tensor.\nNext line")
    assert find_links(text)["github"] == ["https://github.com/tensorflow/tensor2tensor"]
    # a footnote number on the next line is not glued onto the path
    assert find_links(_dehyphenate("see https://github.com/BlinkDL/RWKV-LM/\n40\n"))["github"] == [
        "https://github.com/BlinkDL/RWKV-LM/"]


def test_rank_code_repos_prefers_the_papers_own_repo():
    from papersum.sections import rank_code_repos

    text = ("We tune with Ax https://github.com/facebook/Ax in our study. "
            "Our code is available at https://github.com/pytorch/fairseq.")
    urls = ["https://github.com/facebook/Ax", "https://github.com/pytorch/fairseq"]
    assert rank_code_repos(text, urls, "wav2vec 2.0")[0] == "https://github.com/pytorch/fairseq"

    # only baselines cited -> return nothing rather than a wrong guess
    baselines = ["https://github.com/state-spaces/mamba", "https://github.com/BlinkDL/RWKV-LM"]
    assert rank_code_repos("We compare against Mamba and RWKV.", baselines, "xLSTM") == []


def test_strip_reasoning_drops_think_blocks():
    """A reasoning trace must not reach the JSON parser, closed or truncated."""
    from papersum.llm import strip_reasoning

    assert strip_reasoning('<think>maybe {"a":9} or ```json</think>{"a":1}').strip() == '{"a":1}'
    assert strip_reasoning("<think>ran out of budget mid-thou").strip() == ""
    assert strip_reasoning("plain answer").strip() == "plain answer"


def test_classify_outline_uses_llm_and_falls_back():
    """Subsections the regex calls 'other' get placed; junk from the model is ignored."""
    from papersum.sections import classify_outline

    headings = ["Abstract", "Optimizer", "Label Smoothing", "Conclusion"]

    class OK:
        def json(self, prompt):
            assert "[1] Optimizer" in prompt          # prompt lists indexed headings
            return {"0": "abstract", "1": "experiments", "2": "experiments", "3": "conclusion"}

    assert classify_outline(OK(), headings) == ["abstract", "experiments", "experiments", "conclusion"]

    class Junk:
        def json(self, prompt):
            return {"0": "not-a-kind", "9": "method"}   # bad category, out-of-range index

    assert classify_outline(Junk(), headings) == ["abstract", "other", "other", "conclusion"]

    class Boom:
        def json(self, prompt):
            raise ValueError("no json")

    assert classify_outline(Boom(), headings) == ["abstract", "other", "other", "conclusion"]


def test_pdf_title_skips_the_arxiv_stamp():
    from papersum.loaders import _pdf_title

    spans = [("arXiv:1706.03762v7  [cs.CL]  2 Aug 2023", 20.0, False),
             ("Attention Is All You", 17.2, True), ("Need", 17.2, True),
             ("Ashish Vaswani", 12.0, False)]
    assert _pdf_title(spans) == "Attention Is All You Need"


def test_html_report_is_one_self_contained_page(tmp_path):
    from papersum import report_html

    pdf = tmp_path / "fake.pdf"
    make_pdf(pdf)
    p = Paper(StubLLM()).load(pdf).run_all(use_web=False)
    p.notes["results_json"] = json.dumps([{"group": "WER", "label": "FooNet", "value": 12.3}])
    out = report_html.write(p, tmp_path / "r.html")
    html = out.read_text(encoding="utf-8")
    assert "<title>" in html and 'class="mermaid"' in html
    assert 'class="bar-fill"' in html                      # metrics drawn, not raw JSON
    assert "prefers-color-scheme" in html                  # readable in either theme
    assert "<script src=" not in html                      # nothing external but mermaid's module


def test_md_to_html_escapes_then_formats():
    from papersum.report_html import md_to_html

    assert md_to_html("**bold**") == "<p><strong>bold</strong></p>"
    assert md_to_html("- a\n- b") == "<ul>\n<li>a</li>\n<li>b</li>\n</ul>"
    assert "&lt;script&gt;" in md_to_html("<script>alert(1)</script>")


def test_clean_mermaid_sanitises_labels():
    """Parens and quotes inside a [label] break Mermaid; edge syntax must survive."""
    from papersum.viz import clean_mermaid

    out = clean_mermaid('flowchart LR\n A[Embedding (d=512)] --> B[Multi-Head "Attention"]')
    assert "A[Embedding d=512]" in out and 'B[Multi-Head Attention]' in out
    assert "|Convolution|" in clean_mermaid("flowchart LR\n X -->|Convolution| Y[Encoder]")
    assert clean_mermaid("A --> B").startswith("flowchart TD")     # bare code gets a header


def test_arxiv_parse_id_accepts_ids_and_urls_but_not_paths():
    from papersum.arxiv import parse_id

    assert parse_id("2405.04517") == "2405.04517"
    assert parse_id("arXiv:1706.03762") == "1706.03762"
    assert parse_id("https://arxiv.org/abs/2006.11477") == "2006.11477"
    assert parse_id("https://arxiv.org/pdf/2405.04517v1.pdf") == "2405.04517v1"
    assert parse_id("papers/input/my_paper.pdf") is None      # a local file is not an id
    assert parse_id("attention is all you need") is None      # a title is not an id


def test_arxiv_figure_urls_resolve_against_the_document():
    """Figure srcs already carry the id; a trailing slash would repeat it and 404."""
    from papersum.arxiv import absolutise

    base = "https://arxiv.org/html/1706.03762v7"
    assert absolutise(base, "1706.03762v7/Figures/x.png") == \
        "https://arxiv.org/html/1706.03762v7/Figures/x.png"
    assert absolutise(base, "https://cdn.example/x.png") == "https://cdn.example/x.png"


def test_arxiv_html_loader_uses_real_headings_and_figures():
    from papersum.loaders import _load_arxiv_html
    from papersum.sections import split_sections

    page = """<article>
      <h1 class="ltx_title ltx_title_document">A Paper</h1>
      <div class="ltx_abstract"><p class="ltx_p">Abstract We propose FooNet, a model for
        devoicing detection that beats every prior baseline by a wide margin.</p></div>
      <h2 class="ltx_title ltx_title_section">1 Introduction</h2>
      <p class="ltx_p">Prior work is sequential. """ + "x" * 60 + """</p>
      <h3 class="ltx_title ltx_title_subsection">3.1 Optimizer</h3>
      <p class="ltx_p">We use Adam with warmup. """ + "y" * 60 + """</p>
      <figure><img src="2405.00001v1/fig1.png"/><figcaption>Figure 1: The architecture.</figcaption></figure>
      <table><tr><th>model</th><th>score</th></tr><tr><td>FooNet</td><td>1.8</td></tr></table>
    </article>"""
    doc = _load_arxiv_html(page, "https://arxiv.org/html/2405.00001v1", {"title": "A Paper"})

    assert doc.title == "A Paper"
    assert [s.heading for s in split_sections(doc.text)][:2] == ["Abstract", "1 Introduction"]
    assert doc.figures[0]["url"] == "https://arxiv.org/html/2405.00001v1/fig1.png"
    assert doc.figures[0]["caption"] == "Figure 1: The architecture."
    assert doc.tables == [["model", "score"], ["FooNet", "1.8"]] or doc.tables[0][0] == ["model", "score"]
    assert "Figure 1" not in doc.text          # captions belong to figures, not body text


def test_bars_emphasise_the_proposed_method():
    from papersum.report_html import _bars

    out = _bars([{"group": "BLEU", "label": "Ours", "value": 28.4, "proposed": True},
                 {"group": "BLEU", "label": "ConvS2S", "value": 25.2, "proposed": False}])
    assert "bar-fill own" in out and out.count("bar-fill") == 2   # exactly one emphasised
    assert "legend" in out
    # no "proposed" flag anywhere -> no emphasis, no legend
    plain = _bars([{"group": "BLEU", "label": "A", "value": 1.0}])
    assert "bar-fill own" not in plain and "legend" not in plain


def test_json_slice_repairs_what_small_models_emit():
    from papersum.llm import _json_slice
    import json as _json

    assert _json.loads(_json_slice('```json\n{"a": 1}\n```')) == {"a": 1}
    assert _json.loads(_json_slice('{"a": 1,}')) == {"a": 1}             # trailing comma
    assert _json.loads(_json_slice("{'a': 'b'}")) == {"a": "b"}          # single quotes
    assert _json.loads(_json_slice('Here you go: {"a": 1} hope that helps')) == {"a": 1}


def test_llm_json_retries_once_then_raises_not_crashes():
    from papersum.llm import LLM

    class Flaky(LLM):
        def __init__(self):
            super().__init__("ollama", "x")
            self.seen = []

        def __call__(self, prompt, system=""):
            self.seen.append(prompt)
            return "not json at all" if len(self.seen) == 1 else '{"ok": true}'

    f = Flaky()
    assert f.json("give me json") == {"ok": True}
    assert "could not be parsed" in f.seen[1]        # the retry shows the model its error

    class Hopeless(LLM):
        def __init__(self):
            super().__init__("ollama", "x")

        def __call__(self, prompt, system=""):
            return "still not json"

    try:
        Hopeless().json("x")
        raise AssertionError("should have raised")
    except ValueError as e:
        assert "did not return valid JSON" in str(e)


def test_results_survives_unparseable_json(tmp_path):
    """One bad JSON reply must not discard the stages that already succeeded."""
    pdf = tmp_path / "fake.pdf"
    make_pdf(pdf)

    class BadJSON(StubLLM):
        def __call__(self, prompt, system=""):
            self.calls += 1
            if "metrics" in prompt:
                return "I could not find any numbers, sorry."
            return "stub"

    p = Paper(BadJSON()).load(pdf).results()
    assert p.notes["results"]                         # fell back to prose
    assert json.loads(p.notes["results_json"]) == []  # no metrics, but no crash

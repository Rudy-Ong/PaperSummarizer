"""Render a finished Paper into one self-contained HTML page for the browser.

The layout follows arXiv's experimental HTML rendering — serif body, a centred title
block with authors and identifiers, numbered sections, figures with captions beneath —
so it reads like the paper it summarises. The content is the opposite: only the digest,
with the numbers pulled out as charts and the paper's own key figures reproduced.

Charts follow the emphasis form: the paper's own method carries the accent hue, every
baseline is de-emphasis gray, so the comparison is readable before anything is parsed.
"""
from __future__ import annotations

import html
import json
import re
from datetime import datetime
from pathlib import Path

_MERMAID_JS = "https://cdn.jsdelivr.net/npm/mermaid@11/dist/mermaid.min.js"

SECTION_ORDER = [
    ("plan", "Reading plan"),
    ("background", "Background &amp; gap"),
    ("method", "Motivation &amp; method"),
    ("workflow_mermaid", "Workflow"),
    ("figures_json", "Key figures"),
    ("model_spec", "Model layers"),
    ("results", "Results"),
    ("limitations", "Limitations &amp; next gap"),
    ("conclusion", "Conclusion"),
]


# --------------------------------------------------------------------- markdown
def _inline(s: str) -> str:
    """Escape, then re-apply the inline markdown the prompts actually produce."""
    s = html.escape(s)
    s = re.sub(r"`([^`]+)`", r"<code>\1</code>", s)
    s = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", s)
    s = re.sub(r"(?<![*\w])\*([^*\n]+)\*(?!\*)", r"<em>\1</em>", s)
    s = re.sub(r"\[([^\]]+)\]\((https?://[^)\s]+)\)", r'<a href="\2" rel="noopener">\1</a>', s)
    return s


def md_to_html(text: str) -> str:
    """Minimal Markdown -> HTML for the subset the summary prompts emit."""
    out: list[str] = []
    list_tag: str | None = None

    def close_list():
        nonlocal list_tag
        if list_tag:
            out.append(f"</{list_tag}>")
            list_tag = None

    for raw in (text or "").splitlines():
        line = raw.rstrip()
        if not line.strip():
            close_list()
            continue
        if m := re.match(r"^(#{1,6})\s+(.*)$", line):
            close_list()
            # The page owns h1 (title) and h2 (section), so stage headings start at h3.
            lvl = min(max(len(m.group(1)) + 1, 3), 6)
            out.append(f"<h{lvl}>{_inline(m.group(2))}</h{lvl}>")
        elif m := re.match(r"^\s*[-*+]\s+(.*)$", line):
            if list_tag != "ul":
                close_list()
                out.append("<ul>")
                list_tag = "ul"
            out.append(f"<li>{_inline(m.group(1))}</li>")
        elif m := re.match(r"^\s*\d+[.)]\s+(.*)$", line):
            if list_tag != "ol":
                close_list()
                out.append("<ol>")
                list_tag = "ol"
            out.append(f"<li>{_inline(m.group(1))}</li>")
        else:
            close_list()
            out.append(f"<p>{_inline(line)}</p>")
    close_list()
    return "\n".join(out)


# ----------------------------------------------------------------- components
def _table(rows: list[list], caption: str = "") -> str:
    if not rows:
        return ""
    head, *body = rows
    th = "".join(f"<th>{_inline(str(c))}</th>" for c in head)
    trs = "".join("<tr>" + "".join(f"<td>{_inline(str(c))}</td>" for c in r) + "</tr>" for r in body)
    cap = f"<figcaption>{_inline(caption)}</figcaption>" if caption else ""
    return (f'<figure class="tbl">{cap}<div class="scroll"><table>'
            f"<thead><tr>{th}</tr></thead><tbody>{trs}</tbody></table></div></figure>")


def _fmt(v: float) -> str:
    return f"{v:,.0f}" if abs(v) >= 1000 else f"{v:g}"


def _bars(metrics: list[dict], emphasise: bool = True) -> str:
    """Horizontal bars, grouped by metric, emphasis form.

    The proposed method gets the accent hue and everything else the de-emphasis gray,
    so "did it win?" is answerable at a glance. Each group is scaled to its own peak —
    WER and BLEU on a shared axis would be meaningless.
    """
    metrics = [m for m in metrics if isinstance(m.get("value"), (int, float))]
    if not metrics:
        return ""
    groups: dict[str, list[dict]] = {}
    for m in metrics:
        groups.setdefault(str(m.get("group") or "value"), []).append(m)

    any_proposed = emphasise and any(m.get("proposed") for m in metrics)
    blocks = []
    for gname, items in list(groups.items())[:6]:
        peak = max(abs(float(i["value"])) for i in items) or 1.0
        rows = []
        for i in items[:12]:
            val = float(i["value"])
            pct = max(abs(val) / peak * 100.0, 1.5)
            own = bool(i.get("proposed"))
            cls = "bar-fill own" if (own and any_proposed) else "bar-fill"
            mark = '<span class="dot" aria-hidden="true"></span>' if own and any_proposed else ""
            label = html.escape(str(i.get("label", "")))
            rows.append(
                f'<div class="bar-row{" is-own" if own and any_proposed else ""}">'
                f'<span class="bar-lab" title="{label}">{mark}{_inline(str(i.get("label", "")))}</span>'
                f'<span class="bar-track"><span class="{cls}" style="width:{pct:.1f}%"></span></span>'
                f'<span class="bar-val">{_fmt(val)}</span></div>')
        hib = next((i.get("higher_is_better") for i in items if i.get("higher_is_better") is not None), None)
        note = "" if hib is None else f'<span class="hint">{"higher is better" if hib else "lower is better"}</span>'
        blocks.append(f'<div class="bar-group"><h4>{_inline(gname)}{note}</h4>{"".join(rows)}</div>')

    legend = ('<div class="legend"><span><i class="sw own"></i>this paper</span>'
              '<span><i class="sw"></i>baseline</span></div>') if any_proposed else ""
    return f'<div class="bars">{legend}{"".join(blocks)}</div>'


def _tiles(metrics: list[dict]) -> str:
    """A KPI row for the paper's own headline numbers — a few values, not a chart."""
    own = [m for m in metrics
           if m.get("proposed") and isinstance(m.get("value"), (int, float))][:4]
    if len(own) < 2:
        return ""
    cells = "".join(
        f'<div class="tile"><span class="tile-k">{_inline(str(m.get("group", "")))}</span>'
        f'<span class="tile-v">{_fmt(float(m["value"]))}</span>'
        f'<span class="tile-l">{_inline(str(m.get("label", ""))[:28])}</span></div>' for m in own)
    return f'<div class="tiles">{cells}</div>'


def _figures(spec_json: str) -> str:
    """The paper's own figures, reproduced with the model's note on why each matters."""
    try:
        figs = json.loads(spec_json)
    except (ValueError, TypeError):
        return ""
    out = []
    for f in figs:
        url = html.escape(str(f.get("url", "")))
        if not url:
            continue
        cap = str(f.get("caption", ""))
        why = str(f.get("why", ""))
        why_html = f'<p class="why">{_inline(why)}</p>' if why else ""
        out.append(f'<figure class="paperfig"><img src="{url}" alt="{html.escape(cap[:150])}" '
                   f'loading="lazy">{why_html}'
                   f'<figcaption>{_inline(cap)}</figcaption></figure>')
    return "".join(out)


def _model_layers(spec_json: str) -> str:
    try:
        spec = json.loads(spec_json)
    except (ValueError, TypeError):
        return ""
    layers = spec.get("layers") or []
    if not layers:
        return ""
    rows = [["#", "layer", "type", "dims", "params"]]
    for i, l in enumerate(layers, 1):
        p = l.get("params")
        rows.append([i, l.get("name", ""), l.get("type", ""), l.get("dims", "") or "",
                     f"{p:,}" if isinstance(p, int) else "?"])
    total = spec.get("total_params")
    head = f'<p class="muted">Source: {_inline(str(spec.get("source", "?")))}'
    head += f" · total ≈ {total:,} params</p>" if isinstance(total, int) else "</p>"
    chart = _bars([{"group": "parameters per block", "label": l.get("name", ""),
                    "value": l.get("params")} for l in layers
                   if isinstance(l.get("params"), (int, float))], emphasise=False)
    return head + chart + _table(rows)


# ----------------------------------------------------------------------- page
# Palette: dataviz reference instance. Accent = categorical slot 1 (#2a78d6 light /
# #3987e5 dark); baselines use the muted gray, which is deliberately achromatic —
# this is the emphasis form (one hue + gray), not a categorical set.
_CSS = """
:root{
 --bg:#f9f9f7;--card:#fcfcfb;--fg:#0b0b0b;--fg2:#52514e;--mut:#898781;
 --line:#e1e0d9;--rule:#c3c2b7;--code:#f1f0ec;
 --accent:#2a78d6;--accent-soft:#cde2fb;--base:#898781;--base-soft:#e1e0d9;
}
@media (prefers-color-scheme:dark){:root:not([data-theme=light]){
 --bg:#0d0d0d;--card:#1a1a19;--fg:#fff;--fg2:#c3c2b7;--mut:#898781;
 --line:#2c2c2a;--rule:#383835;--code:#232320;
 --accent:#3987e5;--accent-soft:#184f95;--base:#898781;--base-soft:#383835;}}
:root[data-theme=dark]{
 --bg:#0d0d0d;--card:#1a1a19;--fg:#fff;--fg2:#c3c2b7;--mut:#898781;
 --line:#2c2c2a;--rule:#383835;--code:#232320;
 --accent:#3987e5;--accent-soft:#184f95;--base:#898781;--base-soft:#383835;}
*{box-sizing:border-box}
body{background:var(--bg);color:var(--fg);margin:0;
 font:16px/1.7 Georgia,"Times New Roman",Cambria,serif}
.wrap{max-width:780px;margin:0 auto;padding-block:44px;padding-left:22px;padding-right:22px}

/* --- arXiv-style title block --- */
header.top{text-align:center;margin:0 0 30px;padding-bottom:22px;border-bottom:1px solid var(--rule)}
h1{font-size:1.9rem;line-height:1.28;margin:0 0 14px;font-weight:600;letter-spacing:-.01em}
.authors{color:var(--fg2);font-size:.95rem;margin:0 0 10px}
.ids{display:flex;flex-wrap:wrap;gap:6px;justify-content:center;margin:12px 0 0}
.ids a,.ids span{font:500 .72rem/1 ui-sans-serif,system-ui,sans-serif;text-decoration:none;
 padding:5px 9px;border-radius:999px;border:1px solid var(--line);color:var(--fg2);background:var(--card)}
.ids a:hover{border-color:var(--accent);color:var(--accent)}
.ids .prim{background:var(--accent);border-color:var(--accent);color:#fff}
.stamp{color:var(--mut);font-size:.74rem;margin:14px 0 0;font-family:ui-sans-serif,system-ui,sans-serif}

/* --- abstract-style lede --- */
.lede{background:var(--card);border:1px solid var(--line);border-radius:8px;
 padding:18px 22px;margin:0 0 26px}
.lede h2{border:0;margin:0 0 8px;font-size:.78rem;text-transform:uppercase;
 letter-spacing:.09em;color:var(--mut);font-family:ui-sans-serif,system-ui,sans-serif}
.lede p:last-child,.lede ul:last-child{margin-bottom:0}

nav.toc{position:sticky;top:env(safe-area-inset-top,0px);background:var(--bg);z-index:5;
 border-bottom:1px solid var(--line);margin:0 0 26px;padding:9px 0}
nav.toc ol{display:flex;flex-wrap:wrap;gap:5px 16px;list-style:none;margin:0;padding:0;
 counter-reset:s;font-family:ui-sans-serif,system-ui,sans-serif}
nav.toc a{color:var(--mut);text-decoration:none;font-size:.78rem}
nav.toc li::before{counter-increment:s;content:counter(s) ". ";color:var(--rule);font-size:.78rem}
nav.toc a:hover{color:var(--accent)}

section.card{margin:0 0 34px;counter-increment:sec}
section.card>h2{font-size:1.12rem;margin:0 0 12px;padding-bottom:7px;
 border-bottom:1px solid var(--line);font-weight:600}
section.card>h2::before{content:counter(sec) ". ";color:var(--mut)}
.wrap{counter-reset:sec}
h3{font-size:1rem;margin:20px 0 6px;font-weight:600}
h4{font-size:.74rem;text-transform:uppercase;letter-spacing:.07em;color:var(--mut);
 margin:18px 0 9px;font-family:ui-sans-serif,system-ui,sans-serif;
 display:flex;justify-content:space-between;align-items:baseline;gap:10px}
h4 .hint{text-transform:none;letter-spacing:0;font-weight:400;color:var(--mut);font-size:.72rem}
p{margin:0 0 12px}
a{color:var(--accent)}
ul,ol{margin:0 0 12px;padding-left:22px}
li{margin:4px 0}
code{background:var(--code);padding:1px 5px;border-radius:4px;
 font:.86em/1.5 ui-monospace,SFMono-Regular,Menlo,monospace}
.muted{color:var(--mut);font-size:.85rem}

/* --- KPI tiles --- */
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(130px,1fr));gap:10px;margin:0 0 20px}
.tile{background:var(--card);border:1px solid var(--line);border-left:3px solid var(--accent);
 border-radius:7px;padding:11px 13px;display:flex;flex-direction:column;gap:2px}
.tile-k{font:.68rem/1.3 ui-sans-serif,system-ui,sans-serif;text-transform:uppercase;
 letter-spacing:.06em;color:var(--mut)}
.tile-v{font:600 1.6rem/1.15 ui-sans-serif,system-ui,sans-serif;
 font-variant-numeric:tabular-nums;color:var(--fg)}
.tile-l{font:.72rem/1.3 ui-sans-serif,system-ui,sans-serif;color:var(--fg2)}

/* --- bars: emphasis form, 4px rounded data-end, hairline baseline --- */
.bars{margin:6px 0 18px;font-family:ui-sans-serif,system-ui,sans-serif}
.legend{display:flex;gap:16px;margin:0 0 12px;font-size:.74rem;color:var(--fg2)}
.legend i.sw{display:inline-block;width:11px;height:11px;border-radius:3px;
 background:var(--base);margin-right:6px;vertical-align:-1px}
.legend i.sw.own{background:var(--accent)}
.bar-group{margin:0 0 18px;border-left:1px solid var(--line);padding-left:13px}
.bar-row{display:grid;grid-template-columns:minmax(74px,1.15fr) 2.6fr auto;gap:11px;
 align-items:center;margin:3px 0;font-size:.8rem;color:var(--fg2)}
.bar-row.is-own{color:var(--fg);font-weight:600}
.bar-lab{overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.bar-lab .dot{display:inline-block;width:6px;height:6px;border-radius:50%;
 background:var(--accent);margin-right:5px;vertical-align:1px}
.bar-track{background:var(--base-soft);border-radius:3px;height:15px;overflow:hidden}
.bar-fill{display:block;height:100%;background:var(--base);
 border-radius:0 4px 4px 0}
.bar-fill.own{background:var(--accent)}
.bar-val{font-variant-numeric:tabular-nums;font-size:.78rem;min-width:44px;text-align:right}

/* --- figures --- */
figure.paperfig{margin:0 0 24px;background:var(--card);border:1px solid var(--line);
 border-radius:8px;padding:16px;text-align:center}
figure.paperfig img{max-width:100%;height:auto;border-radius:4px;background:#fff}
figure.paperfig .why{font:600 .88rem/1.5 ui-sans-serif,system-ui,sans-serif;
 color:var(--fg);margin:13px 0 5px;text-align:left}
figure.paperfig figcaption{color:var(--mut);font-size:.79rem;text-align:left;line-height:1.55}
figure.tbl{margin:14px 0}
figcaption{color:var(--mut);font-size:.8rem;margin-bottom:6px}
.scroll{overflow-x:auto}
table{border-collapse:collapse;width:100%;font:.79rem/1.5 ui-sans-serif,system-ui,sans-serif}
th,td{border:1px solid var(--line);padding:6px 9px;text-align:left;vertical-align:top}
th{background:var(--code);font-weight:600}
pre.mermaid{background:var(--card);border:1px solid var(--line);border-radius:8px;
 padding:16px;text-align:center;overflow-x:auto;margin:0}
details{margin-top:10px}
summary{cursor:pointer;color:var(--mut);font-size:.79rem;font-family:ui-sans-serif,system-ui,sans-serif}
details pre{background:var(--code);padding:12px;border-radius:7px;overflow-x:auto;
 font:.78rem/1.5 ui-monospace,Menlo,monospace}
@media (max-width:560px){
 body{font-size:15px}h1{font-size:1.45rem}
 .bar-row{grid-template-columns:1fr auto}.bar-track{grid-column:1/-1}}
@media print{nav.toc{display:none}section.card{break-inside:avoid}}
"""

_JS = """
import mermaid from "%s";
mermaid.initialize({startOnLoad:false, securityLevel:"loose",
  theme: matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "neutral"});
const nodes=[...document.querySelectorAll("pre.mermaid")];
if(nodes.length){ try{ await mermaid.run({nodes}); }
  catch(e){ nodes.forEach(n=>n.style.whiteSpace="pre"); } }
""" % _MERMAID_JS


def _header(paper) -> str:
    doc, meta = paper.doc, (paper.doc.meta if paper.doc else {})
    title = doc.title if doc else "Paper summary"
    bits = [f"<h1>{html.escape(title)}</h1>"]

    authors = meta.get("authors") or []
    if authors:
        shown = ", ".join(authors[:6]) + (f" +{len(authors) - 6} more" if len(authors) > 6 else "")
        bits.append(f'<p class="authors">{html.escape(shown)}</p>')

    pills = []
    if meta.get("abs_url"):
        pills.append(f'<a class="prim" href="{html.escape(meta["abs_url"])}" rel="noopener">'
                     f'arXiv:{html.escape(meta.get("id", ""))}</a>')
    if meta.get("pdf_url"):
        pills.append(f'<a href="{html.escape(meta["pdf_url"])}" rel="noopener">PDF</a>')
    if doc and doc.figures and meta.get("html_url"):
        pills.append(f'<a href="{html.escape(meta["html_url"])}" rel="noopener">arXiv HTML</a>')
    if paper.repo.get("url"):
        stars = paper.repo.get("stars")
        label = paper.repo.get("repo", "code") + (f" ★{stars:,}" if isinstance(stars, int) else "")
        pills.append(f'<a href="{html.escape(paper.repo["url"])}" rel="noopener">{html.escape(label)}</a>')
    for cat in (meta.get("categories") or [])[:3]:
        pills.append(f"<span>{html.escape(cat)}</span>")
    if meta.get("published"):
        pills.append(f'<span>{html.escape(meta["published"])}</span>')
    if pills:
        bits.append(f'<div class="ids">{"".join(pills)}</div>')

    src = meta.get("abs_url") or (str(doc.path) if doc else "")
    bits.append(f'<p class="stamp">Summarised by papersum · {paper.llm.backend}/'
                f'{html.escape(str(paper.llm.model))} · {paper.llm.calls} calls · '
                f'{datetime.now():%Y-%m-%d %H:%M}<br>Source: {html.escape(str(src))}</p>')
    return f'<header class="top">{"".join(bits)}</header>'


def render(paper, title: str | None = None) -> str:
    notes = paper.notes
    doc_title = title or (paper.doc.title if paper.doc else "Paper summary")

    try:
        metrics = json.loads(notes.get("results_json", "[]"))
    except ValueError:
        metrics = []

    present = [(k, t) for k, t in SECTION_ORDER if notes.get(k)]
    # the reading plan leads as an abstract-style lede rather than a numbered section
    lede = ""
    if notes.get("plan"):
        lede = (f'<div class="lede"><h2>Reading plan</h2>{md_to_html(notes["plan"])}</div>')
        present = [(k, t) for k, t in present if k != "plan"]

    toc = "".join(f'<li><a href="#{k}">{t}</a></li>' for k, t in present)
    if paper.doc and paper.doc.tables:
        toc += '<li><a href="#tables">Tables</a></li>'

    body: list[str] = []
    for key, heading in present:
        if key == "workflow_mermaid":
            code = notes[key]
            inner = (f'<pre class="mermaid">{html.escape(code)}</pre>'
                     f"<details><summary>Mermaid source</summary>"
                     f"<pre>{html.escape(code)}</pre></details>")
        elif key == "figures_json":
            inner = _figures(notes[key])
        elif key == "model_spec":
            inner = _model_layers(notes[key])
        elif key == "results":
            inner = _tiles(metrics) + md_to_html(notes[key]) + _bars(metrics)
        else:
            inner = md_to_html(notes[key])
        if inner.strip():
            body.append(f'<section class="card" id="{key}"><h2>{heading}</h2>{inner}</section>')

    if paper.doc and paper.doc.tables:
        tbls = "".join(_table(t[:14], f"Table {i + 1} (extracted from the source)")
                       for i, t in enumerate(paper.doc.tables[:6]))
        body.append(f'<section class="card" id="tables"><h2>Tables</h2>{tbls}</section>')

    return (f"<title>{html.escape(doc_title)}</title>\n"
            f"<style>{_CSS}</style>\n"
            f'<div class="wrap">\n{_header(paper)}\n{lede}\n'
            f'<nav class="toc"><ol>{toc}</ol></nav>\n{"".join(body)}\n</div>\n'
            f'<script type="module">{_JS}</script>\n')


def write(paper, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render(paper), encoding="utf-8")
    return path

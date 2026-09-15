"""Load PDF / Markdown / EPUB into a single text stream with heading markers.

Every loader returns a `Doc`:  text (with `\n## Heading` markers where a heading was
detected), tables (list of 2-D lists, PDF only) and the source path.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Doc:
    path: str
    text: str
    tables: list[list[list[str]]] = field(default_factory=list)
    n_pages: int = 0
    title_hint: str = ""        # PDF loader: biggest type on page 1
    figures: list[dict] = field(default_factory=list)   # arXiv HTML: {id, url, caption}
    meta: dict = field(default_factory=dict)            # arXiv API metadata, when known

    @property
    def title(self) -> str:
        if self.title_hint:
            return self.title_hint
        first = self.text.lstrip().splitlines()[0] if self.text.strip() else ""
        if first.startswith("## ") and 3 < len(first) < 200:      # md/epub: first heading is the title
            return first[3:].strip()
        for line in self.text.splitlines():
            line = line.strip().lstrip("#").strip()
            if 10 < len(line) < 200:
                return line
        return Path(self.path).stem


# --------------------------------------------------------------------------- PDF
def _load_pdf(path: Path) -> Doc:
    import pymupdf  # PyMuPDF

    doc = pymupdf.open(path)
    lines: list[str] = []
    tables: list[list[list[str]]] = []

    # Estimate body font size: the most frequent size across the document.
    sizes: dict[float, int] = {}
    pages_spans = []
    for page in doc:
        blocks = page.get_text("dict")["blocks"]
        spans = []
        for b in blocks:
            for l in b.get("lines", []):
                txt = "".join(s["text"] for s in l["spans"]).strip()
                if not txt:
                    continue
                s0 = l["spans"][0]
                size = round(s0["size"], 1)
                bold = "Bold" in s0["font"] or (s0["flags"] & 16)
                sizes[size] = sizes.get(size, 0) + len(txt)
                spans.append((txt, size, bool(bold)))
        pages_spans.append(spans)
        try:
            for t in page.find_tables():
                rows = t.extract()
                rows = [[(c or "").strip() for c in r] for r in rows]
                if len(rows) >= 2 and any(any(c for c in r) for r in rows):
                    tables.append(rows)
        except Exception:
            pass

    body = max(sizes, key=sizes.get) if sizes else 10.0

    for spans in pages_spans:
        for txt, size, bold in spans:
            is_heading = (size >= body + 1.0 or bold) and len(txt) < 120 and not txt.endswith(".")
            if is_heading and _looks_like_heading(txt):
                lines.append(f"\n## {txt}\n")
            else:
                lines.append(txt)
    text = _dehyphenate("\n".join(lines))
    return Doc(str(path), text, tables, n_pages=len(doc),
               title_hint=_pdf_title(pages_spans[0] if pages_spans else []))


# ---------------------------------------------------------------------- Markdown
def _load_md(path: Path) -> Doc:
    text = path.read_text(encoding="utf-8", errors="ignore")
    # normalise any heading level to "## "
    text = re.sub(r"^#{1,6}\s+", "## ", text, flags=re.M)
    return Doc(str(path), text)


# -------------------------------------------------------------------------- EPUB
def _load_epub(path: Path) -> Doc:
    import ebooklib
    from bs4 import BeautifulSoup
    from ebooklib import epub

    book = epub.read_epub(str(path), options={"ignore_ncx": True})
    lines: list[str] = []
    for item in book.get_items_of_type(ebooklib.ITEM_DOCUMENT):
        soup = BeautifulSoup(item.get_content(), "html.parser")
        for el in soup.find_all(["h1", "h2", "h3", "h4", "p", "li", "table"]):
            txt = el.get_text(" ", strip=True)
            if not txt:
                continue
            if el.name.startswith("h"):
                lines.append(f"\n## {txt}\n")
            else:
                lines.append(txt)
    return Doc(str(path), "\n".join(lines))


# ------------------------------------------------------------------------ helpers
_HEADING_WORDS = re.compile(
    r"^(\d+(\.\d+)*\.?\s+)?(abstract|introduction|background|related work|preliminar|"
    r"method|approach|model|architecture|experiment|setup|result|evaluation|analysis|"
    r"discussion|limitation|ablation|conclusion|future work|acknowledg|reference|appendix|"
    r"[A-Z][A-Za-z\-\s:]{2,80})$",
    re.I,
)


_ARXIV_STAMP = re.compile(r"^arxiv[:\s]", re.I)


def _pdf_title(page_spans: list[tuple[str, float, bool]]) -> str:
    """The paper title = the largest type on page 1, ignoring the arXiv margin stamp.

    Reading the first line of extracted text instead picks up whatever the publisher put
    at the top — for "Attention Is All You Need" that is a Google licensing paragraph.
    The stamp is set larger than the title, so it has to be excluded by pattern; a title
    wrapped over two lines is rejoined by taking the run of lines at that same size.
    """
    cands = [(t.strip(), size) for t, size, _ in page_spans
             if len(t.strip()) > 3 and not _ARXIV_STAMP.match(t.strip())]
    if not cands:
        return ""
    top = max(size for _, size in cands)
    run: list[str] = []
    for t, size in cands:
        if abs(size - top) < 0.1:
            run.append(t)
        elif run:
            break                       # title ended; stop before authors/affiliations
    title = " ".join(run).strip()
    return title if 6 < len(title) < 300 else ""


def _looks_like_heading(txt: str) -> bool:
    if len(txt.split()) > 14:
        return False
    if re.match(r"^\d+(\.\d+)*\.?\s+\S", txt):
        return True
    return bool(_HEADING_WORDS.match(txt))


def _dehyphenate(text: str) -> str:
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)          # line-break hyphens
    # A URL wrapped after a "/" must be rejoined with no space, or the generic
    # line-join below truncates it at the break (https://github.com/ + tensorflow/t2t).
    # The continuation must contain a letter, so a footnote/citation number sitting on
    # the next line ("…/RWKV-LM/\n40") is not glued on as if it were part of the path.
    text = re.sub(r"(https?://\S*/)\n(?=[^\s]*[A-Za-z][^\s]*)", r"\1", text)
    text = re.sub(r"(?<!\n)\n(?!\n|## )", " ", text)       # join wrapped lines
    return re.sub(r"[ \t]{2,}", " ", text)


# ------------------------------------------------------------------ arXiv HTML
_LTX_HEADING = {"ltx_title_section", "ltx_title_subsection", "ltx_title_subsubsection"}


def _load_arxiv_html(html: str, base_url: str, meta: dict | None = None) -> Doc:
    """Parse arXiv's experimental LaTeXML rendering.

    Everything `_load_pdf` infers from font sizes is explicit here: headings carry
    `ltx_title_section`/`ltx_title_subsection` classes in document order, tables are
    real markup, and figures come with captions and image URLs.
    """
    from bs4 import BeautifulSoup

    from . import arxiv

    soup = BeautifulSoup(html, "html.parser")
    art = soup.find("article") or soup.find("div", class_="ltx_page_content") or soup
    meta = meta or {}

    node = art.find(class_="ltx_title_document")
    title = node.get_text(" ", strip=True) if node else (meta.get("title") or "")

    figures: list[dict] = []
    for fig in art.find_all("figure"):
        cap = fig.find("figcaption")
        caption = cap.get_text(" ", strip=True) if cap else ""
        img, obj = fig.find("img"), fig.find("object")
        src = (img.get("src") if img else None) or (obj.get("data") if obj else None)
        if src and caption:
            figures.append({"id": fig.get("id", ""), "caption": caption,
                            "url": arxiv.absolutise(base_url, src)})

    lines: list[str] = []
    if abstract := art.find(class_="ltx_abstract"):
        text = abstract.get_text(" ", strip=True)
        lines += ["\n## Abstract\n", re.sub(r"^Abstract\s*", "", text)]

    for el in art.find_all(["h2", "h3", "h4", "p"]):
        if set(el.get("class") or []) & _LTX_HEADING:
            lines.append(f"\n## {el.get_text(' ', strip=True)}\n")
        elif el.name == "p":
            # captions belong to `figures`; the abstract is already emitted above
            if el.find_parent("figure") or el.find_parent(class_="ltx_abstract"):
                continue
            if txt := el.get_text(" ", strip=True):
                lines.append(txt)

    tables: list[list[list[str]]] = []
    for t in art.find_all("table"):
        rows = [[c.get_text(" ", strip=True) for c in tr.find_all(["td", "th"])]
                for tr in t.find_all("tr")]
        rows = [r for r in rows if any(c for c in r)]
        if len(rows) >= 2:
            tables.append(rows)

    return Doc(meta.get("abs_url") or base_url, "\n".join(lines), tables,
               title_hint=title, figures=figures, meta=meta)


def load_arxiv(query: str) -> Doc:
    """Load a paper from arXiv by id, URL, or title — HTML when available, else PDF."""
    from . import arxiv

    meta = arxiv.resolve(query)
    if not meta:
        raise ValueError(f"No arXiv paper found for {query!r}")
    html, base = arxiv.fetch_html(meta.get("version_id") or meta["id"])
    if html:
        return _load_arxiv_html(html, base, meta)

    import requests                                   # no HTML rendering — fall back to the PDF
    dest = Path("papers/input") / f"{meta['id'].replace('/', '_')}.pdf"
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not dest.exists():
        r = requests.get(meta["pdf_url"], headers={"User-Agent": "papersum/0.1"}, timeout=60)
        r.raise_for_status()
        dest.write_bytes(r.content)
    doc = _load_pdf(dest)
    doc.meta = meta
    doc.title_hint = doc.title_hint or meta.get("title", "")
    return doc


def load_document(path: str | Path) -> Doc:
    from . import arxiv

    if isinstance(path, str) and not Path(path).exists() and arxiv.parse_id(path):
        return load_arxiv(path)
    path = Path(path)
    ext = path.suffix.lower()
    if ext == ".pdf":
        return _load_pdf(path)
    if ext in {".md", ".markdown", ".txt"}:
        return _load_md(path)
    if ext == ".epub":
        return _load_epub(path)
    raise ValueError(f"Unsupported format: {ext} (use .pdf, .md, .epub)")

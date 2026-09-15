"""arXiv access: find a paper by id or title, and prefer its HTML over the PDF.

arXiv publishes an experimental LaTeXML rendering at /html/<id>, which carries the
things the PDF loader has to guess at: real `<h2>`/`<h3>` section headings in document
order, real `<table>` markup, and `<figure>` elements with captions and image URLs.
When it exists it is a far better source than font-size heuristics over a PDF.

    papersum 2405.04517                     # id
    papersum https://arxiv.org/abs/1706.03762
    papersum "attention is all you need"    # title search
"""
from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from urllib.parse import quote, urljoin

import requests

API = "http://export.arxiv.org/api/query"
_ATOM = {"a": "http://www.w3.org/2005/Atom", "arxiv": "http://arxiv.org/schemas/atom"}
_UA = {"User-Agent": "papersum/0.1 (paper summarizer; contact via repo)"}

# 2405.04517, 2405.04517v2, and the pre-2007 style (cs/0112017)
_ID_RE = re.compile(r"(?:arxiv[:\s/]*)?(\d{4}\.\d{4,5}(?:v\d+)?|[a-z-]+(?:\.[A-Z]{2})?/\d{7}(?:v\d+)?)", re.I)
_URL_RE = re.compile(r"arxiv\.org/(?:abs|pdf|html)/([^\s?#]+)", re.I)


def parse_id(s: str) -> str | None:
    """Pull an arXiv id out of a bare id, an abs/pdf/html URL, or None if it isn't one."""
    s = (s or "").strip()
    if m := _URL_RE.search(s):
        return re.sub(r"\.pdf$", "", m.group(1))
    # a bare id only — don't match an id embedded in a sentence or a local filename
    if m := _ID_RE.fullmatch(s):
        return m.group(1)
    return None


def _get(url, **kw):
    try:
        r = requests.get(url, headers=_UA, timeout=30, **kw)
        if r.ok:
            return r
    except requests.RequestException:
        pass
    return None


def _entry(e) -> dict:
    def txt(tag, default=""):
        n = e.find(f"a:{tag}", _ATOM)
        return " ".join(n.text.split()) if n is not None and n.text else default

    raw_id = txt("id")
    aid = (parse_id(raw_id) or raw_id).strip()
    return {
        "id": aid,
        "version_id": aid,
        "title": txt("title"),
        "abstract": txt("summary"),
        "authors": [a.findtext("a:name", "", _ATOM) for a in e.findall("a:author", _ATOM)],
        "published": txt("published")[:10],
        "updated": txt("updated")[:10],
        "categories": [c.get("term") for c in e.findall("a:category", _ATOM)],
        "comment": (e.findtext("arxiv:comment", "", _ATOM) or "").strip(),
        "doi": (e.findtext("arxiv:doi", "", _ATOM) or "").strip(),
        "journal_ref": (e.findtext("arxiv:journal_ref", "", _ATOM) or "").strip(),
        "abs_url": f"https://arxiv.org/abs/{aid}",
        "pdf_url": f"https://arxiv.org/pdf/{aid}",
        "html_url": f"https://arxiv.org/html/{aid}",
    }


def query(search: str = "", id_list: str = "", n: int = 5) -> list[dict]:
    """Raw arXiv API call. Returns [] on any network/parse failure."""
    url = f"{API}?search_query={quote(search)}&id_list={quote(id_list)}&start=0&max_results={n}"
    r = _get(url)
    if not r:
        return []
    try:
        root = ET.fromstring(r.content)
    except ET.ParseError:
        return []
    return [_entry(e) for e in root.findall("a:entry", _ATOM)]


def metadata(arxiv_id: str) -> dict:
    """Everything arXiv knows about one paper."""
    hits = query(id_list=arxiv_id, n=1)
    return hits[0] if hits else {}


def search(title: str, n: int = 5) -> list[dict]:
    """Find a paper by title. Quoted phrase first, falling back to a loose AND search."""
    words = re.findall(r"[A-Za-z0-9]+", title)
    if not words:
        return []
    hits = query(search=f'ti:"{title}"', n=n)
    if not hits:
        hits = query(search=" AND ".join(f"all:{w}" for w in words[:8]), n=n)
    return hits


def resolve(text: str) -> dict:
    """Accept an id, an arXiv URL, or a title, and return the paper's metadata."""
    if aid := parse_id(text):
        return metadata(aid)
    hits = search(text, n=1)
    return hits[0] if hits else {}


def fetch_html(arxiv_id: str) -> tuple[str, str] | tuple[None, None]:
    """Return (html, base_url) for the experimental rendering, or (None, None).

    Not every paper has one — arXiv only generates HTML for LaTeX submissions from
    late 2023 onward, so a PDF fallback is always needed.
    """
    for candidate in (arxiv_id, re.sub(r"v\d+$", "", arxiv_id)):
        url = f"https://arxiv.org/html/{candidate}"
        r = _get(url, allow_redirects=True)
        # arXiv answers a missing rendering with a small "no HTML" notice page, not a 404
        if r and len(r.text) > 20_000 and "ltx_document" in r.text:
            # Return the document URL itself, NOT a directory form. Figure srcs are
            # "<id>v<n>/Figures/x.png", already relative to /html/, and forcing a
            # trailing slash here makes urljoin repeat the id and 404 every image.
            return r.text, r.url
    return None, None


def absolutise(base_url: str, src: str) -> str:
    return src if src.startswith(("http://", "https://", "data:")) else urljoin(base_url, src)

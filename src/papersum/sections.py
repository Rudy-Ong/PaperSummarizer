"""Split a Doc's text into canonical sections and extract links."""
from __future__ import annotations

import re
from dataclasses import dataclass

CANON = [
    ("abstract", r"abstract"),
    ("introduction", r"introduction|background|motivation"),
    ("related", r"related work|prior work|literature"),
    ("method", r"method|approach|model|architecture|framework|proposed|system"),
    ("experiments", r"experiment|setup|implementation|dataset|training detail"),
    ("results", r"result|evaluation|analysis|ablation|finding"),
    ("discussion", r"discussion"),
    ("limitations", r"limitation|threat|future work|broader impact|ethic"),
    ("conclusion", r"conclusion|summary"),
    ("references", r"reference|bibliograph"),
    ("appendix", r"appendix|supplement"),
]


@dataclass
class Section:
    heading: str
    kind: str
    text: str

    def __repr__(self):
        return f"<{self.kind}: '{self.heading}' {len(self.text)} chars>"


def classify(heading: str) -> str:
    h = re.sub(r"^\d+(\.\d+)*\.?\s*", "", heading).lower()
    for kind, pat in CANON:
        if re.search(pat, h):
            return kind
    return "other"


def merge_sections(sections: list[Section]) -> list[Section]:
    """Fold consecutive sections of the same kind into one (e.g. 3.1, 3.2 -> method)."""
    merged: list[Section] = []
    for s in sections:
        if merged and merged[-1].kind == s.kind and s.kind not in {"other", "front"}:
            merged[-1].text += f"\n\n### {s.heading}\n{s.text}"
        else:
            merged.append(s)
    return merged


def split_sections(text: str, merge: bool = True) -> list[Section]:
    """Parse "## heading" markers into sections, classified by the regex heuristic.

    Pass merge=False to get one Section per heading, which is what the LLM outline
    classifier needs — merging has to happen *after* the kinds are reassigned.
    """
    parts = re.split(r"\n## ", "\n" + text)
    sections: list[Section] = []
    pre = parts[0].strip()
    if pre:
        sections.append(Section("(front matter)", "front", pre))
    for p in parts[1:]:
        heading, _, body = p.partition("\n")
        body = body.strip()
        if len(body) < 40:               # merge trivial headings into next block
            continue
        sections.append(Section(heading.strip(), classify(heading), body))
    return merge_sections(sections) if merge else sections


# The fixed vocabulary the outline classifier may return — one of these, nothing else.
CANON_KINDS = [k for k, _ in CANON] + ["other"]

_KIND_HINTS = {
    "abstract": "the paper's own abstract",
    "introduction": "introduction, background, motivation — the problem and why it matters",
    "related": "related/prior work, literature review, comparisons to other papers",
    "method": "the proposed approach: architecture, model, algorithm, formulation, components",
    "experiments": "experimental setup: datasets, training details, hyper-parameters, implementation, baselines",
    "results": "measured outcomes: evaluation, tables of numbers, ablations, analysis of findings",
    "discussion": "interpretation of the results, general discussion",
    "limitations": "limitations, threats to validity, future work, broader impact, ethics",
    "conclusion": "closing summary of the paper",
    "references": "bibliography",
    "appendix": "appendix, supplementary material",
    "other": "front matter, author lists, acknowledgements, notation, or anything that fits nothing above",
}

# The template prompt. Format with: title, kinds, hints, outline.
OUTLINE_PROMPT = """\
Paper title: {title}

Below is the complete outline of the paper: every heading and subheading that was
detected, one per line, in document order, each with its index.

{outline}

Task: assign EVERY index to exactly one category from this fixed list:
{kinds}

What each category means:
{hints}

Rules:
- A subsection belongs to the category of the work it describes, not its own wording.
  "Optimizer", "Masking", "Positional Encoding" are parts of the proposed approach -> method.
  "Training Data and Batching", "Hardware and Schedule" describe the setup -> experiments.
  "Machine Translation", "Model Variations" report measured outcomes -> results.
- Use the surrounding headings for context: an unnumbered subsection usually continues
  the category of the numbered section above it.
- Every index must appear exactly once. Use "other" only when nothing else fits.

Return ONLY a JSON object mapping index (as a string) to category, e.g.
{{"0": "abstract", "1": "introduction", "2": "method"}}"""


def build_outline_prompt(headings: list[str], title: str = "") -> str:
    """Render OUTLINE_PROMPT for a list of headings. Exposed so it can be inspected/tuned."""
    outline = "\n".join(f"[{i}] {h}" for i, h in enumerate(headings))
    hints = "\n".join(f"- {k}: {_KIND_HINTS[k]}" for k in CANON_KINDS)
    return OUTLINE_PROMPT.format(title=title or "(unknown)", outline=outline,
                                 kinds=", ".join(CANON_KINDS), hints=hints)


def classify_outline(llm, headings: list[str], title: str = "", batch: int = 60,
                     fallback: list[str] | None = None) -> list[str]:
    """Classify a whole outline in one LLM call, returning one kind per heading.

    The regex in `classify` only sees a heading in isolation, so real subsections
    ("Optimizer", "Masking", "Why Self-Attention") fall through to "other" and drop out
    of `by_kind("method")`. Showing the model the entire outline at once gives it the
    document order it needs to place them. Falls back to the regex result per-heading
    whenever the model omits an index or returns a category outside CANON_KINDS.
    """
    kinds = list(fallback) if fallback else [classify(h) for h in headings]
    for start in range(0, len(headings), batch):
        chunk = headings[start:start + batch]
        try:
            data = llm.json(build_outline_prompt(chunk, title))
        except Exception:
            continue                                   # keep the regex result for this chunk
        if not isinstance(data, dict):
            continue
        for key, kind in data.items():
            try:
                i = int(str(key).strip().strip("[]"))
            except ValueError:
                continue
            if 0 <= i < len(chunk) and kind in CANON_KINDS:
                kinds[start + i] = kind
    return kinds


def by_kind(sections: list[Section], *kinds: str, fallback_chars: int = 6000) -> str:
    """Concatenate sections of the given kinds; if none, return the first N chars."""
    out = [f"## {s.heading}\n{s.text}" for s in sections if s.kind in kinds]
    if out:
        return "\n\n".join(out)
    return "\n\n".join(f"## {s.heading}\n{s.text}" for s in sections)[:fallback_chars]


URL_RE = re.compile(r"https?://[^\s)\]>\"']+")


def find_links(text: str) -> dict[str, list[str]]:
    urls = sorted({u.rstrip(".,;") for u in URL_RE.findall(text)})
    return {
        "github": [u for u in urls if "github.com" in u or "gitlab.com" in u],
        "huggingface": [u for u in urls if "huggingface.co" in u],
        "other": [u for u in urls if not any(k in u for k in ("github.com", "gitlab.com", "huggingface.co"))],
    }


_AVAILABILITY = re.compile(
    r"(code|implementation|source|models?|weights|scripts?|toolkit)[^.]{0,80}?"
    r"(is |are |publicly |freely )*(available|released|open[- ]sourced|can be found)"
    r"|we (release|open[- ]source|provide)|available at|released at|found at",
    re.I,
)
_STOP = {"the", "a", "an", "of", "for", "and", "with", "on", "in", "to", "is", "are",
         "using", "via", "towards", "toward", "learning", "networks", "network", "model",
         "models", "deep", "neural", "you", "need", "all"}


def rank_code_repos(text: str, urls: list[str], title: str = "", min_score: int = 1) -> list[str]:
    """Order candidate repo URLs so the paper's *own* code repo comes first.

    `find_links` returns URLs sorted alphabetically, so picking urls[0] reliably grabs
    whichever repo happens to sort first — usually a baseline or a tool cited in passing
    (e.g. facebook/Ax instead of pytorch/fairseq). Rank instead by two signals: the URL
    sitting next to an "code is available at ..." sentence, and the repo name echoing
    words from the paper title. Ties keep the original order.

    URLs scoring below `min_score` are dropped rather than returned as a weak guess: a
    paper that only cites its *baselines'* repos should fall through to web search, since
    handing the wrong project's source to the model-layer stage yields a confidently
    wrong architecture diagram. Returning nothing is the safer failure.
    """
    refs = text.lower().rfind("\n## reference")
    title_words = {w for w in re.findall(r"[a-z0-9]+", (title or "").lower())
                   if len(w) > 2 and w not in _STOP}

    def score(url: str) -> int:
        s = 0
        name = re.sub(r"^https?://[^/]+/", "", url).strip("/").lower()
        name_words = set(re.findall(r"[a-z0-9]+", name))
        if title_words & name_words:
            s += 3
        for m in re.finditer(re.escape(url), text):
            window = text[max(0, m.start() - 220):m.start()]
            if _AVAILABILITY.search(window):
                s += 4
            if refs != -1 and m.start() > refs:      # cited in the bibliography, not the paper's own code
                s -= 2
            break
        if name.count("/") < 1:                      # bare "github.com/org" — not a repo
            s -= 5
        return s

    scored = sorted(((score(u), i, u) for i, u in enumerate(urls)), key=lambda t: (-t[0], t[1]))
    return [u for s, _, u in scored if s >= min_score]

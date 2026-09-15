"""The step-by-step pipeline. Each method = one notebook cell.

    paper = Paper(llm).load("papers/x.pdf")
    paper.plan(); paper.background(); paper.method(); paper.workflow_diagram()
    paper.results(); paper.limitations(); paper.conclusion()
    paper.export("report.md")
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from . import retrieve, viz
from .llm import LLM
from .loaders import Doc, load_document
from .sections import (Section, by_kind, classify_outline, find_links, merge_sections,
                       rank_code_repos, split_sections)


@dataclass
class Paper:
    llm: LLM
    doc: Doc | None = None
    sections: list[Section] = field(default_factory=list)
    links: dict = field(default_factory=dict)
    repo: dict = field(default_factory=dict)
    web: list = field(default_factory=list)
    notes: dict[str, str] = field(default_factory=dict)   # step -> markdown output

    # ------------------------------------------------------------ 2. load
    def load(self, path: str | Path) -> "Paper":
        return self.load_doc(load_document(path))

    def load_doc(self, doc: Doc) -> "Paper":
        """Attach an already-loaded Doc (e.g. from `load_arxiv`) and split it."""
        self.doc = doc
        self.sections = split_sections(self.doc.text)
        self.links = find_links(self.doc.text)
        where = doc.meta.get("abs_url") or Path(doc.path).name
        counts = [f"{len(self.doc.text):,} chars", f"{len(self.sections)} sections",
                  f"{len(self.doc.tables)} tables"]
        if self.doc.n_pages:
            counts.insert(0, f"{self.doc.n_pages} pages")
        if self.doc.figures:
            counts.append(f"{len(self.doc.figures)} figures")
        viz.md(f"**Loaded** `{where}` — " + ", ".join(counts) + f"\n\n**Title:** {self.doc.title}")
        viz.table([["#", "kind", "heading", "chars"]] +
                  [[i, s.kind, s.heading[:60], len(s.text)] for i, s in enumerate(self.sections)],
                  "Detected sections")
        return self

    def classify_outline(self) -> "Paper":
        """Re-assign every heading to a canonical kind by showing the LLM the whole outline.

        The regex classifier judges each heading alone, so subsections land in "other" and
        never reach `by_kind("method")`. One call over the full outline fixes that; the
        regex result stays as the per-heading fallback.
        """
        raw = split_sections(self.doc.text, merge=False)
        kinds = classify_outline(self.llm, [s.heading for s in raw], self.doc.title,
                                 fallback=[s.kind for s in raw])
        # "front" is assigned structurally, not from heading text — the model cannot see
        # that the pre-heading block is front matter, so keep it.
        kinds = [s.kind if s.kind == "front" else k for s, k in zip(raw, kinds)]
        moved = [(s.heading, s.kind, k) for s, k in zip(raw, kinds) if s.kind != k]
        for s, k in zip(raw, kinds):
            s.kind = k
        self.sections = merge_sections(raw)
        viz.md(f"**Outline classified** — {len(moved)} of {len(raw)} headings reassigned, "
               f"{len(self.sections)} sections after merging")
        if moved:
            viz.table([["heading", "was", "now"]] + [[h[:60], a, b] for h, a, b in moved[:40]],
                      "Reassigned headings")
        return self

    # ------------------------------------------------------------ 3. retrieve + plan
    def fetch_related(self, use_web: bool = True) -> "Paper":
        """Pull code repo + a few web hits. Safe offline (returns nothing)."""
        gh = rank_code_repos(self.doc.text, self.links.get("github", []), self.doc.title)
        if not gh and use_web:
            hits = retrieve.search_repo_for_paper(self.doc.title)
            gh = [h["href"] for h in hits[:1]]
            if gh:
                viz.md(f"_No repo URL in paper; web search suggests_ {gh[0]}")
        if gh:
            self.repo = retrieve.github_repo(gh[0])
            if self.repo.get("readme"):
                viz.md(f"**Repo:** [{self.repo['repo']}]({self.repo['url']}) "
                       f"(★ {self.repo.get('stars', '?')}) — {self.repo.get('description') or ''}\n\n"
                       f"Fetched files: `{'`, `'.join(self.repo['files']) or 'none'}`")
            else:
                viz.md(f"_Repo {gh[0]} not reachable (offline or private)._")
        hf = self.links.get("huggingface", [])
        if hf:
            cfg = retrieve.huggingface_config(hf[0])
            if cfg.get("config"):
                self.repo["hf_config"] = cfg
                viz.md(f"**HF config** `{cfg['repo_id']}`:\n```json\n{json.dumps(cfg['config'], indent=1)[:1500]}\n```")
        if use_web:
            self.web = retrieve.web_search(self.doc.title, n=5)
            if self.web:
                viz.md("**Web context:**\n" + "\n".join(f"- [{h['title']}]({h['href']})" for h in self.web))
        return self

    def plan(self) -> "Paper":
        outline = "\n".join(f"- [{s.kind}] {s.heading} ({len(s.text)} chars)" for s in self.sections)
        abstract = by_kind(self.sections, "abstract", "front", fallback_chars=3000)[:3000]
        out = self.llm(
            f"Paper title: {self.doc.title}\n\nSection outline:\n{outline}\n\nAbstract/front matter:\n{abstract}\n\n"
            "Task: in ≤120 words, (1) state the paper type (empirical / method / survey / system / theory), "
            "(2) list which sections a reader should read first and why, "
            "(3) name the single question the reader should keep in mind. Use a short markdown list.")
        self.notes["plan"] = out
        viz.md("### Reading plan\n" + out)
        return self

    # ------------------------------------------------------------ 4. background & gap
    def background(self) -> "Paper":
        ctx = by_kind(self.sections, "abstract", "introduction", "related")
        out = self.llm(
            f"Text:\n{ctx}\n\nWrite two short sections in markdown:\n"
            "**Research background** — what problem area, what prior approaches exist (name them as the paper does).\n"
            "**Research gap** — what specifically is missing/unsolved that this paper targets. "
            "Quote or paraphrase the sentence(s) where the authors state the gap. ≤200 words total.")
        self.notes["background"] = out
        viz.md("### Background & gap\n" + out)
        return self

    # ------------------------------------------------------------ 5. motivation & method
    def method(self) -> "Paper":
        ctx = by_kind(self.sections, "introduction", "method", "experiments")
        out = self.llm(
            f"Text:\n{ctx}\n\nWrite in markdown:\n"
            "**Motivation** — why the authors chose this direction (2-3 sentences).\n"
            "**Method** — the pipeline as numbered steps: input → components → output. Name each component, "
            "its role, and any key hyper-parameters/architectural choices stated in the text. "
            "**Contributions** — bullet list, in the authors' words where possible. ≤300 words.")
        self.notes["method"] = out
        viz.md("### Motivation & method\n" + out)
        return self

    # ------------------------------------------------------------ 6. workflow / model diagram
    def workflow_diagram(self) -> "Paper":
        ctx = by_kind(self.sections, "method")[:12_000]
        code = self.llm(
            f"Method text:\n{ctx}\n\nPrevious summary:\n{self.notes.get('method', '')}\n\n"
            "Produce a Mermaid `flowchart LR` of the proposed method/workflow: data → processing → model "
            "components → training objective → outputs. Max 14 nodes, short labels (≤5 words), no quotes, "
            "no parentheses inside labels, use subgraphs for major modules. Output only the mermaid code.",
            system="You output only valid Mermaid code.")
        self.notes["workflow_mermaid"] = viz.clean_mermaid(code)
        viz.md("### Workflow")
        viz.mermaid(code)
        if self.repo.get("files") or self.repo.get("hf_config"):
            self.model_diagram()
        return self

    def model_diagram(self) -> "Paper":
        """Layer-level diagram of the model, from repo source / HF config + paper text."""
        src = "\n\n".join(f"# {p}\n{c[:2500]}" for p, c in list(self.repo.get("files", {}).items())[:4])
        hf = json.dumps(self.repo.get("hf_config", {}).get("config", {}))[:2000]
        ctx = by_kind(self.sections, "method", "experiments")[:8000]
        try:
            spec = self.llm.json(
                f"Paper method text:\n{ctx}\n\nRepository source:\n{src}\n\nHF config:\n{hf}\n\n"
                "Extract the model as an ordered list of layers/blocks. For each give: name, type, "
                "shape/dims or hyper-params if stated (e.g. d_model=512, heads=8, layers=6), and an estimated "
                "parameter count (integer, or null if unknown). Also give total_params (int or null) and source "
                "('paper'|'code'|'config'|'estimate'). JSON: {\"layers\":[{\"name\":..,\"type\":..,\"dims\":..,"
                "\"params\":..}], \"total_params\":.., \"source\":..}")
        except ValueError:
            viz.md("_Model-layer spec could not be parsed — skipping the layer diagram._")
            return self
        if not isinstance(spec, dict):
            return self
        self.notes["model_spec"] = json.dumps(spec, indent=1)
        layers = spec.get("layers", [])
        rows = [["#", "layer", "type", "dims", "params"]]
        rows += [[i + 1, l.get("name", ""), l.get("type", ""), l.get("dims", ""),
                  f"{l['params']:,}" if isinstance(l.get("params"), int) else "?"] for i, l in enumerate(layers)]
        viz.md(f"### Model layers (source: {spec.get('source', '?')}, total ≈ "
               f"{spec['total_params']:,} params)" if isinstance(spec.get("total_params"), int)
               else f"### Model layers (source: {spec.get('source', '?')})")
        viz.table(rows)
        nodes = []
        for i, l in enumerate(layers[:16]):
            lab = f"{l.get('name', 'layer')}\\n{l.get('dims') or ''}".replace('"', "").replace("(", "").replace(")", "")
            nodes.append(f'L{i}["{lab}"]')
        edges = " --> ".join(f"L{i}" for i in range(len(nodes)))
        if nodes:
            viz.mermaid("flowchart TB\n" + "\n".join(nodes) + "\n" + edges)
        viz.bars([{"label": l.get("name", ""), "value": l.get("params")} for l in layers
                  if isinstance(l.get("params"), (int, float))], title="Parameters per block", group_key=None)
        return self

    def key_figures(self, keep: int = 4) -> "Paper":
        """Pick the figures worth reproducing, from their captions.

        arXiv's HTML carries every figure — 26 for xLSTM — which is more than anyone
        wants in a summary. The captions say what each one shows, so the model can rank
        them and say why each earns its place. No figures (a PDF source) is a no-op.
        """
        figs = self.doc.figures if self.doc else []
        if not figs:
            return self
        listing = "\n".join(f"[{i}] {f['caption'][:300]}" for i, f in enumerate(figs))
        try:
            data = self.llm.json(
                f"Paper: {self.doc.title}\n\nFigure captions:\n{listing}\n\n"
                f"Pick the {keep} figures a reader should see to understand this paper — favour the "
                "architecture/method diagram and the headline results plot; skip minor or repetitive "
                "ones. For each give its index and one sentence (≤20 words) on what it shows and why "
                'it matters. JSON: {"figures":[{"index":<int>,"why":<str>}]}')
            picked = data.get("figures", []) if isinstance(data, dict) else []
        except Exception:
            picked = [{"index": i, "why": ""} for i in range(min(keep, len(figs)))]
        chosen = []
        for p in picked[:keep]:
            i = p.get("index")
            if isinstance(i, int) and 0 <= i < len(figs) and not any(c["url"] == figs[i]["url"] for c in chosen):
                chosen.append({**figs[i], "why": str(p.get("why", ""))[:200]})
        if not chosen:
            chosen = [{**f, "why": ""} for f in figs[:keep]]
        self.notes["figures_json"] = json.dumps(chosen)
        viz.md("### Key figures\n" + "\n".join(f"- **{c['caption'][:70]}** — {c['why']}" for c in chosen))
        return self

    # ------------------------------------------------------------ 7. results
    def results(self, max_tables: int = 4) -> "Paper":
        ctx = by_kind(self.sections, "results", "experiments")[:14_000]
        tbls = "\n\n".join("TABLE %d:\n%s" % (i, "\n".join(" | ".join(r) for r in t[:12]))
                           for i, t in enumerate(self.doc.tables[:max_tables]))
        try:
            data = self.llm.json(
                f"Results text:\n{ctx}\n\nExtracted tables:\n{tbls}\n\n"
                "Extract the main quantitative results as JSON: "
                "{\"summary\": <≤120 words: main findings, baselines beaten, by how much>, "
                "\"metrics\": [{\"group\": <metric name e.g. WER>, \"label\": <method/system name>, "
                "\"value\": <number>, \"higher_is_better\": <bool>, \"proposed\": <true if this row is "
                "THIS paper's own method, false for a baseline>}]}. Max 24 entries, only numbers "
                "present in the text/tables, include the proposed method AND baselines.")
        except ValueError:
            # Losing the chart is survivable; losing the eight finished stages is not.
            viz.md("_Results JSON could not be parsed — falling back to a prose summary._")
            data = {"summary": self.llm(f"Results text:\n{ctx}\n\nSummarise the main "
                                        "quantitative findings in ≤120 words: what was measured, "
                                        "which baselines were beaten, and by how much."),
                    "metrics": []}
        if not isinstance(data, dict):
            data = {"summary": "", "metrics": data if isinstance(data, list) else []}
        self.notes["results"] = data.get("summary", "")
        self.notes["results_json"] = json.dumps(data.get("metrics", []))
        viz.md("### Results\n" + data.get("summary", ""))
        for i, t in enumerate(self.doc.tables[:max_tables]):
            viz.table(t[:12], f"Table {i} (as extracted from PDF)")
        viz.bars(data.get("metrics", []), title="Key metrics")
        return self

    # ------------------------------------------------------------ 8. limitations & next gap
    def limitations(self) -> "Paper":
        ctx = by_kind(self.sections, "limitations", "discussion", "conclusion")
        out = self.llm(
            f"Text:\n{ctx}\n\nResults summary:\n{self.notes.get('results', '')}\n\nMethod summary:\n"
            f"{self.notes.get('method', '')}\n\nWrite in markdown:\n"
            "**Stated limitations** — what the authors admit (bullets).\n"
            "**Next research gap** — 3 concrete follow-up study ideas, each one line. ≤250 words.")
        self.notes["limitations"] = out
        viz.md("### Limitations & next gap\n" + out)
        return self

    # ------------------------------------------------------------ 9. conclusion
    def conclusion(self) -> "Paper":
        ctx = by_kind(self.sections, "conclusion", "abstract")[:6000]
        out = self.llm(
            f"Conclusion/abstract:\n{ctx}\n\nYour notes so far:\n"
            + "\n\n".join(f"[{k}]\n{v}" for k, v in self.notes.items() if not k.endswith(("_json", "_mermaid", "_spec")))
            + "\n\nWrite a one-paragraph plain-language conclusion (≤120 words): what was shown, "
              "what it changes, and one sentence on who should care.")
        self.notes["conclusion"] = out
        viz.md("### Conclusion\n" + out)
        return self

    # ------------------------------------------------------------ export
    def export(self, path: str | Path = "report.md") -> Path:
        path = Path(path)
        parts = [f"# {self.doc.title}\n", f"_Source: `{self.doc.path}`_\n"]
        if self.repo.get("url"):
            parts.append(f"_Code: {self.repo['url']}_\n")
        order = ["plan", "background", "method", "workflow_mermaid", "results", "limitations", "conclusion"]
        titles = {"plan": "Reading plan", "background": "Background & gap", "method": "Motivation & method",
                  "workflow_mermaid": "Workflow", "results": "Results", "limitations": "Limitations & next gap",
                  "conclusion": "Conclusion"}
        for k in order:
            if k not in self.notes:
                continue
            body = self.notes[k]
            if k == "workflow_mermaid":
                body = f"```mermaid\n{body}\n```"
            parts.append(f"\n## {titles[k]}\n\n{body}\n")
        path.write_text("\n".join(parts), encoding="utf-8")
        viz.md(f"Report written to `{path}`")
        return path

    def export_html(self, path: str | Path | None = None) -> Path:
        """Write the whole run as one self-contained HTML page (Mermaid + charts rendered)."""
        from . import report_html

        path = Path(path) if path else Path("report.html")
        out = report_html.write(self, path)
        viz.md(f"HTML report written to `{out}` — open it in a browser")
        return out

    # ------------------------------------------------------------ run all
    def run_all(self, use_web: bool = True) -> "Paper":
        self.classify_outline().fetch_related(use_web).plan().background().method()
        self.workflow_diagram().key_figures()
        return self.results().limitations().conclusion()

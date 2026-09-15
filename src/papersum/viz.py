"""Display helpers for Jupyter / VS Code notebooks: Markdown, Mermaid, tables, bars."""
from __future__ import annotations

import html
import json
import re
import uuid

from IPython.display import HTML, Markdown, display

_MERMAID_JS = "https://cdn.jsdelivr.net/npm/mermaid@11/dist/mermaid.min.js"


def _in_notebook() -> bool:
    try:
        from IPython import get_ipython

        return type(get_ipython()).__name__ == "ZMQInteractiveShell"
    except Exception:
        return False


def md(text: str):
    display(Markdown(text))


def clean_mermaid(code: str) -> str:
    """Strip code fences and obvious syntax hazards from LLM-produced mermaid."""
    code = re.sub(r"^```(?:mermaid)?\s*|```\s*$", "", code.strip(), flags=re.M).strip()
    if not re.match(r"^(graph|flowchart|sequenceDiagram|classDiagram|stateDiagram)", code):
        code = "flowchart TD\n" + code
    # Quotes and parentheses inside a [label] both break the parser, and models emit them
    # anyway ("Embedding (d=512)") however firmly the prompt asks them not to. Keep the
    # text, drop the offending punctuation, so the diagram still renders.
    def _label(m):
        inner = m.group(1).translate(str.maketrans("", "", '"()'))
        return "[" + re.sub(r"\s{2,}", " ", inner).strip() + "]"

    return re.sub(r"\[([^\[\]]*)\]", _label, code)


def mermaid(code: str, height: int | None = None):
    """Render a mermaid diagram inline (works in Jupyter, JupyterLab, VS Code)."""
    code = clean_mermaid(code)
    uid = "m" + uuid.uuid4().hex[:8]
    esc = html.escape(code)
    display(HTML(f"""
<div id="{uid}" style="max-width:100%;overflow-x:auto"><pre class="mermaid">{esc}</pre></div>
<script type="module">
  import mermaid from "{_MERMAID_JS}";
  mermaid.initialize({{ startOnLoad: false, theme: "neutral", securityLevel: "loose" }});
  const el = document.querySelector("#{uid} .mermaid");
  try {{ await mermaid.run({{ nodes: [el] }}); }}
  catch (e) {{ el.style.whiteSpace = "pre"; el.textContent = "mermaid error: " + e + "\\n\\n" + el.textContent; }}
</script>"""))
    display(Markdown("<details><summary>mermaid source</summary>\n\n```mermaid\n" + code + "\n```\n</details>"))


def table(rows: list[list[str]], caption: str = ""):
    if not rows:
        return
    head, *body = rows
    parts = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    parts += ["| " + " | ".join(str(c) for c in r) + " |" for r in body]
    md((f"**{caption}**\n\n" if caption else "") + "\n".join(parts))


def bars(items: list[dict], title: str = "", value_key: str = "value", label_key: str = "label",
         group_key: str | None = "group"):
    """Horizontal grouped bar chart from [{label, value, group?}] dicts."""
    import matplotlib

    # Outside a notebook (CLI, pytest) the default backend here is interactive (tkagg)
    # and plt.show() blocks until a human closes the window — so `papersum x.pdf` hangs.
    # Nothing can display the figure in that context anyway, so drop to a headless canvas.
    if not _in_notebook() and matplotlib.get_backend().lower() != "agg":
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    items = [i for i in items if isinstance(i.get(value_key), (int, float))]
    if not items:
        md("_no numeric values to plot_")
        return
    groups = sorted({i.get(group_key, "") for i in items}) if group_key else [""]
    fig, axes = plt.subplots(1, len(groups), figsize=(5 * len(groups), max(2.5, 0.4 * len(items) / len(groups) + 1)), squeeze=False)
    for ax, g in zip(axes[0], groups):
        sub = [i for i in items if not group_key or i.get(group_key, "") == g]
        ax.barh([str(i[label_key])[:40] for i in sub], [i[value_key] for i in sub], color="#4a7ebb")
        ax.set_title(g or title, fontsize=10)
        ax.invert_yaxis()
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for y, i in enumerate(sub):
            ax.text(i[value_key], y, f" {i[value_key]:g}", va="center", fontsize=8)
    fig.suptitle(title if group_key and len(groups) > 1 else "", fontsize=11)
    fig.tight_layout()
    if _in_notebook():
        plt.show()
    else:
        plt.close(fig)

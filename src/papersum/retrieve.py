"""External retrieval: GitHub repo structure, HuggingFace configs, web search.

All functions are best-effort and return empty results (never raise) when offline.
"""
from __future__ import annotations

import os
import re

import requests

GH_API = "https://api.github.com"
_HEADERS = {"Accept": "application/vnd.github+json"}
if os.getenv("GITHUB_TOKEN"):
    _HEADERS["Authorization"] = f"Bearer {os.environ['GITHUB_TOKEN']}"

MODEL_FILE_RE = re.compile(r"(model|net|arch|layer|module|config|backbone|encoder|decoder)", re.I)
CONFIG_EXT = (".json", ".yaml", ".yml", ".toml")


def _get(url, headers=None, **kw):
    try:
        r = requests.get(url, headers=headers or _HEADERS, timeout=20, **kw)
        if r.ok:
            return r
    except requests.RequestException:
        pass
    return None


def github_repo(url: str, max_files: int = 6, max_chars: int = 4000) -> dict:
    """Return README + a few model/config source files from a GitHub repo."""
    m = re.search(r"github\.com/([^/\s]+)/([^/\s#?]+)", url)
    if not m:
        return {}
    owner, repo = m.group(1), m.group(2).removesuffix(".git")
    out = {"repo": f"{owner}/{repo}", "url": f"https://github.com/{owner}/{repo}",
           "readme": "", "files": {}, "tree": []}

    branch = "HEAD"
    r = _get(f"{GH_API}/repos/{owner}/{repo}")
    if r:
        branch = r.json().get("default_branch", "HEAD")
        out["stars"] = r.json().get("stargazers_count")
        out["description"] = r.json().get("description")

    r = _get(f"{GH_API}/repos/{owner}/{repo}/readme", headers={**_HEADERS, "Accept": "application/vnd.github.raw"})
    if not r:  # API rate-limited? raw fallback needs no auth
        for name in ("README.md", "readme.md", "README.rst", "README"):
            r = _get(f"https://raw.githubusercontent.com/{owner}/{repo}/{branch}/{name}")
            if r:
                break
    if r:
        out["readme"] = r.text[:max_chars]

    r = _get(f"{GH_API}/repos/{owner}/{repo}/git/trees/{branch}", params={"recursive": "1"})
    if not r:
        return out
    paths = [t["path"] for t in r.json().get("tree", []) if t["type"] == "blob"]
    out["tree"] = paths[:400]

    candidates = [p for p in paths if p.endswith(".py") and MODEL_FILE_RE.search(p)
                  and "test" not in p.lower() and p.count("/") <= 3]
    candidates += [p for p in paths if p.endswith(CONFIG_EXT) and MODEL_FILE_RE.search(p) and p.count("/") <= 2]
    for p in candidates[:max_files]:
        raw = _get(f"https://raw.githubusercontent.com/{owner}/{repo}/{branch}/{p}")
        if raw:
            out["files"][p] = raw.text[:max_chars]
    return out


def huggingface_config(url: str) -> dict:
    m = re.search(r"huggingface\.co/([^/\s]+/[^/\s#?]+)", url)
    if not m:
        return {}
    repo_id = m.group(1)
    r = _get(f"https://huggingface.co/{repo_id}/raw/main/config.json")
    return {"repo_id": repo_id, "config": r.json()} if r else {"repo_id": repo_id}


def web_search(query: str, n: int = 5) -> list[dict]:
    """DuckDuckGo search via `ddgs` (pip install ddgs). Returns [{title, href, body}]."""
    try:
        from ddgs import DDGS
    except ImportError:
        try:
            from duckduckgo_search import DDGS  # older package name
        except ImportError:
            return []
    try:
        with DDGS() as d:
            return list(d.text(query, max_results=n))
    except Exception:
        return []


def search_repo_for_paper(title: str) -> list[dict]:
    hits = web_search(f"{title} github", n=5)
    return [h for h in hits if "github.com" in h.get("href", "")]
